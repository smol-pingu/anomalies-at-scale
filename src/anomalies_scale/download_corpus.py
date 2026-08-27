"""Fetch the raw files a run needs, and decide what those files are called.

The corpus of normality and the data to be scored are configured as two separate lists of
URLs. That split is not cosmetic: labelled benchmarks generally ship the two as distinct
contiguous periods - SMD gives ``train/`` and ``test/`` as separate directories per machine -
and cutting a test period out of the corpus series is only needed for datasets that do not.

Either list may hold any number of URLs, so a dataset spread over many files (one per
machine, say) is expressed directly rather than being concatenated up front.

This module owns two things the workflow needs at different moments:

**Naming**, via :func:`raw_targets` and :func:`raw_files`. Snakemake has to know at DAG
construction which files a run will produce, before anything is fetched, so resolving URLs
onto filenames has to be a pure function of the configuration.

**Fetching**, via :func:`download_file`. Downloads stream to a temporary sibling and are
renamed into place only on success. Snakemake treats an existing output as a finished one, so
a half-written file left by an interrupted run would otherwise be indistinguishable from a
complete download and would silently poison every stage downstream.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

import requests

#: The two sides a raw file can belong to.
SIDES = ("corpus", "test")

#: Where raw files live, as ``<root>/<dataset>/<side>/<name>``.
DEFAULT_ROOT = Path("data/raw")

#: Streaming chunk size, in bytes.
CHUNK_SIZE = 1 << 20


def target_name(url, side, position):
    """Filename a URL is downloaded to.

    Uses the URL's last path segment so downloaded files stay recognisable, ignoring any
    query string. Falls back to a positional name when that segment is empty or has already
    been claimed - several machines in one dataset can easily share a filename, and silently
    overwriting one with another would be worse than an ugly name.
    """
    name = Path(urlsplit(str(url)).path).name
    if not name:
        return "{0}_{1:03d}.csv".format(side, position)
    return name


def raw_targets(urls, side):
    """Map each URL onto the filename it is downloaded to.

    Parameters
    ----------
    urls : iterable of str
        Configured URLs for this side. A bare string is accepted as a single URL.
    side : str
        ``'corpus'`` or ``'test'``, used only to name files that cannot be named from
        their URL.

    Returns
    -------
    dict
        ``{filename: url}``, in configuration order.
    """
    if isinstance(urls, str):
        urls = [urls]
    urls = list(urls or [])

    targets = {}
    for position, url in enumerate(urls):
        name = target_name(url, side, position)
        if name in targets:
            name = "{0}_{1:03d}{2}".format(side, position, Path(name).suffix or ".csv")
        targets[name] = str(url)
    return targets


def raw_files(urls, side, dataset, root=DEFAULT_ROOT):
    """Paths of the raw files expected for one side.

    One path per configured URL. When no URLs are configured this instead reports whatever
    is already in the directory, so a dataset placed there by hand drives the rest of the
    pipeline without needing to be described in the configuration.

    An empty result for ``side='test'`` is meaningful rather than an error: it says the test
    period has to be cut out of the corpus series instead.
    """
    directory = Path(root) / dataset / side
    targets = raw_targets(urls, side)
    if targets:
        return [str(directory / name) for name in sorted(targets)]
    if directory.is_dir():
        return sorted(str(path) for path in directory.iterdir() if path.is_file())
    return []


def download_file(path, url, timeout=60, chunk_size=CHUNK_SIZE):
    """Download one URL to one path, atomically.

    Streams rather than buffering the whole response, since these lists are meant to carry
    whole datasets. Writes to ``<path>.part`` and renames on success, so an interrupted or
    failed download never leaves a file that later stages would mistake for complete.

    Returns
    -------
    int
        Bytes written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        written = 0
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:                      # keep-alive chunks are empty
                    handle.write(chunk)
                    written += len(chunk)

        if written == 0:
            raise IOError("{0} returned an empty body".format(url))

        partial.replace(path)                  # atomic, and overwrites on Windows
        return written
    finally:
        if partial.exists():
            partial.unlink()


def download_target(path, urls, side, name, **kwargs):
    """Download the single file `name` of `side`, resolving it against `urls`.

    This is what the workflow calls: Snakemake asks for one output path at a time, and the
    URL it corresponds to has to be recovered from the configuration by filename.

    Raises
    ------
    ValueError
        If no configured URL produces this filename - the file is neither described in the
        configuration nor already on disk. ValueError rather than KeyError so the workflow
        prints the explanation on its own lines instead of as an escaped repr.
    """
    targets = raw_targets(urls, side)
    if name not in targets:
        raise ValueError(
            "no URL configured for {0!r} on the {1!r} side.\n"
            "  Configured there: {2}\n"
            "  Either add its URL to raw.{1}_urls, or place the file at {3} yourself."
            .format(name, side, sorted(targets) or "nothing", path))
    return download_file(path, targets[name], **kwargs)


def download_side(urls, side, dataset, root=DEFAULT_ROOT, show_progress=True, **kwargs):
    """Download every file configured for one side. Used by the command line, not the workflow.

    Skips files that are already present, so an interrupted bulk download resumes cheaply.
    """
    directory = Path(root) / dataset / side
    targets = raw_targets(urls, side)
    written = {}

    for number, (name, url) in enumerate(sorted(targets.items()), start=1):
        destination = directory / name
        if destination.exists():
            if show_progress:
                print("[{0}/{1}] {2} already present".format(number, len(targets), name))
            continue
        written[name] = download_file(destination, url, **kwargs)
        if show_progress:
            print("[{0}/{1}] {2} <- {3} ({4:,} bytes)".format(
                number, len(targets), name, url, written[name]))

    return written


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch raw corpus and test files for a dataset.")
    parser.add_argument("--url", action="append", dest="urls", default=[],
                        help="URL to fetch; repeatable")
    parser.add_argument("--side", choices=SIDES, default="corpus",
                        help="which side these URLs belong to (default corpus)")
    parser.add_argument("--output", default=None,
                        help="download a single URL to exactly this path; requires one --url")
    parser.add_argument("--dataset", default=None,
                        help="dataset name, used as <root>/<dataset>/<side>/")
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="root directory for raw files (default data/raw)")
    parser.add_argument("--list", action="store_true",
                        help="print the filenames these URLs resolve to and exit")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")

    args = parser.parse_args(argv)

    if args.list:
        for name, url in sorted(raw_targets(args.urls, args.side).items()):
            print("{0}\t{1}".format(name, url))
        return

    if args.output is not None:
        if len(args.urls) != 1:
            parser.error("--output takes exactly one --url")
        written = download_file(args.output, args.urls[0])
        if not args.quiet:
            print("{0} <- {1} ({2:,} bytes)".format(args.output, args.urls[0], written))
        return

    if args.dataset is None:
        parser.error("--dataset is required unless --output is given")
    if not args.urls:
        parser.error("no --url given")

    download_side(args.urls, args.side, args.dataset, root=args.root,
                  show_progress=not args.quiet)


if __name__ == "__main__":
    main()
