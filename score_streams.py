"""Command line entry point for the whole workflow.

    python score_streams.py --config config/config.yaml

Runs every stage from raw files to report. Despite the name - which matches the project brief -
this is not just the scoring stage; use ``--until score`` to stop there.

Why this exists rather than calling snakemake directly
-----------------------------------------------------
It is a front door to the same Snakefile, not a second implementation, so nothing here can
disagree with what the workflow does. What it adds is the handful of things that are awkward
about driving Snakemake by hand:

``--set detect.method=isolation_forest``
    Snakemake's own ``--config`` only reaches top-level keys, and its nested form parses each
    override independently - which turns a nested ``null`` into the four characters ``"null"``
    and has already cost this project one debugging session. Overrides here are merged into a
    loaded copy of the config and written to a temporary file, so a value means what YAML says
    it means.

``--summary``
    Prints what the run found and what it cost, so the headline numbers do not require opening
    a JSON file. ``--summary`` alone reports the last run without recomputing anything.

``--unlock``
    Surfaces the recovery for a directory left locked by an interrupted run, which is otherwise
    a thing you have to know to look for.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT / "config" / "config.yaml"

#: Stages `--until` and `--only` accept, in the order the workflow runs them.
STAGES = ("download", "canonicalise", "umap", "preprocess", "window", "corpus", "normalise",
          "reduce", "covariance", "index", "forest", "calibrate", "score", "evaluate", "report")


def parse_override(text):
    """Turn ``detect.method=isolation_forest`` into ``{"detect": {"method": ...}}``.

    The value is parsed as YAML, so ``3`` is an int, ``0.5`` a float, ``true`` a bool and
    ``null`` a None - rather than all of them being strings, which is the failure mode of
    passing nested overrides through Snakemake's own ``--config``.
    """
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "expected key=value, got {0!r} - for example detect.neighbours=3".format(text))

    key, _, raw = text.partition("=")
    value = yaml.safe_load(raw)

    nested = value
    for part in reversed(key.strip().split(".")):
        nested = {part: nested}
    return nested


def deep_merge(base, updates):
    """Recursively merge `updates` into `base`, returning a new dict."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(path, overrides, dataset=None, labels=None):
    """Load the config, apply overrides, and return ``(config, path_to_use)``.

    When nothing is overridden the original file is passed through untouched, so an ordinary
    run leaves no temporary files and the config Snakemake reports is the one on disk.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit("no config at {0}".format(path))

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for override in overrides or []:
        config = deep_merge(config, override)
    if dataset:
        config = deep_merge(config, {"dataset": dataset})
    if labels:
        config = deep_merge(config, {"evaluate": {"labels": labels}})

    if not overrides and not dataset and not labels:
        return config, path

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="anomalies_config_", delete=False, encoding="utf-8")
    with handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config, Path(handle.name)


def snakemake_command(config_path, args):
    """The argv this run turns into. Assembled rather than executed, so `--print` can show it."""
    command = [sys.executable, "-m", "snakemake",
               "--configfile", str(config_path),
               "--cores", str(args.cores)]

    if args.dry_run:
        command.append("--dry-run")
    if args.unlock:
        command.append("--unlock")
    if args.forcerun:
        command += ["--forcerun"] + list(args.forcerun)
    if args.until:
        command += ["--until", args.until]
    if args.only:
        command += ["--allowed-rules", args.only, "--until", args.only]
    if args.rulegraph:
        command.append("--rulegraph")
    if args.extra:
        command += list(args.extra)
    return command


def summary_path(config):
    return PROJECT / "results" / "{0}_summary.json".format(config["dataset"])


def print_summary(config):
    """What the run found, and what it cost, without opening a JSON file."""
    path = summary_path(config)
    if not path.exists():
        print("\nno summary at {0} - has the pipeline run?".format(path))
        return

    metrics = json.loads(path.read_text(encoding="utf-8"))
    print("\n{0}".format(config["dataset"]))
    print("  detector      {0}{1}".format(
        config["detect"]["method"],
        "" if config["detect"]["method"] == "mahalanobis"
        else " ({0})".format(config["detect"]["forest"]["space"])))

    if metrics.get("f1") is not None:
        print("  quality       P {precision:.3f}   R {recall:.3f}   F1 {f1:.3f}   "
              "adjusted F1 {adjusted_f1:.3f}".format(**metrics))
        print("  flagged       {flagged_fraction:.2%} of {n_points:,} point(s), against a true "
              "rate of {anomaly_rate:.2%}".format(**metrics))

    throughput = metrics.get("throughput") or {}
    counts, rates = throughput.get("counts", {}), throughput.get("rates", {})
    if counts:
        print("  corpus        {0:,} interval(s) x {1} term(s), rank {2}".format(
            counts.get("corpus_intervals", 0), counts.get("signature_terms", 0),
            counts.get("retained_rank", 0)))
    if rates:
        parts = []
        if rates.get("points_scored_per_second"):
            parts.append("{0:,.0f} point(s)/s".format(rates["points_scored_per_second"]))
        if rates.get("queries_per_second"):
            parts.append("{0:,.0f} quer(y|ies)/s".format(rates["queries_per_second"]))
        if rates.get("peak_rss_mb"):
            parts.append("peak {0:,.0f} MB".format(rates["peak_rss_mb"]))
        if parts:
            print("  throughput    {0}".format("   ".join(parts)))
    print("\n  full results  {0}".format(path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="score_streams.py",
        description="Run the signature anomaly-detection workflow end to end.",
        epilog="examples:\n"
               "  python score_streams.py\n"
               "  python score_streams.py --dataset cmapss_fd002 --cores 4\n"
               "  python score_streams.py --set detect.method=isolation_forest -n\n"
               "  python score_streams.py --summary",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="workflow configuration (default: config/config.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", type=parse_override,
                        metavar="KEY=VALUE",
                        help="override one config value, dotted; repeatable. The value is read "
                             "as YAML, so 3 is an int and null is None")
    parser.add_argument("--dataset", help="shorthand for --set dataset=NAME")
    parser.add_argument("--labels", help="shorthand for --set evaluate.labels=PATH")
    parser.add_argument("--cores", default=os.cpu_count() or 1,
                        help="cores for Snakemake (default: every core on this machine)")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="show what would run, and run nothing")
    parser.add_argument("--until", choices=STAGES, help="stop after this stage")
    parser.add_argument("--only", choices=STAGES, help="run only this stage, if its inputs exist")
    parser.add_argument("--forcerun", nargs="+", metavar="RULE",
                        help="rerun these rules even if their outputs look current. Needed when "
                             "a config value read inside a run: block changes, which Snakemake "
                             "does not track")
    parser.add_argument("--unlock", action="store_true",
                        help="release a working directory left locked by an interrupted run")
    parser.add_argument("--rulegraph", action="store_true",
                        help="print the rule graph as DOT and exit; pipe to `dot -Tpng`")
    parser.add_argument("--summary", action="store_true",
                        help="print the last run's results; on its own, runs nothing")
    parser.add_argument("--print", dest="print_command", action="store_true",
                        help="show the snakemake command this would run, and exit")
    parser.add_argument("extra", nargs="*",
                        help="further arguments passed straight to snakemake")

    args = parser.parse_args(argv)
    config, config_path = resolve_config(args.config, args.overrides, args.dataset, args.labels)

    if args.summary and not any((args.dry_run, args.unlock, args.forcerun, args.rulegraph)):
        print_summary(config)
        return 0

    command = snakemake_command(config_path, args)
    if args.print_command:
        print(subprocess.list2cmdline(command))
        return 0

    result = subprocess.run(command, cwd=str(PROJECT))
    if result.returncode:
        # Snakemake's own message has already been printed; add the one recovery that is not
        # obvious from it.
        if not args.unlock:
            print("\nif the directory is locked by an interrupted run: "
                  "python score_streams.py --unlock", file=sys.stderr)
        return result.returncode

    if not args.dry_run and not args.rulegraph and not args.unlock:
        print_summary(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
