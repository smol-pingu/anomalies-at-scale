"""Anomalies-at-scale workflow - skeleton.

    download -> [umap] -> preprocess -> [window] -> corpus -> covariance -> index
                                                                                |
                                            report <-- evaluate <-- score <-- calibrate

`window` is in the DAG whenever `window.size` is set, and cuts every stream to one fixed length
before anything is signed. The dyadic grid is relative - `granularity` counts cells, not points -
so without it a single depth band holds intervals of wildly different lengths and a query is
compared against whatever happens to share its band.

`umap` is in the DAG whenever `umap.dimension` is positive, and reduces the channel count before
anything is signed - signature dimension is roughly ``d ** trunc``, so the channel count is the
base of an exponent and the only lever that acts on it.

It is **pure preprocessing**. One projection is fitted on corpus data alone, then *both* the
corpus and the test streams are transformed into its latent space, and every stage after it
reads latent streams without knowing a reduction happened. Nothing downstream takes a
projection as input, and `score` in particular does not - its test windows arrive already
latent, so applying one there would reduce an already-reduced stream.

Both sides passing through the same fitted object is the whole requirement: signatures of paths
in different coordinate systems are not comparable, and a Mahalanobis distance between them
means nothing.

The workflow implements the **pooled** detector: one corpus of every dyadic interval of every
corpus window, whitened once by a single global ``Sigma^-1/2`` and searched by depth band, so
scoring a candidate interval is one nearest-neighbour query. `config["detect"]["mode"]`
selects between that and the per-interval alternative; only `pooled` is wired here.

Calibration is by k-fold over streams, which is why `preprocess` emits two stream sets rather
than three. The metric and the index are built from the whole corpus; `calibrate` then rebuilds
a reference set per fold, so every interval is scored against a corpus that does not contain its
own stream. The fixed fit/withheld holdout that used to serve this gave the same guarantee to
only a fifth of the corpus.
"""

import re
import sys
from pathlib import Path

from snakemake.exceptions import WorkflowError

configfile: "config/config.yaml"

# Make the package importable without an install step. Done before the imports below,
# which is why they are not at the top of the file.
sys.path.insert(0, str(Path(workflow.basedir) / "src"))

from anomalies_scale.canonical_streams import canonicalise_files, write_canonical
from anomalies_scale.covariance_creation import create_covariance, read_corpus
from anomalies_scale.crossvalidated_thresholding import crossvalidated_thresholds
from anomalies_scale.download_corpus import download_target, raw_files
from anomalies_scale.evaluation import build_report, evaluate_scores
from anomalies_scale.index_creation import build_index, load_index
from anomalies_scale.isolation_detector import (
    DepthForests, crossvalidated_forest_thresholds, fit_forests)
from anomalies_scale.normalisation import Normaliser, normalise_corpus
from anomalies_scale.preprocessing import preprocess_streams
from anomalies_scale.AUC import operating_point_auc, score_points, sweep_operating_points
from anomalies_scale.bagging import run_bagged
from anomalies_scale.signature_computer import compute_corpus
from anomalies_scale.throughput import collect as collect_throughput, discover_benchmarks
from anomalies_scale.stream_evaluation import evaluate_predictions, plot_scored_streams
from anomalies_scale.stream_scoring import score_streams
from anomalies_scale.umap_projection import reduce_signature_corpus
from anomalies_scale.windowing import resolve_size, window_streams


def latent_dimension(value):
    """Read `umap.dimension`: -1 for off, or the size of the latent space.

    Rejects 0 rather than reading it as "off". One number carrying both the switch and the
    size is only unambiguous if the off value is unmistakable, and a zero-dimensional latent
    space is not something anyone means to request - so it is far more likely a typo for -1 or
    for a real dimension, and silently guessing which would be worse than stopping.
    """
    dimension = int(value)
    if dimension == 0 or dimension < -1:
        raise WorkflowError(
            "umap.dimension is {0}; expected -1 to switch the reduction off, or a positive "
            "number of latent channels.".format(dimension))
    return None if dimension < 0 else dimension


#: The latent dimension, or None when the reduction is off. Read once, because it decides both
#: what `preprocess` reads and whether `score` is handed a projection - and those must agree.
UMAP_DIMENSION = latent_dimension(config.get("umap", {}).get("dimension", -1))


# Imported only when the reduction is on: the module pulls in umap-learn, and through it numba
# and llvmlite, which cost seconds to import and are an unnecessary dependency for every run
# that leaves the reduction switched off.
if UMAP_DIMENSION is not None:
    from anomalies_scale.umap_projection import project_corpus_and_test

DATASET = config["dataset"]

#: Pin the `dataset` wildcard to the one being built. Without this, `{dataset}` is a greedy
#: match and any artifact whose name extends another's becomes ambiguous - `score` writes
#: `{dataset}_scores.parquet`, which also matches `point_scores`'s output with `dataset` read as
#: `..._point`. Snakemake reports that as an AmbiguousRuleException naming two rules that have
#: nothing to do with each other, which is a confusing way to learn that a filename collided.
#: The workflow builds exactly one dataset per run, so constraining it to that name is exact.
wildcard_constraints:
    dataset=re.escape(DATASET),

#: The two stream sets: `corpus` is assumed normal and is what the detector is built from,
#: `test` is what gets scored.
#:
#: There used to be three. `fit` built the metric and `withheld` calibrated the threshold
#: against normal data the metric had not seen, because a threshold taken from the corpus's own
#: nearest-neighbour distances understates what unseen data scores by ~13x. `calibrate` now
#: answers that by k-fold over streams, which gives every interval that guarantee instead of the
#: fifth of them that happened to be withheld, so the fixed holdout has been removed.
SPLITS = ["corpus", "test"]

#: Raw files land one per URL, under a directory naming which side they belong to. The two
#: sides arrive separately because that is how labelled benchmarks ship: SMD gives train/ and
#: test/ as distinct contiguous periods, and cutting a test period out of the corpus series
#: is only needed when a dataset does not.
RAW = "data/raw/{dataset}/{side}/{name}"

#: One side's whole set of streams, canonicalised: one row per stream, `stream` / `Time` /
#: `Stream`, with values as a (length, k) block. Every dataset shape this project reads -
#: SMD's directory of headerless files, C-MAPSS's single file with a unit column, NAB's one
#: timestamped CSV - collapses to this here, so nothing downstream has to know which it was.
CANONICAL = "data/streams/{dataset}_{side}.parquet"
CANONICAL_CORPUS = CANONICAL.replace("{side}", "corpus")
CANONICAL_TEST = CANONICAL.replace("{side}", "test")

#: Both sides in latent coordinates, one parquet per stream, plus the fitted projection.
#: Only built when `umap.dimension` is positive.
#:
#: Both sides are projected here rather than only the corpus, because the reduction is pure
#: preprocessing: everything downstream reads latent streams and never learns that a reduction
#: happened. The projection is still persisted - it is what was learnt, it is expensive, and a
#: rerun should not have to refit it.
PROJECTED_CORPUS = "data/projected/{dataset}_corpus.parquet"
PROJECTED_TEST = "data/projected/{dataset}_test.parquet"
PROJECTION = "data/projected/{dataset}_projection.pkl"

WINDOWS = "data/preprocessed/{dataset}_{split}.parquet"
CORPUS_STREAMS = WINDOWS.replace("{split}", "corpus")
TEST_STREAMS = WINDOWS.replace("{split}", "test")

#: The same three sets cut into fixed-length windows, when `window.size` is set. Each window is
#: a stream in its own right, named `{source}_{n}`, and carries `source` / `window` / `offset`
#: alongside the canonical columns so the operation can be undone at evaluation time.
#:
#: The size is in the filename because Snakemake tracks files, not the config values a `run:`
#: block reads. Without it, changing `window.size` would leave the previous size's windows in
#: place, judged up to date, and the whole run would quietly use them.
WINDOWED = "data/preprocessed/{dataset}_{split}_w{size}.parquet"

#: What `preprocess` decided, as readable JSON beside the three sets it wrote. Not an output
#: of the rule: making it one would mean Snakemake deleted it on failure, and the case where
#: you most want to know which streams were discarded is the run that then fell over.
PREPROCESS_SUMMARY = "data/preprocessed/{dataset}_preprocess.json"

#: The fitted per-component normalisation, when `signature.normalise` is on. Saved because the
#: test signatures must be put in the same units as the corpus, and a normaliser refitted on
#: test data would place the two sides in different ones.
NORMALISER = "data/corpus/{dataset}_normaliser.json"

#: The signature corpus: every grid interval of every corpus stream. One file, since there is
#: no longer a split to keep apart - the metric, the index and the calibration all read the same
#: rows, and which of them a given row may influence is decided by `calibrate`'s folds rather
#: than by which file it was written to.
CORPUS = "data/corpus/{{dataset}}_corpus.{0}".format(config["signature"]["format"])
#: The metric, as a dense matrix carrying its term names. `metric.file_format` picks the
#: layout - npz by default, csv when the numbers are worth eyeballing without numpy. The suffix
#: is what `covariance_creation` dispatches on, so the two stay in step automatically, and a
#: matrix already on disk in the other form still loads.
METRIC_FILE_FORMAT = config["metric"].get("file_format", "npz")
if METRIC_FILE_FORMAT not in ("npz", "csv"):
    raise WorkflowError(
        "metric.file_format is {0!r}; expected 'npz' or 'csv'. Note this is not metric.form, "
        "which chooses between pinv and inv_sqrt.".format(METRIC_FILE_FORMAT))
COVARIANCE = "data/corpus/{{dataset}}_covariance.{0}".format(METRIC_FILE_FORMAT)

#: Whether the metric is stored as its rank-r factor and searched in r latent coordinates
#: rather than D signature terms. Exact - the projection is an isometry - and worth r/D of the
#: index memory and search cost. Only `npz` can carry a factor; csv falls back to the dense form.
FACTORED_METRIC = bool(config["metric"].get("factored", False))
if FACTORED_METRIC and METRIC_FILE_FORMAT != "npz":
    raise WorkflowError(
        "metric.factored is true but metric.file_format is {0!r}. The factor is two arrays and "
        "only npz carries it; a csv would have to store the matrix multiplied out, which is the "
        "thing being avoided. Use npz, or set factored to false.".format(METRIC_FILE_FORMAT))
COVARIANCE_DIAGNOSTICS = "data/corpus/{dataset}_covariance.json"
#: An orthonormal basis of the corpus's numerically non-zero row space - a few hundred
#: directions, not the eleven `variance_keep` retains. Only the off-manifold test reads it, and
#: it needs this cut rather than the metric's: a direction the corpus has little variance in is
#: unusual, one it has none in is unrepresentable.
SUBSPACE = "data/corpus/{dataset}_subspace.npz"

#: The corpus rewritten in a latent signature space, when `signature_umap.dimension` is
#: positive. Still a corpus - same identity columns, fewer `sig_*` ones - so `covariance`,
#: `index` and `calibrate` read it without knowing anything changed.
REDUCED_CORPUS = "data/corpus/{{dataset}}_reduced.{0}".format(
    config["signature"]["format"])
SIGNATURE_PROJECTION = "data/corpus/{dataset}_signature_projection.pkl"
SIGNATURE_PROJECTION_DIAGNOSTICS = "data/corpus/{dataset}_signature_projection.json"
#: The searchable set: one flat L2 index over every whitened corpus interval, plus the row
#: metadata (depth, split, interval identity) without which it is only a raw lookup.
INDEX = "data/index/{dataset}_index.faiss"
INDEX_META = "data/index/{dataset}_index.npz"
#: The isolation forests, one per depth band, when `detect.method` selects them. An
#: alternative to INDEX rather than an addition: whichever the method names is built, and
#: the other never enters the DAG.
FOREST = "data/index/{dataset}_forest.pkl"
#: One threshold per interval width, not a scalar: a level-k signature term scales like
#: (increment)**k, so wide intervals produce far larger vectors than narrow ones and a single
#: number would behave partly as a width filter rather than an anomaly test.
PROJECTION_DIAGNOSTICS = "data/projected/{dataset}_projection.json"
THRESHOLDS = "data/index/{dataset}_thresholds.csv"
SCORES = "results/{{dataset}}_scores.{0}".format(config["signature"]["format"])
#: The per-interval verdict, the per-stream roll-up, and one figure per stream drawn.
INTERVALS = "results/{{dataset}}_intervals.{0}".format(config["signature"]["format"])
#: Metrics as JSON, not a table. `evaluate` writes a flat dict of scalars - precision,
#: recall, F1, the baselines - and `build_report` dispatches on this suffix, so calling
#: it .parquet would hand JSON text to pd.read_parquet.
SUMMARY = "results/{dataset}_summary.json"
#: What each stage cost, one TSV per job, written by Snakemake's own `benchmark:` directive:
#: wall-clock seconds, peak resident memory, CPU seconds, mean load and I/O. `{stage}` carries
#: each rule's own wildcards where it has any, since two jobs of one rule must not share a file.
#:
#: Times alone are not throughput - a second is only meaningful against the work it bought - so
#: `evaluate` pairs these with the counts in `SCORE_STATS` and the covariance diagnostics, and
#: writes the derived rates into the summary. `download` and `report` are deliberately not
#: benchmarked: neither scales with the data, so neither belongs on a throughput curve.
BENCHMARK_DIR = "benchmarks"
BENCHMARK = BENCHMARK_DIR + "/{{dataset}}_{stage}.tsv"
#: Counts from the scoring pass - streams, points, and the number of interval queries the
#: widest-first search actually issued. The last is the denominator for queries per second and
#: is data-dependent: the search only bisects where a test fails, so dirty streams cost more.
SCORE_STATS = "results/{dataset}_score_stats.json"

#: One continuous score per point, from a fixed sliding window - what ROC-AUC and PR-AUC rank.
#: A separate stage from `score` rather than an extra output of it, so the two costs stay
#: separable in the benchmarks: `score` issues a few thousand adaptive queries, this one issues
#: a fixed batch per stream, and merging them into one timing would waste the distinction.
POINT_SCORES = "results/{dataset}_point_scores.parquet"

#: Feature bagging. When on, `score` is replaced by `bagged`, which builds a corpus, metric,
#: calibration and search per channel subset and combines the verdicts by vote. `corpus`,
#: `covariance`, `index` and `calibrate` leave the DAG with it: a draw's artifacts are
#: meaningless to any other draw, so there is nothing shared to build once.
BAGGED = config["detect"].get("bagged") or {}
BAGGED_DRAWS = int(BAGGED.get("draws", -1) or -1)
USE_BAGGING = BAGGED_DRAWS > 0
if BAGGED_DRAWS == 0 or (BAGGED_DRAWS < 0 and BAGGED_DRAWS != -1):
    raise WorkflowError(
        "detect.bagged.draws is {0}; use -1 to switch bagging off, or a positive number of "
        "draws.".format(BAGGED_DRAWS))

#: Per-draw diagnostics, and the vote fraction per point. The latter is a continuous score in
#: [0, 1] and so is what the ranking metrics read - a bagged run needs no separate sliding-window
#: pass, because the vote count already is one.
BAGGED_DIAGNOSTICS = "results/{dataset}_bagged.json"

# `bagged` and `score` both write SCORES, and `bagged` and `point_scores` both write
# POINT_SCORES - only one of each pair is ever wanted, but both rules exist in the file either
# way, so Snakemake needs telling which wins. Declared in both directions so the losing rule is
# unreachable rather than merely deprioritised.
if USE_BAGGING:
    ruleorder: bagged > score
    ruleorder: bagged > point_scores
else:
    ruleorder: score > bagged
    ruleorder: point_scores > bagged

#: Whether the detector is scored against ground truth at all. Off, the pipeline still detects
#: and still reports what it found; it simply measures nothing, and the two stages that exist
#: only to feed metrics leave the DAG with it.
EVALUATE_METRICS = bool(config["evaluate"].get("metrics", True))
if EVALUATE_METRICS and not config["evaluate"].get("labels"):
    raise WorkflowError(
        "evaluate.metrics is true but evaluate.labels is null. There is nothing to measure "
        "against. Give labels, or set evaluate.metrics to false to run detection alone.")

#: Increments the ranking window spans, or None to leave the stage out of the DAG entirely -
#: the same switch-off convention as `umap.dimension` and `window.size`.
POINT_WINDOW = (config["evaluate"].get("point_scores") or {}).get("window")
POINT_WINDOW = None if POINT_WINDOW in (None, False, -1) else int(POINT_WINDOW)
POINT_STRIDE = max(int((config["evaluate"].get("point_scores") or {}).get("stride", 1) or 1), 1)
#: Point scores exist only to be ranked against labels, so they follow `evaluate.metrics`. A
#: bagged run supplies its own - the vote fraction - so the sliding-window stage stands down.
USE_POINT_SCORES = POINT_WINDOW is not None and EVALUATE_METRICS and not USE_BAGGING

if USE_POINT_SCORES and POINT_WINDOW < 1:
    raise WorkflowError(
        "evaluate.point_scores.window is {0}; it is a width in increments, so it must be at "
        "least 1. Use null to switch the stage off.".format(POINT_WINDOW))

#: The detector re-run at scaled thresholds - one row per operating point, and an area under the
#: points they trace. Measures the deployed system rather than the representation, and costs a
#: full search per point, which is why it is off unless asked for.
OPERATING_POINTS = "results/{dataset}_operating_points.csv"
SCALES = (config["evaluate"].get("operating_points") or {}).get("scales") or []
SCALES = [float(s) for s in SCALES]
USE_OPERATING_POINTS = bool(SCALES) and EVALUATE_METRICS and not USE_BAGGING
if SCALES and USE_BAGGING:
    raise WorkflowError(
        "evaluate.operating_points.scales is set and detect.bagged.draws is {0}. Sweeping the "
        "threshold means re-running the detector once per scale, and a bagged detector is "
        "{0} detectors - the sweep would cost {1} full runs. Clear the scales, or switch "
        "bagging off.".format(BAGGED_DRAWS, BAGGED_DRAWS * len(SCALES)))
if any(scale <= 0 for scale in SCALES):
    raise WorkflowError(
        "evaluate.operating_points.scales must all be positive - they multiply the calibrated "
        "thresholds. Got {0}.".format(SCALES))
FIGURES = "figures/{dataset}"
REPORT = "reports/{dataset}_report.html"


def side_urls(side):
    """Configured URLs for one side. Empty means 'already on disk, or to be split out'."""
    return config["raw"].get("{0}_urls".format(side)) or []


#: What each side will produce, resolved once so the DAG knows its files up front.
CORPUS_FILES = raw_files(side_urls("corpus"), "corpus", DATASET)
TEST_FILES = raw_files(side_urls("test"), "test", DATASET)

#: Whether the channel reduction is in the DAG at all.
USE_UMAP = UMAP_DIMENSION is not None


def corpus_input():
    """What `preprocess` reads for the corpus side: latent streams, or the canonical set."""
    return PROJECTED_CORPUS if USE_UMAP else CANONICAL_CORPUS


def test_input():
    """What `preprocess` reads for the test side, or nothing at all when there is none.

    Both sides go through the same projection, so nothing downstream of this point can tell a
    latent stream from a raw one. That is the whole of the toggle's effect on the workflow -
    a stage is interposed and the file paths change; no other rule is aware of it.

    An empty result is meaningful and must stay empty. `raw.test_urls` being unset says the
    test period is to be cut off the end of the corpus series at `split.base_fraction`, so
    naming a canonical test file here would have `preprocess` read "a file is expected" as
    "test data was supplied" and skip the cut.
    """
    if not TEST_FILES:
        return []
    return PROJECTED_TEST if USE_UMAP else CANONICAL_TEST


#: Increments per window, or None when the stage is off. Read once: it decides both what
#: `corpus` and `score` read, and the depth band a short final window is judged in - and those
#: two must agree or the corpus and the queries are measured on different scales.
WINDOW_SIZE = resolve_size(config.get("window", {}).get("size", -1))

#: Whether the windowing stage is in the DAG at all.
USE_WINDOWS = WINDOW_SIZE is not None


#: What actually gets signed and scored. Interposed exactly as the UMAP stage is - a rule is
#: added, the paths change, and no other rule learns that anything happened. Windows *are*
#: canonical stream sets; the extra columns they carry are ignored by every reader that does not
#: want them.
SIGNING = (WINDOWED.replace("{size}", str(WINDOW_SIZE)) if USE_WINDOWS else WINDOWS)
SIGNING_CORPUS = SIGNING.replace("{split}", "corpus")
SIGNING_TEST = SIGNING.replace("{split}", "test")


#: Latent dimension of the signature space, or None when the corpus is used at full width.
SIGNATURE_DIMENSION = latent_dimension(
    config.get("signature_umap", {}).get("dimension", -1))
USE_SIGNATURE_UMAP = SIGNATURE_DIMENSION is not None

#: What the metric, the index and the calibration are built from.
METRIC_CORPUS = REDUCED_CORPUS if USE_SIGNATURE_UMAP else CORPUS

#: Which question is asked of each candidate interval.
#:
#:   mahalanobis       distance to the nearest normal interval, in the whitened space.
#:   isolation_forest  how few random axis-aligned splits isolate it.
DETECT_METHOD = config["detect"].get("method", "mahalanobis")
if DETECT_METHOD not in ("mahalanobis", "isolation_forest"):
    raise WorkflowError(
        "detect.method is {0!r}; expected 'mahalanobis' or 'isolation_forest'."
        .format(DETECT_METHOD))
USE_FOREST = DETECT_METHOD == "isolation_forest"

#: Which neighbour's distance is the score. Read once and passed to both `calibrate` and
#: `score`, because the threshold and the score must be drawn from the same distribution.
NEIGHBOURS = int(config["detect"].get("neighbours", 1))
if NEIGHBOURS < 1:
    raise WorkflowError(
        "detect.neighbours is {0}; it is an ordinal - 1 is the nearest neighbour - so it must "
        "be at least 1.".format(NEIGHBOURS))
if NEIGHBOURS != 1 and USE_FOREST:
    raise WorkflowError(
        "detect.neighbours is {0} and detect.method is 'isolation_forest'. A forest reports how "
        "easily a point is isolated and matches nothing, so it has no n-th neighbour to measure "
        "against. Set detect.neighbours to 1, or use the mahalanobis method.".format(NEIGHBOURS))

#: Relative residual above which an interval is called off-manifold, or None when the test is
#: off. Read once because it decides three things that must agree: whether `covariance` writes
#: the row-space basis, whether `index` keeps the unwhitened vectors, and whether `score` tests.
OFF_MANIFOLD = config["detect"].get("off_manifold")
# The string forms are handled too: `--config` parses each override on its own, so a nested
# value written `null` on the command line arrives as the four characters rather than as None.
if isinstance(OFF_MANIFOLD, str):
    OFF_MANIFOLD = OFF_MANIFOLD.strip().lower()
    OFF_MANIFOLD = None if OFF_MANIFOLD in ("", "null", "none", "false") else OFF_MANIFOLD
OFF_MANIFOLD = None if OFF_MANIFOLD in (None, False) else float(OFF_MANIFOLD)

if OFF_MANIFOLD is not None and USE_FOREST:
    raise WorkflowError(
        "detect.off_manifold is set and detect.method is 'isolation_forest'. The residual test "
        "compares a query against the corpus vector it matched, and a forest matches nothing - "
        "it reports how easily a point is isolated, not what it resembles. Switch the test off, "
        "or use the mahalanobis method.")

if OFF_MANIFOLD is not None and not 0 < OFF_MANIFOLD < 1:
    raise WorkflowError(
        "detect.off_manifold is {0}; it is a fraction of a difference's norm, so it belongs in "
        "(0, 1). SigMahaKNN uses 1e-3. Use null to switch the test off.".format(OFF_MANIFOLD))


#: Whether signature components are normalised before the metric is fitted.
USE_NORMALISER = bool(config["signature"].get("normalise", False))

if USE_SIGNATURE_UMAP and USE_NORMALISER:
    raise WorkflowError(
        "signature.normalise and signature_umap.dimension are both set, and they are two "
        "answers to the same question. The reduction standardises every term before embedding - "
        "that is what its scaler is for - so a normaliser fitted beforehand would be applied "
        "twice, and one fitted afterwards would be applied to latent coordinates the metric "
        "already handles. Switch one of them off.")


def normaliser_input():
    """The fitted normaliser, or nothing when normalisation is off.

    An empty list rather than a missing key, so the rules that consume it can ask for
    `input.get("normaliser")` without branching on the config themselves.
    """
    return NORMALISER if USE_NORMALISER else []


def apply_normaliser(corpus_path, normaliser_path):
    """Read a corpus and put it in the normaliser's units, if there is one.

    Returns a frame rather than a path, so the caller never has to hold both cases.
    """
    frame = read_corpus(corpus_path)
    if not normaliser_path:
        return frame
    return Normaliser.load(normaliser_path).transform(frame)


def ground_truth():
    """Labels for `evaluate`, or None. Absent labels are normal, not an error."""
    labels = config["evaluate"].get("labels")
    return labels if labels and Path(labels).exists() else None


def placeholder(rule_name, intent, outputs, reads=()):
    """Fail with an explanation of what this stage is for, rather than silently doing nothing.

    Keeps an unimplemented stage self-documenting: the error names the stage, what it should
    produce, and which config keys it will read when written.
    """
    outputs = [outputs] if isinstance(outputs, str) else list(outputs)
    message = [
        "rule '{0}' is a placeholder and has no implementation yet.".format(rule_name),
        "  It should {0}.".format(intent),
        "  Expected output: {0}".format(", ".join(str(o) for o in outputs)),
    ]
    if reads:
        message.append("  Config it will read: {0}".format(", ".join(reads)))
    raise WorkflowError("\n".join(message))


# ---------------------------------------------------------------------------------------
rule all:
    """Default target: the full chain, ending at the report."""
    input:
        expand(REPORT, dataset=DATASET),


rule download:
    """Stage 1 of 9. Fetch one raw file.

    Corpus and test data are configured as two separate lists of URLs, and every URL becomes
    its own job, so a dataset spread over many files - one per machine, say - downloads in
    parallel and a failure part-way through only costs the files that had not arrived.

    Leaving a list empty is legitimate: place the files under `data/raw/{dataset}/{side}/`
    yourself and they are picked up from there. An empty `test` list additionally means the
    test period should be cut out of the corpus series by `preprocess`.
    """
    output:
        RAW,
    wildcard_constraints:
        side="corpus|test",
    run:
        download_target(output[0], side_urls(wildcards.side),
                        wildcards.side, wildcards.name)


def canonical_sources(wildcards):
    """The raw files one side canonicalises from, resolved at DAG-build time.

    An input function rather than a fixed list, because the files depend on the `side`
    wildcard. Returning them here is what tells Snakemake to create a `download` job for each
    one that is not already present - and equally, when no URLs are configured, to create none
    and simply use whatever was placed in the directory by hand.
    """
    return raw_files(side_urls(wildcards.side), wildcards.side, DATASET)


rule canonicalise:
    """Stage 1a of 9. Collapse one side's raw files into a single canonical stream set.

    Every dataset arrives differently - a directory of headerless per-machine files, a single
    file with a unit column distinguishing a hundred engines, one CSV with a timestamp - and
    each shape previously travelled its own code path through the pipeline. They all become
    one table here: a row per stream, carrying `stream`, `Time` and a `(length, k)` block of
    values.

    Separate from `download` on purpose. Fetching stays one job per URL, so a dataset spread
    over 28 files still downloads in parallel and a failure part-way through only costs the
    files that had not arrived; this rule then waits for all of them, because the channel-count
    check is only meaningful where every stream is visible at once.

    Streams are *not* padded to a common length. A fleet of engines with different lifetimes is
    the normal case, and equalising them would invent data.
    """
    input:
        canonical_sources,
    output:
        CANONICAL,
    benchmark:
        BENCHMARK.format(stage="canonicalise_{side}"),
    wildcard_constraints:
        side="corpus|test",
    run:
        write_canonical(
            canonicalise_files(
                input,
                time_col=config["raw"]["time_column"],
                value_cols=config["raw"]["value_column"],
                stream_col=config["signature"]["stream_column"],
                show_progress=True,
            ),
            output[0],
        )


rule umap:
    """Stage 1b of 9, and only in the DAG when `umap.dimension` is positive. Reduce channels.

    Fits **one** UMAP on the corpus of normality, then a distance-weighted kNN regressor from
    ambient coordinates to embedded ones. That regressor is what every stream goes through -
    corpus included - so both sides land in identical coordinates by construction, and there is
    no out-of-sample problem to work around.

    This replaced an `AlignedUMAP` arrangement that was the wrong tool rather than a mistuned
    one: AlignedUMAP is built for overlapping slices whose `relations` identify genuinely shared
    samples, it was handed disjoint slices asserting identity between different objects, and it
    offers no `transform` for unseen data at all.

    Context windows. With `umap.context` above 1 each row is a delay embedding anchored at its
    newest point, so there is exactly one row - and one latent point - per observation, path
    length is preserved, and point indices keep meaning what they meant. Rows never cross a
    stream boundary, and the fit set is strided by the context width so that overlapping windows
    cannot weight an observation by however many of them happen to contain it. Only the fit is
    strided; every row is transformed.

    Before lead-lag, always. That transform pairs each channel with its own past, taking k
    channels to 2k; reducing afterwards would let UMAP mix a channel with its delayed copy and
    dissolve the pairing. Reducing first gives k -> d and then d -> 2d.

    Why it has to be here, before preprocess
    ----------------------------------------
    Signature dimension is roughly ``d ** trunc``. It grows polynomially in the length of the
    path but exponentially in the truncation level, and the base of that exponent is the
    channel count, so `d` is the only lever that acts on it. SMD's 39 channels at trunc 4 give
    2,374,320 terms - a corpus that cannot be built at all, and no later stage can recover from
    that. Four latent channels give 780.

    Why the projection is saved
    ---------------------------
    Test streams must be mapped by the *same* fitted object, and are, in `score`. A signature is
    only comparable to another signature of a path in the same coordinates; a projection
    refitted on test data would place the same point elsewhere, and the Mahalanobis distance
    between the two would be meaningless. So the fit happens exactly once, on corpus data
    alone, and is persisted.

    Diagnostics are advisory
    ------------------------
    `path_smoothness` is the one to read, and it replaced `alignment_drift`. It compares the
    mean latent step against the mean ambient step, both scale-normalised, because the next
    stage *integrates* the path: an embedding that jitters between adjacent instants inflates
    every level-2 signature term with length the data does not have. Near 1 is good. The stage
    reports a verdict and does not halt - whether a poor embedding is acceptable is a judgement
    about the data, and the person reading the report is better placed to make it.

    One check this stage cannot perform: whether anomalies survive the reduction. UMAP is
    fitted to normal data and an anomaly is off that manifold, which is exactly where the
    mapping is least constrained - it may carry anomalies back onto normal-looking coordinates
    and erase them. No unsupervised metric can settle that; it needs the detector run on
    labelled data with and without the reduction, which this pipeline can now do end to end.
    """
    input:
        corpus=CANONICAL_CORPUS,
        test=CANONICAL_TEST,
    output:
        corpus=PROJECTED_CORPUS,
        test=PROJECTED_TEST,
        projection=PROJECTION,
        diagnostics=PROJECTION_DIAGNOSTICS,
    benchmark:
        BENCHMARK.format(stage="umap"),
    run:
        project_corpus_and_test(
            corpus_files=input.corpus,
            test_files=input.test,
            projection_path=output.projection,
            corpus_dir=output.corpus,
            test_dir=output.test,
            diagnostics_path=output.diagnostics,
            n_components=UMAP_DIMENSION,
            context=config["umap"]["context"],
            n_neighbors=config["umap"]["n_neighbors"],
            min_dist=config["umap"]["min_dist"],
            standardise=config["umap"]["standardise"],
            random_state=config["umap"]["random_state"],
            fit_stride=config["umap"]["fit_stride"],
            knn_neighbours=config["umap"]["knn_neighbours"],
            show_progress=True,
        )


rule preprocess:
    """Stage 2 of 9. Name and screen.

    Everything between the canonical stream sets and two sets ready to sign: canonical column
    naming, and screening out streams that cannot be signed. A stream carrying a single
    non-finite value is removed whole and reported by name - one NaN propagates through
    `iisignature.sig` into every term of its interval and from there into the metric, and there
    is no partial recovery from that.

    It does not cut windows. Streams arrive as streams and leave as streams; each split is one
    canonical file rather than a directory, which is what `corpus` reads.

    Channels that never move are deliberately kept. A zero-variance direction is annihilated by
    the pseudo-inverse rather than amplified by it, so a constant channel contributes nothing to
    the Mahalanobis distance instead of dividing by zero in it.

    The only division left is corpus versus test, and it is usually already made: labelled
    benchmarks ship the two sides as distinct periods, in which case `split.base_fraction` is
    not read at all. When they do not, the test period is taken off the end of each corpus
    series at that fraction - positionally, never at random, because these are time series and a
    random cut would put neighbouring stretches of one series on opposite sides of it.

    The fit/withheld cut this rule used to make is gone. It existed so the threshold could be
    measured against normal data the metric had never seen; `calibrate` now gets that from
    k-fold over streams, for the whole corpus rather than a fifth of it.
    """
    input:
        corpus=corpus_input(),
        test=test_input(),
    output:
        corpus=CORPUS_STREAMS,
        test=TEST_STREAMS,
    benchmark:
        BENCHMARK.format(stage="preprocess"),
    params:
        # Declared rather than only read inside `run:`, so that changing them reruns the stage.
        # Snakemake tracks files and declared params; a config value read from inside the body
        # is invisible to it, and the sets would be judged up to date at the old width.
        lag=config["preprocess"]["lag"],
        lag_method=config["preprocess"]["lag_method"],
    run:
        preprocess_streams(
            input.corpus,
            input.test,
            corpus_dir=output.corpus,
            test_dir=output.test,
            base_fraction=config["split"]["base_fraction"],
            lag=params.lag,
            lag_method=params.lag_method,
            time_col=config["raw"]["time_column"],
            value_cols=config["raw"]["value_column"],
            stream_col=config["signature"]["stream_column"],
            summary_path=PREPROCESS_SUMMARY.format(dataset=DATASET),
            show_progress=True,
        )


rule window:
    """Stage 2b of 9, and only in the DAG when `window.size` is set. Cut streams to one length.

    The corpus is built from every interval whose endpoints lie on a stream's dyadic grid, and
    that grid is *relative* - `signature.granularity` says how many cells, not how many points.
    So a corpus built from streams of different lengths holds intervals of different lengths
    inside a single depth band, and a query is compared against whatever happens to share its
    band. On C-MAPSS, whose engines run 31 to 303 cycles, that is a ten-fold spread within one
    depth: a 31-cycle degradation fills the whole of a short engine's depth-0 interval and under
    a third of a long one's, so the same event signs quite differently depending on which engine
    it happened to.

    Fixing the window collapses that spread - depth *d* becomes a fixed number of points
    everywhere - which is what the notebook does by construction and what the pipeline's recall
    deficit against it points at.

    Geometry. A window carries `window.size` **increments**, so `size + 1` points, and
    consecutive windows share their boundary point. That is what makes the dyadic grid divide
    exactly: 16 increments at granularity 3 give edges 0, 2, ..., 16, eight even cells. Sixteen
    *points* instead would give one cell a single increment wide.

    The tail is kept short rather than padded. Carrying the last observation forward would append
    a run of zero increments that signs as nearly nothing - and on run-to-failure data that run
    sits exactly where the failure is. A short tail can still be too short for the grid, so the
    corpus splits drop anything under `2**granularity + 1` points while `test` keeps everything:
    the widest-first search needs no grid, and discarding those points would mean never scoring
    the end of a stream.
    """
    input:
        WINDOWS,
    output:
        WINDOWED,
    benchmark:
        BENCHMARK.format(stage="window_{split}_w{size}"),
    wildcard_constraints:
        split="|".join(SPLITS),
        size=r"\d+",
    run:
        window_streams(
            input[0],
            size=int(wildcards.size),
            # `test` is scanned, not signed on a grid, so nothing is dropped there.
            min_points=(2 if wildcards.split == "test"
                        else 2 ** config["signature"]["granularity"] + 1),
            output_path=output[0],
            show_progress=True,
        )


rule corpus:
    """Stage 3 of 9. Build the signature corpus for one split.

    Every stream is cut into the intervals whose endpoints lie on the finest dyadic grid, and
    each is signed with a base point. Not only the dyadic tree: the widest-first search extends
    a clean block outward and asks about runs of adjacent cells, and a run of three cells is not
    a dyadic node. Scoring one against a corpus holding only 2- and 4-cell intervals compares it
    to the wrong width, so the unions are built too - 36 intervals per stream at granularity 3
    against the tree's 15.

    Each interval is signed directly from its own points rather than composed from its children.
    Chen composition would cost about one pass per stream instead of fifteen, but
    `iisignature.sigcombine` is float64-typed and float32-accurate, and that is a floor rather
    than an accumulation - it caps how deep a truncation is worth taking.
    """
    input:
        SIGNING_CORPUS,
    output:
        CORPUS,
    benchmark:
        BENCHMARK.format(stage="corpus"),
    run:
        compute_corpus(
            input[0],
            trunc=config["signature"]["trunc"],
            granularity=config["signature"]["granularity"],
            output_path=output[0],
            show_progress=True,
        )


rule normalise:
    """Stage 3b of 9, and only in the DAG when `signature.normalise` is on.

    Puts every signature component on a comparable scale. A level-k term is a k-fold iterated
    integral and scales like the increment to the k-th power, so the raw corpus concentrates its
    variance in whichever level has the largest increments - measured at 1,363x between levels 1
    and 2 on C-MAPSS.

    The whitening already rescales each retained direction, so this is not about the distance.
    It is about which directions get retained: `metric.variance_keep` cuts by share of total
    variance, so without this the retained rank is entirely level-2 directions and the level-1
    structure is discarded before the metric sees it. Measured, that cut moved from 8 directions
    to 39.

    Fitted on the corpus and saved, because the test signatures have to be put in the same units
    and a normaliser refitted on test data would place the two sides in different ones.
    """
    input:
        corpus=CORPUS,
    output:
        normaliser=NORMALISER,
    benchmark:
        BENCHMARK.format(stage="normalise"),
    run:
        normalise_corpus(
            read_corpus(input.corpus),
            method=config["signature"]["normalise_method"],
            normaliser_path=output.normaliser,
            show_progress=True,
        )


rule reduce:
    """Stage 3c of 9, and only in the DAG when `signature_umap.dimension` is positive.

    Compresses the *signature* space rather than the channel space. The two are different
    levers on the same problem and this is the weaker one arithmetically - the corpus still has
    to be built at full width, so nothing here makes an unbuildable truncation buildable, and
    the saving is linear rather than exponential.

    What it buys instead is that the path is never touched. `umap` discards information while
    the data is still a path, and the signature can only integrate what survives; measured on
    C-MAPSS that cost precision heavily, 0.853 down to about 0.50, because degraded windows were
    mapped back among the normal ones before the signature ever saw them. Reducing afterwards
    lets the signature see every channel at full width and only then compresses the description.

    The output is still a corpus, so `covariance`, `index` and `calibrate` are unchanged. Only
    `score` needs the projection, because it computes its signatures on the fly and has to place
    them in the same latent space - which is exactly why the projection is a fitted function and
    not an embedding.
    """
    input:
        corpus=CORPUS,
    output:
        corpus=REDUCED_CORPUS,
        projection=SIGNATURE_PROJECTION,
        diagnostics=SIGNATURE_PROJECTION_DIAGNOSTICS,
    benchmark:
        BENCHMARK.format(stage="reduce"),
    run:
        reduce_signature_corpus(
            input.corpus,
            output_corpus=output.corpus,
            projection_path=output.projection,
            diagnostics_path=output.diagnostics,
            n_components=SIGNATURE_DIMENSION,
            n_neighbors=config["signature_umap"]["n_neighbors"],
            min_dist=config["signature_umap"]["min_dist"],
            knn_neighbours=config["signature_umap"]["knn_neighbours"],
            fit_rows=config["signature_umap"]["fit_rows"],
            random_state=config["signature_umap"]["random_state"],
            show_progress=True,
        )


rule covariance:
    """Stage 4 of 9. Fit the Mahalanobis metric.

    Fitted on the corpus - never on test data - and rank-truncated by retained variance, since a
    signature corpus is nowhere near full rank and inverting its empty directions would amplify
    rounding error into the distance.

    It is fitted on the *whole* corpus, which is a change from when a `fit` split existed. The
    k-fold in `calibrate` holds a stream out of the searchable index but not out of this, so a
    calibration interval is scored under a metric that has seen its own stream. That is a much
    weaker leak than the one the folds close - a covariance over a hundred streams barely moves
    when one is removed, where an index containing an interval's own near-duplicates halved the
    threshold - but it is not zero, and closing it would mean refitting the metric per fold.

    `form: inv_sqrt` is what the index needs. Applying Sigma^-1 to a flat L2 index instead
    computes Sigma^-2 distances, which is silent and wrong by orders of magnitude.
    """
    input:
        corpus=METRIC_CORPUS,
        normaliser=normaliser_input(),
    output:
        covariance=COVARIANCE,
        diagnostics=COVARIANCE_DIAGNOSTICS,
        subspace=SUBSPACE,
    benchmark:
        BENCHMARK.format(stage="covariance"),
    run:
        create_covariance(
            apply_normaliser(input.corpus, input.get("normaliser")),
            output_path=output.covariance,
            variance_keep=config["metric"]["variance_keep"],
            form=config["metric"]["form"],
            diagnostics_path=output.diagnostics,
            subspace_path=output.subspace,
            # Store the rank-r factor rather than the dense matrix, and let every stage
            # downstream work in r latent coordinates. Exact, not approximate: the projection is
            # an isometry, which this asserts before writing.
            factored=FACTORED_METRIC,
            show_progress=True,
        )


rule index:
    """Stage 5 of 9. Whiten the corpus and build the searchable FAISS index.

    Applying Sigma^-1/2 first is what makes plain Euclidean distance *be* Mahalanobis distance,
    which is the only reason a flat L2 index is the right structure.

    The searchable set is every corpus interval. Coverage of normality is what a
    nearest-neighbour detector is short of, so nothing normal is held back from it - the
    question of what a stream scores against a corpus that does not contain it is asked in
    `calibrate`, by rebuilding a reference set per fold, rather than by keeping rows out of this.

    `detect.index.type` decides how each depth band is structured for searching. It is a
    load-time property - the artifact here is a flat vector store either way.
    """
    input:
        corpus=METRIC_CORPUS,
        covariance=COVARIANCE,
        normaliser=normaliser_input(),
    output:
        index=INDEX,
        meta=INDEX_META,
    benchmark:
        BENCHMARK.format(stage="index"),
    run:
        build_index(
            {"corpus": apply_normaliser(input.corpus, input.get("normaliser"))},
            input.covariance,
            output_index=output.index,
            output_meta=output.meta,
            band=config["detect"]["band"],
            nn_reference_size=config["detect"]["nn_reference_size"],
            random_state=config["detect"]["random_state"],
            # Keeping the unwhitened vectors is the whole cost of the off-manifold test, and
            # it is the corpus over again - so it is paid only when the test is armed.
            store_raw=OFF_MANIFOLD is not None,
            show_progress=True,
        )


rule forest:
    """Stage 5b of 9, and only in the DAG when `detect.method` is `isolation_forest`.

    Fits one isolation forest per depth band, on the intervals within `detect.band` of that
    depth - the same reference set the FAISS index would have searched, asked a different
    question. There is no metric to fit and nothing to invert, so this is the cheaper of the two
    stages despite doing the same amount of reading.

    `detect.forest.space` decides the coordinates. `whitened` reuses the covariance and is the
    conservative choice; `raw` skips it and flags roughly twice as much. Both are measured in
    the module docstring - the difference is real and not a tuning detail, because a forest cuts
    on axes and is therefore sensitive to the basis in a way Mahalanobis distance is not.
    """
    input:
        corpus=METRIC_CORPUS,
        covariance=COVARIANCE,
    output:
        FOREST,
    benchmark:
        BENCHMARK.format(stage="forest"),
    run:
        forest = config["detect"]["forest"]
        fit_forests(
            input.corpus,
            input.covariance,
            band=config["detect"]["band"],
            space=forest["space"],
            n_estimators=forest["n_estimators"],
            max_samples=forest["max_samples"],
            random_state=config["detect"]["random_state"],
            output_path=output[0],
            show_progress=True,
        )


rule calibrate:
    """Stage 6 of 9. Choose one threshold per depth, by cross-validation over streams.

    The threshold has to answer how far from the corpus *unseen* normal data sits, and every
    way of measuring that on data the corpus already contains gives an answer that is too
    small. Scoring each interval against the index while skipping its own self-match removes
    one row and leaves the problem: a stream contributes 36 overlapping grid intervals, all of
    them indexed, so an interval's nearest surviving neighbour is usually another interval of
    the same stream covering nearly the same points. That measures how well a stream matches
    itself. On this corpus it halved the threshold.

    So the streams are partitioned into `calibrate.folds` folds, and each fold's intervals are
    scored against an index built from the other folds only. Every interval is calibrated
    exactly once, under the condition the detector actually meets - a stream it has not seen.

    This reads the corpus rather than the built index, because a reference set has to be
    rebuilt per fold; the index in `data/index` contains every stream and cannot answer the
    question. Thresholds come out keyed by depth, which matters as much as the folds do: a
    level-k signature term scales like the increment to the k-th power, so one scalar over every
    width behaves partly as a width filter rather than an anomaly test.
    """
    input:
        corpus=METRIC_CORPUS,
        covariance=COVARIANCE,
        normaliser=normaliser_input(),
    output:
        THRESHOLDS,
    benchmark:
        BENCHMARK.format(stage="calibrate"),
    run:
        calibrate_thresholds = (
            crossvalidated_forest_thresholds if USE_FOREST else crossvalidated_thresholds)
        extra = ({"space": config["detect"]["forest"]["space"],
                  "n_estimators": config["detect"]["forest"]["n_estimators"],
                  "max_samples": config["detect"]["forest"]["max_samples"]}
                 if USE_FOREST else
                 {"normaliser": input.get("normaliser") or None,
                  # The same n `score` will use. Calibrating at one and scoring at another
                  # would compare a quantile of one distribution against draws from a
                  # systematically wider one.
                  "neighbours": NEIGHBOURS})

        calibrate_thresholds(
            input.corpus,
            input.covariance,
            k=config["calibrate"]["folds"],
            statistic=config["calibrate"]["statistic"],
            band=config["detect"]["band"],
            random_state=config["calibrate"]["random_state"],
            # Fold on the source stream, so every window of one engine is held out together.
            # Sibling windows share a unit, its wear and its calibration offsets, and a fold
            # that queried one against another would measure how well an engine matches itself.
            windowed=USE_WINDOWS,
            output_path=output[0],
            show_progress=True,
            **extra,
        )


rule score:
    """Stage 7 of 9, unless `detect.bagged.draws` is set, in which case `bagged` replaces it.

    Widest-first: the whole stream is tested, and whatever fails is bisected and retested down
    to a resolution floor, then the clean regions are grown outward by binary search. What is
    reported is the complement - the points no clean interval covers - as inclusive [lo, hi]
    point indices.

    The normaliser is passed when there is one, because a test signature has to be put in the
    same units the corpus was normalised into before the covariance and the index mean anything
    against it.
    """
    input:
        test=SIGNING_TEST,
        index=[] if USE_FOREST else INDEX,
        meta=[] if USE_FOREST else INDEX_META,
        forest=FOREST if USE_FOREST else [],
        covariance=COVARIANCE,
        thresholds=THRESHOLDS,
        normaliser=normaliser_input(),
        subspace=SUBSPACE if OFF_MANIFOLD is not None else [],
        projection=SIGNATURE_PROJECTION if USE_SIGNATURE_UMAP else [],
    output:
        scores=SCORES,
        stats=SCORE_STATS,
    benchmark:
        BENCHMARK.format(stage="score"),
    run:
        if config["detect"]["mode"] != "pooled":
            raise WorkflowError(
                "detect.mode is {0!r}; this workflow implements the pooled detector. Use the "
                "per_interval workflow for the alternative."
                .format(config["detect"]["mode"]))

        structure = config["detect"]["index"]
        if USE_FOREST:
            # `DepthForests` presents the same surface the scorer uses of a `PooledIndex`, and
            # carries the matrix that puts a fresh signature in its own space - identity when
            # the forests were fitted on raw signatures.
            detector = DepthForests.load(input.forest)
            metric = detector.scorer_covariance
        else:
            detector = load_index(
                input.index, input.meta,
                index_type=structure["type"],
                nlist=structure["nlist"],
                nprobe=structure["nprobe"],
                hnsw_m=structure["hnsw_m"],
                ef_construction=structure["ef_construction"],
                ef_search=structure["ef_search"],
                random_state=config["detect"]["random_state"],
            )
            metric = input.covariance

        score_streams(
            input.test,
            detector,
            metric,
            threshold=input.thresholds,
            # `or None`: an absent optional input is an empty list, not None, and the
            # modules test `is not None` - so [] would be taken for a normaliser.
            normaliser=input.get("normaliser") or None,
            sig_tol=config["detect"]["sig_tol"],
            tol=config["detect"]["tol"],
            # Depth from absolute width against the standard window, not from each stream's own
            # length. A short final window measured against itself reports full width, lands at
            # depth 0, and is judged against references several times its length - which makes
            # it score low for reasons unconnected to the data, at the end of the stream, which
            # on run-to-failure data is where the failure is.
            span=WINDOW_SIZE,
            # `or None` again: an absent optional input arrives as an empty list.
            subspace=input.get("subspace") or None,
            subspace_threshold=OFF_MANIFOLD,
            signature_projection=input.get("projection") or None,
            # Ignored by a forest, which reports one score and has no n-th neighbour; the
            # workflow refuses the combination above rather than letting it pass silently.
            neighbours=NEIGHBOURS,
            # With the corpus in a latent space the index width no longer encodes the
            # truncation, so it cannot be inferred and has to be stated.
            trunc=config["signature"]["trunc"] if USE_SIGNATURE_UMAP else None,
            output_path=output.scores,
            # The counts the benchmark's seconds are divided by. Written here rather than
            # recovered later because the query total is a property of the search's execution,
            # not of anything it leaves on disk.
            stats_path=output.stats,
            show_progress=True,
        )


rule bagged:
    """Stage 5-7 of 9 at once, and only in the DAG when `detect.bagged.draws` is positive.

    Feature bagging: each draw takes round(sqrt(d)) of the d value channels, builds a corpus
    from those alone, fits a metric, calibrates it, searches the test streams, and rasterises
    what it flagged. A point's vote count is how many draws flagged it.

    This rule spans what `corpus`, `covariance`, `index`, `calibrate` and `score` do separately,
    because a draw's artifacts are meaningless to any other draw - different channels mean
    different signature terms and a different whitened space - so there is nothing shared to
    build once and cache. Persisting them per draw would cost disk for artifacts nothing can
    reuse; they are built in memory and discarded.

    It writes `SCORES` in the shape `score_streams` produces, so `evaluate` and `report` need no
    branch, and `POINT_SCORES` as the vote fraction, which is a continuous score per point and
    therefore what ROC-AUC and PR-AUC rank. A bagged run needs no sliding-window pass; the vote
    count already is one.
    """
    input:
        # `SIGNING_*` already resolves to the windowed sets when `window.size` is set, which is
        # what `corpus` and `score` read - so a bagged draw sees exactly the streams the
        # single-detector path would have.
        corpus=SIGNING_CORPUS,
        test=SIGNING_TEST,
    output:
        scores=SCORES,
        stats=SCORE_STATS,
        points=POINT_SCORES,
        diagnostics=BAGGED_DIAGNOSTICS,
    benchmark:
        BENCHMARK.format(stage="bagged"),
    run:
        import json

        scored, votes, diagnostics = run_bagged(
            input.corpus,
            input.test,
            draws=BAGGED_DRAWS,
            features=BAGGED.get("features"),
            votes=int(BAGGED.get("votes", 1) or 1),
            trunc=config["signature"]["trunc"],
            granularity=config["signature"]["granularity"],
            variance_keep=config["metric"]["variance_keep"],
            band=config["detect"]["band"],
            folds=config["calibrate"]["folds"],
            statistic=config["calibrate"]["statistic"],
            random_state=int(BAGGED.get("random_state", 0) or 0),
            span=WINDOW_SIZE,
            sig_tol=config["detect"]["sig_tol"],
            tol=config["detect"]["tol"],
            neighbours=NEIGHBOURS,
            detector=DETECT_METHOD,
            forest_settings=config["detect"]["forest"],
            scores_path=output.scores,
            output_path=output.points,
            diagnostics_path=output.diagnostics,
            show_progress=True,
        )

        # The same counts `score_streams` writes, so the throughput block reads a bagged run the
        # same way it reads any other. Queries are not comparable across the two - a bagged run
        # issues them once per draw - so the draw count is recorded beside them.
        Path(output.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(output.stats).write_text(json.dumps({
            "n_streams": int(len(scored)),
            "n_points": int(sum(len(v) for v in votes["point_score"])),
            "n_queries": 0,
            "n_off_manifold": 0,
            "reference_size": 0,
            "dimension": 0,
            "neighbours": NEIGHBOURS,
            "draws": BAGGED_DRAWS,
            "seconds": diagnostics["seconds"],
        }, indent=2), encoding="utf-8")


rule point_scores:
    """Stage 7b of 9, and only in the DAG when `evaluate.point_scores.window` is set.

    Gives every point a continuous score, on a fixed sliding window, so that `evaluate` can
    compute ROC-AUC and PR-AUC. Those summarise a detector over every threshold, and `score`
    emits a decision rather than a ranking - a binary flag has already had one threshold applied
    and discarded the rest.

    A sibling of `score` rather than an extra output of it. The two share every input and could
    have been one rule, but their costs are unlike: `score` issues a few thousand *adaptive*
    queries whose count depends on how anomalous the data is, this issues a fixed batch per
    stream. One `benchmark:` covering both would report a number attributable to neither, which
    would waste the instrumentation the throughput work exists to provide. Keeping them apart
    also lets the ranking be recomputed without redoing the search, and lets the two run in
    parallel.
    """
    input:
        test=SIGNING_TEST,
        index=[] if USE_FOREST else INDEX,
        meta=[] if USE_FOREST else INDEX_META,
        forest=FOREST if USE_FOREST else [],
        covariance=COVARIANCE,
        thresholds=THRESHOLDS,
        normaliser=normaliser_input(),
        projection=SIGNATURE_PROJECTION if USE_SIGNATURE_UMAP else [],
    output:
        POINT_SCORES,
    benchmark:
        BENCHMARK.format(stage="point_scores"),
    run:
        structure = config["detect"]["index"]
        if USE_FOREST:
            detector = DepthForests.load(input.forest)
            metric = detector.scorer_covariance
        else:
            detector = load_index(
                input.index, input.meta,
                index_type=structure["type"],
                nlist=structure["nlist"],
                nprobe=structure["nprobe"],
                hnsw_m=structure["hnsw_m"],
                ef_construction=structure["ef_construction"],
                ef_search=structure["ef_search"],
                random_state=config["detect"]["random_state"],
            )
            metric = input.covariance

        score_points(
            input.test,
            detector,
            metric,
            threshold=input.thresholds,
            window=POINT_WINDOW,
            stride=POINT_STRIDE,
            # The same span `score` uses, so a window here lands in the depth band its absolute
            # width belongs to rather than one decided by whichever stream it sits in.
            span=WINDOW_SIZE,
            normaliser=input.get("normaliser") or None,
            signature_projection=input.get("projection") or None,
            trunc=config["signature"]["trunc"] if USE_SIGNATURE_UMAP else None,
            neighbours=NEIGHBOURS,
            output_path=output[0],
            show_progress=True,
        )


rule operating_points:
    """Stage 7c of 9, and only in the DAG when `evaluate.operating_points.scales` is set.

    Re-runs the real detector - the same widest-first search, the same span, the same neighbours
    - with every calibrated threshold multiplied by a common factor, and records where each
    setting lands on the ROC plane. `evaluate` integrates the points into an area.

    This is the *system-level* curve: the score, the search and the calibration measured
    together, which is the faithful answer to "how does this detector perform across operating
    points". `point_scores` measures the representation with the search factored out. They are
    different questions and the summary reports both.

    It is legitimate rather than an approximation because the flagged sets are nested in the
    threshold - raising it can only remove flags - so the curve is monotone. What it is not is
    cheap: every point on it is a complete search over every test stream.
    """
    input:
        test=SIGNING_TEST,
        index=[] if USE_FOREST else INDEX,
        meta=[] if USE_FOREST else INDEX_META,
        forest=FOREST if USE_FOREST else [],
        covariance=COVARIANCE,
        thresholds=THRESHOLDS,
        normaliser=normaliser_input(),
        subspace=SUBSPACE if OFF_MANIFOLD is not None else [],
        projection=SIGNATURE_PROJECTION if USE_SIGNATURE_UMAP else [],
    output:
        OPERATING_POINTS,
    benchmark:
        BENCHMARK.format(stage="operating_points"),
    run:
        structure = config["detect"]["index"]
        if USE_FOREST:
            detector = DepthForests.load(input.forest)
            metric = detector.scorer_covariance
        else:
            detector = load_index(
                input.index, input.meta,
                index_type=structure["type"],
                nlist=structure["nlist"],
                nprobe=structure["nprobe"],
                hnsw_m=structure["hnsw_m"],
                ef_construction=structure["ef_construction"],
                ef_search=structure["ef_search"],
                random_state=config["detect"]["random_state"],
            )
            metric = input.covariance

        print("sweeping {0} operating point(s); each is a full search over every test "
              "stream".format(len(SCALES)), flush=True)
        sweep_operating_points(
            input.test,
            detector,
            metric,
            threshold=input.thresholds,
            truth=ground_truth(),
            scales=SCALES,
            lead_lag=bool(config["preprocess"]["lag"]
                          and config["preprocess"]["lag_method"] == "lead_lag"),
            # Everything below is passed through to `score_streams`, so the detector behaves
            # exactly as it does in `score` and only the threshold differs.
            normaliser=input.get("normaliser") or None,
            sig_tol=config["detect"]["sig_tol"],
            tol=config["detect"]["tol"],
            span=WINDOW_SIZE,
            subspace=input.get("subspace") or None,
            subspace_threshold=OFF_MANIFOLD,
            signature_projection=input.get("projection") or None,
            neighbours=NEIGHBOURS,
            trunc=config["signature"]["trunc"] if USE_SIGNATURE_UMAP else None,
            output_path=output[0],
            show_progress=True,
        )


rule evaluate:
    """Stage 8 of 9. Plot the flagged streams, and measure them if there are labels.

    Plotting needs no ground truth and always runs. Metrics run only when `evaluate.labels`
    points at a file, and are pooled over every point of every stream rather than averaged per
    stream - a fleet is mostly quiet, and averaging would let a short normal stream count for
    as much as a long one with a failure in it.
    """
    input:
        scores=SCORES,
        # Counts and costs, so the summary carries what each stage bought as well as what it
        # found. Both are read defensively - a stage that was cached never re-timed, and a run
        # made before benchmarking existed has no TSV at all.
        stats=SCORE_STATS,
        diagnostics=COVARIANCE_DIAGNOSTICS,
        # Absent when the stage is switched off, and `evaluate` then reports the same metrics it
        # always did - the ranking ones simply do not appear.
        points=POINT_SCORES if USE_POINT_SCORES else [],
        operating=OPERATING_POINTS if USE_OPERATING_POINTS else [],
    output:
        summary=SUMMARY,
        figures=directory(FIGURES),
    benchmark:
        BENCHMARK.format(stage="evaluate"),
    run:
        import json

        # Whether the streams were interleaved with their own past. A doubled stream is
        # indistinguishable from an ordinary one once it is on disk, so this cannot be sniffed
        # and comes from the configuration that asked for it. Without it the metrics would be
        # measured in lead-lag indices - twice as many points, each anomaly apparently twice as
        # long - and would not compare with a run that had it off.
        LEAD_LAG = bool(config["preprocess"]["lag"]
                        and config["preprocess"]["lag_method"] == "lead_lag")

        # `truth=None` is what switches every label-dependent metric off - precision, recall,
        # F1, the AD ladder and the areas all live behind that one branch in
        # `evaluate_predictions`, so there is no second code path to keep in step.
        metrics = evaluate_predictions(input.scores,
                                       truth=ground_truth() if EVALUATE_METRICS else None,
                                       lead_lag=LEAD_LAG,
                                       point_scores=input.get("points") or None,
                                       show_progress=True)
        plot_scored_streams(
            input.scores,
            truth=ground_truth(),
            output_dir=output.figures,
            max_plots=config["evaluate"]["max_plots"],
            max_channels=config["evaluate"]["max_channels"],
            lead_lag=LEAD_LAG,
            show_progress=True,
        )
        # The system-level curve, if it was swept. Kept distinct from `roc_auc` in the name,
        # because the two measure different things and a reader who conflates them will draw the
        # wrong conclusion: one is the representation, the other the whole detector.
        if input.get("operating"):
            import pandas as _pd

            metrics.update(operating_point_auc(_pd.read_csv(input.operating)))
            print("  operating-point AUC {operating_point_auc:.3f} over "
                  "{operating_points} swept threshold(s)".format(**metrics))

        # What the run cost, beside what it found. `benchmark:` records seconds and memory;
        # these are the denominators that turn them into rates, and the two are only meaningful
        # together - forty seconds says nothing until you know whether it built thirty thousand
        # corpus intervals or three million.
        metrics["throughput"] = collect_throughput(
            benchmarks=discover_benchmarks(BENCHMARK_DIR, wildcards.dataset),
            artifacts={
                "corpus": CORPUS.format(dataset=wildcards.dataset),
                "covariance": COVARIANCE.format(dataset=wildcards.dataset),
                "index": (FOREST if USE_FOREST else INDEX).format(dataset=wildcards.dataset),
                "thresholds": THRESHOLDS.format(dataset=wildcards.dataset),
                "scores": input.scores,
            },
            score_stats=input.stats,
            covariance_diagnostics=input.diagnostics,
        )

        Path(output.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(output.summary).write_text(json.dumps(metrics, indent=2, default=str),
                                        encoding="utf-8")


rule report:
    """Stage 9 of 9. Collect the run into one readable document.

    Reads what the earlier stages wrote rather than recomputing any of it, so the report
    cannot disagree with the artifacts it describes. Figures are inlined as data URIs rather
    than linked: a report that only renders from the directory it was built in cannot be sent
    to anyone, and would not survive `clean`.

    The configuration is included for the same reason the thresholds are. A distance of 1994
    means nothing on its own - it is a fact about a particular truncation level, granularity
    and retained variance, and those belong on the same page as the number they produced.
    """
    input:
        scores=SCORES,
        summary=SUMMARY,
        thresholds=THRESHOLDS,
        figures=FIGURES,
    output:
        REPORT,
    run:
        build_report(
            output[0],
            dataset=wildcards.dataset,
            # Two distinct things, previously passed as one: the pooled metrics, and the
            # per-stream table. They now render under their own headings.
            metrics=input.summary,
            per_stream=input.scores,
            thresholds=input.thresholds,
            figures_dir=input.figures,
            config=config,
            show_progress=True,
        )


rule clean:
    """Remove every derived artifact, leaving data/raw untouched."""
    run:
        import shutil

        for path in ("data/projected", "data/preprocessed", "data/corpus", "data/index",
                     "results", "reports", "figures"):
            if Path(path).exists():
                shutil.rmtree(path)
                print("removed", path)
