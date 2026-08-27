# Remaining work

State as of 2026-08-26. Written to be picked up cold: each step says what it is, why it matters,
what it touches, how to tell it worked, and roughly what it costs.

## Where things stand

Landed recently, so the numbers below are the current ones:

| change | effect |
|---|---|
| `metric.file_format: npz` | covariance 8.8 MB → 3.2 MB, 5.1× faster to read, and exact where CSV lost 1 ulp |
| `benchmark:` on 13 rules + `throughput.py` | per-stage wall time, memory, and derived rates in the summary |
| `score_streams.py` | CLI for the whole workflow, with `--set`, `--summary`, `--unlock` |
| `point_scores` stage | continuous score per point → ROC-AUC, PR-AUC |
| `operating_points` stage | detector re-run at scaled thresholds → system-level AUC |
| Exathlon AD1–AD4 | range-based precision/recall at four cumulative levels |
| report restructure + `evaluate.metrics` | detection and evaluation are now separable |
| `detect.neighbours: 3` | **F1 0.573 → 0.636 on FD001**, and +0.05–0.07 on the other three |

FD001 at current defaults: `P 0.839 R 0.512 F1 0.636`, adjusted F1 0.938, ROC-AUC 0.902,
PR-AUC 0.689, operating-point AUC 0.897, AD1–AD4 = 0.884 / 0.628 / 0.528 / 0.452.

---

## 1. Decide the `operating_points` grid before the final run

**What.** `evaluate.operating_points.scales` currently holds seven multipliers. Each one is a
complete widest-first search over every test stream.

**Why it matters.** The search is ~25 s on FD001 but ~150 s on FD004, so seven scales across four
subsets is roughly 1.5–2 hours of the final run. `[0.5, 1.0, 2.0, 4.0]` still brackets the F1
optimum (which sits between 0.5 and 1.0 — see below) and roughly halves that.

**Context.** The FD001 sweep at `neighbours: 3`:

```
scale   flagged        P       R      F1
 0.50    34.67%    0.401   0.925   0.559
 1.00     9.18%    0.839   0.512   0.636   ← calibrated setting, and the best of the seven
 2.00     2.22%    1.000   0.148   0.257
```

Calibration is sitting on the optimum, which is a stronger endorsement of `calibrate.statistic:
p95` than the earlier p80-vs-p95 argument managed. Worth keeping the sweep for that reason alone.

**Verify.** `python score_streams.py --summary` shows `operating_point_auc` and the count of
points behind it.

---

## 2. A run-comparison command

**What.** `score_streams.py --compare`, globbing `results/*_summary.json` into one table:
subset, detector, F1, adjusted F1, ROC-AUC, PR-AUC, AD1–AD4, points/sec, peak RSS.

**Why.** The final run produces four or more summaries and nothing currently reads more than one.
This is the difference between a paste-able results section and eight JSON files.

**Touches.** `score_streams.py` (a new branch beside `--summary`), or a `main()` in
`throughput.py`. ~40 lines.

**Verify.** Run it over the four C-MAPSS summaries and check the table matches what
`--summary` prints for each individually.

---

## 3. `git init` and a `.gitignore`

**What.** There is no version control. No `.git`, no `.gitignore`.

**Why.** The project brief's first named deliverable is *"a reproducible repository with
Snakefile, config.yaml, environment.yml or pyproject.toml, a notebook…, a command-line entry
point"*. Every component exists; the repository does not.

**How.** `.gitignore` first — the directory is ~9 GB and almost all of it is derived or vendored:

```
ignore                                    keep
data/            ~7,300 MB                src/              0.7 MB
.venv/            1,025 MB                notebooks/        5.6 MB
OmniAnomaly-master/ 464 MB                config/, Snakefile, score_streams.py
NAB-master/         306 MB   (see 10)     environment.yaml, tests/, experiments/
reports/            214 MB                NEXT_STEPS.md
figures/            161 MB
results/, CMAPSS/, .snakemake/, __pycache__/, Project Files/
```

`Project Files/` is 11 MB of papers (HNSW 1603.09320, SigNova 2402.14892, six others) — keep
locally, don't commit: size and copyright both argue against it. `benchmarks/` is a judgement
call: the TSVs are tiny and are the throughput evidence, so tracking them is defensible.

**Verify.** `git status` shows ~7 MB staged, not gigabytes.

---

## 4. Experiment scripts — decided against (2026-08-27)

**Decision.** The development scripts behind the sweeps are **not** going into the repository.
They stay in the temp scratchpad and go with it.

**What this means in practice.** The measurement *tables* survive — they are written into
`config.yaml` beside the settings they justify: the window-size sweep under `window.size`, the
retained-variance reasoning under `metric.variance_keep`, the detector comparison under
`detect.method`, the neighbour sweep under `detect.neighbours`, the lag comparison under
`preprocess.lag_method`, the dead-channel argument under `detect.off_manifold`. What is lost is
the path to *re-run* them, not the record of what they found.

Several would not have run against the current API anyway - they import `create_corpus_merged`,
which was dissolved, and hardcode `_covariance.csv` paths that are now `.npz`.

**If the write-up needs to describe how a default was chosen**, the config comments are the
source. Cite them as measurements made during development rather than as reproducible artifacts.

**Consequence for sequencing:** this was the only item that had to happen before the first
commit. Nothing now blocks `git init` except the README, the licence, and deciding where the
project brief lives.

---

## 6. The featuriser abstraction, then the summary-statistic baseline

**What.** The project brief's research question — *"When do signature features beat ordinary
summary statistics, and when are they overkill?"* — **has never been tested.** There is no
summary-statistic comparison anywhere.

**Why it is blocked.** The featuriser is hardcoded in two places that must agree:
`signature_computer.interval_signatures` (builds the corpus) and
`stream_scoring.StreamScorer.signature` (builds query vectors on the fly). Both call
`base_pointed_signature`. They need a shared pluggable featuriser — `featurise(path_slice) ->
ndarray` plus a name and a width→dimension function — selected by a `features.method` config key.
`StreamScorer.truncation` also infers the level from the index width, which is meaningless for a
non-signature featuriser; there is precedent for the escape hatch, since `trunc` is already
passed explicitly when `signature_umap` is on.

**Then the arms are cheap:**

```
signature trunc 2      650 terms at 25 channels  (current)
signature trunc 3      the brief caps at "2 or 3"
log-signature trunc 3  5,525 terms — code already written in run_logsig.py, validated
summary statistics     per channel: mean, std, min, max, first, last, slope → ~7d
raw resampled window   flattened
```

**One asymmetry to handle.** Summary statistics are width-agnostic (mean, std, slope are defined
at any interval length) so they drop straight into the variable-width dyadic corpus. Raw
flattening is **not** — it needs each interval resampled to a fixed length first. State that in
the write-up rather than resampling everything silently.

**With AUC now in the pipeline** this comparison is much sharper than it would have been: report
signatures vs summary stats on ROC-AUC, PR-AUC, AD1–AD4 *and* F1 on identical windows.

---

## 7. Multi-seed whatever you quote

**What.** Every number produced so far is a single seed.

**Why.** The ridge sweep showed adjacent settings swinging by ±0.05 where they should have behaved
near-identically — FD001 went 0.659 → 0.446 → 0.592 across γ = 0.1, 0.3, 0.5. **Differences below
about 0.02 cannot be resolved at one seed**, and several effects of interest are that size.

**How.** ≥3 values of `calibrate.random_state`, report mean ± spread. Only for the numbers that go
in the write-up — not the whole ablation grid.

---

## 8. Run the throughput sweeps

**What.** The instrumentation is built; no measurements have been taken.

```
corpus size    subsample 25 / 50 / 100% of streams
dimension      umap.dimension 4 / 8 / 16 / 24 / 48  → 20 / 72 / 306 / 650 / 2,401 terms
index type     detect.index.type exact / ivf / hnsw
```

**Why these three.** Exact FAISS search is O(N) in corpus size — that is the scalability wall, and
the justification for the index-type axis. Everything is linear in dimension except the
covariance fit, which is **D²**. `detect.index.type` has never been swept and is the most
production-flavoured axis available.

**Two traps.** Cached stages never write a benchmark TSV, so build timings need forced reruns.
And `--cores` defaults to every core on the machine — pin it or the numbers won't compare.

**Each index type needs its own `calibrate` run.** Approximate search can only miss a nearer
neighbour, so distances inflate and thresholds do not transfer.

---

## 9. Factored covariance

**What.** Store the rank-*r* factor instead of the dense D×D matrix. `covariance_matrix` already
computes `kept` (r×D, orthonormal rows) and `scale` internally, then densifies to `(kept.T *
scale) @ kept` and throws the factor away.

**Worth.** Rank is 8–11 of 650:

```
dense   650 × 650 × 8 B = 3.38 MB       whitening  N·D²      = 1.24e10 flops (FD001)
factor   11 × 650 × 8 B =   57 KB       factored   2·N·D·r   = 4.2e8         → ~30× fewer
```

59× smaller on disk, ~30× fewer flops in the whitening — a genuine throughput result, not a
storage tidy-up.

**Cost.** Changes the interface: everything downstream does `X @ covariance.T`, so it needs a
small object with an `apply(X)` method plus a `.dense()` for `check_is_whitening`. Do it *after*
step 8, so the benchmark harness can show what it bought.

---

## 10. NAB removal

**What.** Remove NAB as a dataset from the project.

```
NAB-master/                          306 MB vendored repo — delete or move outside the repo
legacy/notebooks/NAB_notebook.ipynb  the old NAB experiments — keep under legacy/
Snakefile:125                        one passing mention in a comment
src/anomalies_scale/*.py             one passing mention in a comment
```

Removal is clean — nothing in `config/`, the Snakefile logic, or `src/` depends on it.

**Declare the deviation.** The project brief names NAB as the **primary** labelled benchmark
(*"NAB as the labelled benchmark, C-MAPSS as the multivariate safety dataset"*). Dropping it is a
departure from the specification and must be stated in the write-up, not left implicit.

**The defensible justification** is Wu & Keogh, *"Current Time Series Anomaly Detection Benchmarks
are Flawed and are Creating the Illusion of Progress"* (TKDE 2021), which criticises NAB
specifically — alongside Yahoo, NASA SMAP/MSL and OMNI/SMD — for triviality, unrealistic anomaly
density and mislabelling. Pair that with the positive argument: C-MAPSS is multivariate where NAB
is univariate, and Exathlon supplies both scale and labelled anomaly *types*, which is what the
"what do signatures detect well" question needs and NAB cannot provide.

**Keep the legacy notebook.** It contains the only LOF-with-Mahalanobis comparison in the project
(`LocalOutlierFactor(novelty=True)` on `train_whitened`, swept over k ∈ {1,5,10,20}). It has no
saved outputs, so there are no numbers — but the code is the record that the comparison was
considered.

---

## 11. Exathlon run

**What.** Port the pipeline to Exathlon (Jacob et al., VLDB 2021) as the scale-and-anomaly-type
dataset.

**Verified specifications** (from the paper, not memory):

```
93 traces  =  59 undisturbed  +  34 disturbed        97 anomaly instances, 6 types
2,283 metrics per trace at 1 Hz     243 Spark driver + 700 executor (5×140) + 1,340 OS (4×335)
~7 hours per trace; 649 hours, 2,335,781 items, 24.6 GB total
CC BY-NC-SA 4.0 · github.com/exathlonbenchmark/exathlon
```

**59 undisturbed traces is the corpus** — 1.4M items of genuinely clean operation, not a
train/test cut of contaminated data.

**Labels are richer than binary**, and this is the main attraction. Each anomaly carries two
nested intervals:

```
(app_id, trace_id, anomaly_type,
 root_cause_start, root_cause_end,          when the disturbance was applied
 extended_effect_start, extended_effect_end) when metrics returned to normal
```

Build **both** label files (RCI-only and RCI∪EEI) and make RCI∪EEI primary — it's what the
paper's AD levels score against. Flagging inside the effect interval but not the cause is a
meaningfully different result and both are worth reporting.

### The blocker: 2,283 channels

Signature dimension is roughly `width^trunc`, so 2,284 channels with time adjoined at truncation
2 is `2,284 + 2,284² ≈ 5.2 million` terms. **Unbuildable** — the covariance alone would be
5.2M × 5.2M.

So `umap.dimension` stops being optional and becomes load-bearing for the first time. The
arithmetic is tidy: **reduce to 24 latent channels** and you get `25 + 625 = 650` terms, exactly
the dimension C-MAPSS runs at, so every measurement transfers as a reference point.

This reframes the UMAP work. On C-MAPSS the channel reduction measured badly (best 0.434 against
0.573 unreduced) — but there it was solving a problem that didn't exist. Here there is no
unreduced baseline to lose to; the only question is *which* reduction.

### Sizing

At 1 Hz a trace is ~25,000 points. `window.size: 16` would give ~87,000 windows × 36 grid
intervals ≈ **3.1M corpus rows** (~16 GB at 650 dims) — not viable, and physically wrong:
anomalies here last 1 minute to 2.8 hours, where C-MAPSS's 16-cycle window matched a 31-cycle
degradation. **Start at `window.size: 128` or `256`** — a few hundred thousand intervals.
`detect.nn_reference_size` is the fallback.

### Ingestion steps

1. **Loader.** `download_corpus` assumes URL lists; Exathlon ships zips with `extract_data.sh`.
   New `canonicalise` path producing `(stream, Time, Stream)`, one trace per stream, Time in
   seconds. 59 undisturbed → `corpus/`, 34 disturbed → `test/`.
2. **Nulls.** 700 of 2,283 columns are executor metrics with *"null values set for inactive
   executors"*. Decide before anything else: drop above a threshold, forward-fill, or zero-fill
   with an indicator. This is not cosmetic — a channel that is null-then-constant is annihilated
   by the pseudo-inverse, which is the dead-channel failure `detect.off_manifold` exists for and
   that the ridge sweep failed to fix.
3. **Labels.** Convert the ground-truth tuples to the per-stream `[[lo, hi]]` convention, both
   variants.
4. **Provenance.** Carry `app_id` (job type) and `anomaly_type` through as stream metadata.
   `anomaly_type` enables per-type recall — the most interesting breakdown Exathlon offers and
   something nothing in `stream_evaluation` currently produces.

### Two things that will bite

**Ten job types.** The 93 traces come from 10 streaming jobs — the same heterogeneity axis as
C-MAPSS's operating conditions but more of it, and 6 conditions is exactly where this pipeline is
weakest (FD002 0.473, FD004 0.564 at `neighbours: 3`, against 0.636/0.615 for single-condition).
Expect that amplified. The mitigation is per-job corpora, which is the regime-wise treatment
that has been on the list unbuilt for a while, and `app_id` makes it natural.

**Anomaly rate.** By arithmetic off the paper's table (not a stated figure), anomalous time is
roughly 4–5% of the full 649 hours and ~12% within the disturbed traces. Sparser than C-MAPSS's
15%, so precision will carry more weight.

### The AD levels already match

Exathlon evaluates with Tatbul et al.'s range-based precision/recall at four cumulative levels —
which is **already implemented** in `stream_evaluation.range_metrics`. Note the caveat recorded
there: the paper defers its exact (α, δ, γ) parameters to a technical report that is not public,
so our implementation is faithful to the stated semantics rather than bit-comparable with the
paper's tables. Say so if you quote both.

---

## 12. Notebook reruns and cleanup

Three notebooks: `CMAPSS_notebook.ipynb` (1.75 MB), `Omni_notebook.ipynb` (1.79 MB, broken),
`toy_notebook.ipynb` (2.02 MB).

**a. Pin the kernel.** `CMAPSS_notebook.ipynb`'s `kernelspec` says `python3`, which resolves to
system Python 3.12 without `iisignature`. Any headless execution fails — CI, a make target,
`nbconvert`. It only works interactively because VS Code lets you pick the kernel by hand. The
venv kernel is registered as `anomalies-at-scale`. Either set the metadata, or always run:

```
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=anomalies-at-scale --ExecutePreprocessor.timeout=3600 \
  notebooks/CMAPSS_notebook.ipynb
```

**b. Reconcile sections 7 and 8.** They now use **different threshold schemes**. Section 8 was
moved to the pipeline's k=5 folds over engines with a per-depth threshold; section 7 still uses
the old leave-one-engine-out with a threshold pooled across depths. Section 7 was left alone
because its threshold and its detector share a whitening fitted on the fit engines, and moving
one without the other would put scores and threshold in different spaces — but moving both
changes section 5's detector results. Decide and make them consistent.

**c. Rerun against the new defaults.** The notebook was last executed before `neighbours: 3`.
Its narrative quotes pipeline numbers (0.573 on FD001) that are now 0.636. Every such figure in
the prose needs checking after a rerun.

**d. `PERCENTILE = 99.9` is still the notebook's operating point** against the pipeline's p95, and
section 8b's own table shows what that costs: FD002 gives F1 0.140 at p99.9 against 0.537 at p95,
FD004 0.155 against 0.565. The thresholds rest on 4 and 5 calibration intervals respectively.
This is the single largest remaining discrepancy between notebook and pipeline.

**e. `Omni_notebook.ipynb` is broken** — by the `create_corpus_merged` dissolve. Fix it or move
it to `legacy/notebooks/` beside the NAB one. A broken notebook in a repo assessed on
reproducibility is worse than an absent one.

**f. Backup.** `scratchpad/CMAPSS_notebook.backup.ipynb` is the pre-k=5 version. Move it
somewhere durable before the scratchpad is cleared, or discard it deliberately.

---

## Traps that have already cost time

- **Snakemake does not track config read inside `run:` blocks.** Changing a config value alone
  will leave stages looking up to date. Delete the downstream artefacts, or `--forcerun`.
- **`detect.band` is baked into the index metadata** (`load_index` reads `band=int(meta["band"])`),
  so changing it needs the index deleted, not just the thresholds.
- **Snakemake will not rebuild a deleted intermediate** while the final target still looks
  current. Delete the report too, or force.
- **A stale lock** from an interrupted run fails the next one: `python score_streams.py --unlock`.
- **`--cores` defaults to every core**, which is right interactively and wrong for benchmarking.
- **PowerShell `Select-Object -First N` kills the upstream process**, which surfaces as a spurious
  non-zero exit code from an otherwise successful command.
