# UEQ roadmap: September 2026 to September 2027

Status: proposal for maintainer review, dated 2026-09-25. The evidence behind it is in [USE_CASES.md](USE_CASES.md): a code audit, a competitor survey, and eight domain studies, each checked by a second reviewer whose job was to refute it. A draft of this roadmap was then reviewed by a critic, and its flagship was tested with a working prototype on the real dataset. Both reviews changed the plan; section 5 records how.

## Summary

- UEQ's easiest path gives wrong answers today. `UQ(LinearRegression())` returns 95% "prediction intervals" that cover 7.7% of outcomes, and `UQ(RandomForestRegressor())` covers 48.2%. On NASA C-MAPSS engine data, the 1.0.1 default covers 22-36% of true remaining useful life (RUL) at a nominal 90%, depending on the setup. `UQ(LogisticRegression())` returns integer labels plus or minus a number instead of prediction sets. Fixing this comes before any repositioning.
- Only a small core is statistically sound: split conformal regression, global LAC conformal classification, rolling-window online conformal, and the basic interval metrics. That core is a good foundation.
- All eight real-world domains scored 3 or 4 out of 10 for UEQ, mostly on differentiation. MAPIE 1.5, crepes, darts, Nixtla, sktime, TorchCP, UQLM and LM-Polygraph already ship the methods. UEQ should stop adding methods.
- New position: UEQ audits prediction intervals and sets after deployment, whichever library produced them. It reports coverage per segment and over time with uncertainty that respects clustered data (many rows per engine, patient or store), handles labels that arrive late, and states when a failure is conditional rather than marginal. It repairs marginal coverage post hoc without touching the model. It keeps a small, validated split-conformal core so that `UQ(model)` is correct by default.
- First proof: an engine-clustered coverage audit on NASA C-MAPSS turbofan data. A prototype run on the real data showed it is feasible, with caveats that reshape the demo. Marginal coverage stays near 0.90 even under the fault-mode shift, so rolling windows and ACI have nothing to repair. The failure is conditional: engines with a fault mode absent from training get much lower coverage, with misses on the unsafe side, and only a segment-level, engine-clustered, delay-aware audit sees it. UEQ's own intervals were the widest in the comparison, so UEQ is positioned as the auditor, not the best producer.
- The plan assumes one maintainer at 5-8 hours a week. Each phase cuts before it adds, and each phase says what slips first. A go/no-go gate at month 6 is based on external use of the evaluation API released at month 3.

## 1. Where UEQ stands today

Every figure below was measured by running the code (details in USE_CASES.md). The "on PyPI" column matters: PyPI still serves 1.0.1 (2025-09-29). Everything merged since then, including PR #30, has never been released.

| Component | On PyPI 1.0.1? | Status | Evidence |
|---|---|---|---|
| Split conformal regression, residual score | yes | Correct under exchangeability | 0.906 coverage at a 0.90 target over 200 trials. C-MAPSS FD001, 20 test engines: 0.882 / 0.913 / 0.912 over 3 seeds, width 67-69 cycles |
| Asymmetric signed-residual ('quantile') score | no | Correct, but not CQR | 0.902 over 200 trials. C-MAPSS: 0.906-0.912 |
| Global LAC (`inverse_probability`) sets | yes | Correct when labels are 0..K-1 | 0.899 at a 0.90 target |
| `OnlineConformalUQ`, rolling window | no | Works as a heuristic | After a noise shift, static coverage fell to 0.416; rolling (w=500) recovered to 0.869. No prefit mode, symmetric scores only, no delayed-label API; the default window of 500 inflated C-MAPSS widths from 67 to 78-88 cycles because feedback arrives in engine-sized bursts |
| coverage, width, Winkler interval score | partly (Winkler added after 1.0.1) | Correct | Probed; `coverage` and `interval_width` matched an independent audit exactly on C-MAPSS |
| Top-label ECE in `diagnostics` | no | Correct | Probed |
| Bootstrap, the auto default for every sklearn regressor | yes | Not a prediction interval (it is a confidence interval of the mean) | 95% intervals cover 7.7% (LinearRegression) and 48.2% (RandomForest); another probe gave 10.6% and 56%. C-MAPSS at 90%: 0.217-0.256 with the default 100 models (width about 11 cycles, 34 s per seed); 0.356-0.358 in a second setup |
| `DeepEnsembleUQ` | yes | Epistemic only | 7% coverage at a nominal 95% |
| `BayesianLinearUQ` | yes | Noise precision is fixed | 0.7% to 100% coverage depending on the scale of y. Its `alpha` means prior precision |
| `UQ(classifier)` auto-routing | yes | Broken | Runs conformal in regression mode and returns label +/- q. SVC and RidgeClassifier are routed to bootstrap |
| Labels other than 0..K-1 | yes | Broken | String labels crash; -1/+1 labels silently give trivial sets; sets hold column indices, not labels |
| `UQ.predict` with MC dropout; conformal with multi-output y | yes | Silently wrong | MC dropout's (mean, std) is passed through as (mean, intervals); multi-output y gives a meaningless q |
| Small calibration sets | yes | Guarantee silently lost | Every quantile branch clamps to the largest score instead of returning an infinite bound or the full label set |
| `nonconformity='normalized'` | no | Broken | 71%, 19% and 4% coverage at a 90% target in three probes |
| `nonconformity='margin'` | no | Broken | Returns every class for every input |
| `class_conditional=True` (Mondrian) | no | Broken | Minority-class coverage 22-61% at a 90% target; 0.287 on a 5%-minority binary task |
| `AdaptiveConformalUQ` | no | Recalibration does nothing | q identical to the plain rolling window while `recalibration_count` reached 300 |
| `DriftAwareRecalibrator`, `UncertaintyInflator` | no | Does nothing / heuristic | Drift score 0.0 under a 3-sigma shift while coverage fell to 0.47; q never changes |
| `UQMonitor`, `UQ.monitor` | yes | Broken | Ignores its constructor baseline, so drift is always 0.0; never sees y; `UQ.monitor` has a bare `except` |
| `CrossFrameworkEnsembleUQ` | yes | Broken | Drops the PyTorch member with a `print()` and carries on |
| Interval ECE/MCE in `metrics.py` | yes | Meaningless | Bins by array position: 0.41 for exactly calibrated intervals |
| `check_calibration` | no | False alarm | Flags valid split conformal as miscalibrated because its width is constant |

Process and packaging:

- There is no CI; `.github/` holds only `FUNDING.yml`. The 76 tests pass, but the assertions are loose (0.7 <= coverage <= 1.0, 0 <= ECE <= 1). The online, recalibration, monitoring, performance, cross-ensemble and benchmark modules have no tests.
- torch and matplotlib are hard dependencies. `import ueq` takes 2-3 s and needs PyTorch, even for scikit-learn users. `setup.py` still claims Python 3.8 support.
- `setup.py` says 1.0.2, the README header says v1.0.1, and only tags v1.0.0 and v1.0.1 exist. The README says "MIT License", but `LICENSE` and `setup.py` say Apache-2.0; the README is also the PyPI long description. The README's monitoring and xgboost cross-framework examples do not run as written, and `examples/benchmarks/finance_stock_volatility.py` needs the undeclared `yfinance`.
- The README's own roadmap section promises evidential, BNN and Laplace work in v1.2.x and plugin, distributed and structured-output work in v2.0. The `setup.py` description ("Phoenix Edition ... production-ready features") and keywords are the PyPI summary. `docs/PRODUCTION_GUIDE.md`, `docs/EXAMPLES.md`, `examples/production_demo.py` and `examples/cross_framework_demo.py` document the broken monitor, cross-ensemble and batch-size optimizer.
- 25 of the last 30 commits came from a coding bot with no statistical review. PR #30 implemented nine issues in one merge (#5, #6, #8, #11, #12, #15, #18, #22, #27) and brought in most of the broken paths above, with all tests passing.
- The only visible external contributor, @refexa, commented on #17 on 2026-02-17 ("Issue resolved, check and feedback please"); the maintainer asked for a PR.

## 2. Positioning

**Statement.** UEQ is a small, scikit-learn-first Python toolkit for one question about a deployed model: do its prediction intervals and sets still hold, per segment and over time, as late labels arrive? It accepts intervals, quantiles or sets as arrays from any library, with first-class MAPIE and crepes adapters in 1.x, and forecasting-library adapters only if the month-6 gate passes. It reports coverage with uncertainty that accounts for clustered rows, separates marginal from conditional failures, and repairs marginal coverage post hoc with conformal recalibration and adaptive conformal inference (ACI). It keeps its own split-conformal producer small and correct.

**Who it is for.**
- Data scientists and ML engineers whose model already emits intervals, quantiles or sets, whose ground truth arrives late, and who need coverage evidence for themselves or for a reviewer. Examples of late ground truth: remaining useful life is known when the unit fails, demand at the end of the horizon, loan outcomes months later.
- scikit-learn and LightGBM users who want a correct default interval without installing PyTorch.

**What it is not.**
- Not another library of UQ methods. Evidential, BNN/VI, Laplace, SWAG and similar belong to Lightning-UQ-Box, laplace-torch and TorchUncertainty. CQR, APS/RAPS, Mondrian, risk control and conformal predictive systems belong to MAPIE and crepes.
- Not a fix for conditional failures. When the audit finds a segment that under-covers, UEQ reports it; the fix is a better producer (CQR, Mondrian, normalized scores in MAPIE or crepes) or more training data, not a UEQ recalibration.
- Not a forecasting framework (darts, Nixtla, sktime), an LLM uncertainty tool (UQLM, LM-Polygraph), or a general drift and observability platform (Evidently, NannyML).
- Not a compliance product. Evidence reports support documentation, and UEQ never claims conformity with any regulation.

**Library shape after the redirect.**

1. *Core*: a small set of validated producers. These are split conformal regression (residual and asymmetric scores), LAC prediction sets, and `conformalize()`, which makes any upstream interval marginally valid on a calibration set. This is the default path of `UQ(model)`.
2. *Evaluate*: array-in functions for coverage, lower and upper miss rates, width, Winkler and pinball scores, and set size. Confidence intervals take a `cluster=` unit id and use a cluster bootstrap or per-unit coverage; without it they warn that rows are assumed independent. Breakdowns by group and time window use quantities known at prediction time. Outcome-conditional breakdowns (for example by true RUL) are available only as labelled diagnostics compared against an in-distribution baseline, not against 1-alpha.
3. *Monitor and repair*: an outcome ledger that records prediction time, resolution time and attributes that arrive with the label; coverage reported per prediction cohort once the cohort has resolved; a stated, multiplicity-controlled breach test; ACI over any score stream; and a versioned JSON evidence report.

**Why this position.** The landscape survey found a gap after the interval producers. No open-source tool packages coverage monitoring of deployed intervals with delayed labels, per segment and horizon. None evaluates time-series conformal methods independently of the library that produced them. uncertainty-toolbox, the closest evaluation toolkit, has not released since January 2023 and assumes Gaussian outputs. MAPIE issue #505 is a public example of the need: CQR plus production recalibration for energy-load forecasting, closed as "Discussion in progress". The C-MAPSS prototype added a concrete reason: on real data, the interesting failures were conditional, clustered and hidden by label delay, which is the part the producers do not report.

The honest counterweight: the reviewers judged this glue thin. MAPIE's `RiskMonitoring` plus a groupby covers part of it. OpenSTEF replaced a MAPIE-based pull request with about 50 lines of inlined conformal code (PR #1060, merged 2026-08-18). The plan therefore stays dependency-light and gets UEQ correct first. It then puts a gate at month 6 that asks for evidence of external use before building further.

## 3. Beachhead: an engine-clustered coverage audit on NASA C-MAPSS

### Why C-MAPSS and not the alternatives

The main disagreement between the proposals was the flagship dataset. One proposal wanted Elia's published P10/P90 wind and solar forecasts. That is the purest "audit someone else's intervals" demo, and it matches MAPIE #505. Elia failed verification, though. Its site was blocked in every attempt, so the licence (CC BY 4.0) and the field names are known only from search snippets.

C-MAPSS is the only verified, credential-free regression dataset in the research with enough independent units and with delayed labels, segments and a shift built in. NASA Milling was also downloaded and parsed, but has only 16 tools. HarvestStat Africa is verified but gives roughly one exchangeable unit per year per country. BANKING77 and CLINC150 are verified but are classification tasks that need prefit and label-encoding work first. C-MAPSS can still show the "audit someone else's intervals" thesis: the prototype audited intervals from MAPIE, crepes and bootstrap next to UEQ's own, on identical splits. Elia becomes the Phase 3 second demo if its licence is verified.

| Candidate | Data access | Licence | Ready with today's correct code? | Decision |
|---|---|---|---|---|
| NASA C-MAPSS (industrial RUL) | Verified twice: HTTP 200, 12,429,152 bytes, SHA-256 pinned, FD001-FD004 parsed | Unverified; likely US-government work (see Dataset) | Yes: residual split conformal ran end to end in the prototype | **Beachhead** |
| Elia P10/P90 wind and PV (energy) | Blocked in every attempt | CC BY 4.0 per snippets only | Needs evaluation + ACI | Phase 3, only if licence and fields are verified |
| HEFTCom2024 (energy) | NWP features gated on IEEE DataPort | Mixed/unverified | No | Rejected; the "calibration = revenue" claim was refuted |
| UCI Taiwan credit default (finance) | Corroborated secondhand only; archive.ics.uci.edu was never reached | CC BY 4.0 (corroborated) | No, classification paths are broken | Phase 3 benchmark, after access is verified |
| BANKING77 intents (LLM routing) | Verified | CC BY 4.0 | No, needs prefit + label encoding | Phase 3 optional example |
| HarvestStat Africa (agri) | Verified, CSVs in git | MIT | Partly | Optional community example; effective calibration n is about the number of years |
| M5, Favorita, Olist, Home Credit, BAF | Kaggle login | Competition rules or non-commercial | - | Rejected for CI and loaders |
| PhysioNet 2019 sepsis (health) | Unverified | Unverified | No | Rejected; the "defer band" defers 31-69% of patient-hours |

C-MAPSS is a proving ground, not a market position. The industrial reviewer scored the domain 4/10, rul-datasets already ships C-MAPSS loaders, and crepes and MAPIE already have sharper interval methods; the prototype confirmed that on the same splits. The benchmark makes no maintenance-cost claims.

### Dataset

- **Name:** NASA PCoE Turbofan Engine Degradation Simulation (C-MAPSS), subsets FD001-FD004.
- **URL:** https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- **Citation (from the readme in the archive):** A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
- **Verified contents:** the outer zip holds only `6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip`; the inner zip has 14 files (readme, the PHM08 paper, and train, test and RUL files for FD001-FD004; 45.3 MB unzipped). Each row has unit, cycle, 3 operating settings and 21 sensors (the readme says "26 columns" but numbers sensors up to 26). FD001 train is 20,631 rows, 100 engines, lives of 128-362 cycles. FD003 train is 24,720 rows, 100 engines, lives of 145-525. FD002 (48,759 rows) and FD004 (61,249 rows) have six operating regimes. Train/test engines parse as 100/100, 260/259, 100/100 and 249/248 (the readme says 248/249 for FD004). The readme gives two fault modes (HPC and fan degradation) for FD003 and FD004 but does not say which engine has which. Seven FD001 sensors are constant.
- **Train vs test files:** train trajectories run to failure; test trajectories end before failure. Anything that needs labels to resolve at failure, including the fleet replay, uses train files only. Test files are used only for the official last-cycle protocol.
- **Licence:** unverified. The archive contains no licence, copyright or terms text in any file, including the PDF. data.nasa.gov, catalog.data.gov, nasa.gov and kaggle.com were blocked. One web-search summary, probably of a Kaggle mirror, calls it a U.S. Government Work; that is not a primary source.
- **Access rule:** download at runtime into a user cache and check the byte size and SHA-256 (`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`; fetched twice on 2026-09-25 with a plain HTTPS GET in under a second). Parse the inner zip in memory. Never vendor or redistribute. Print the citation. In CI, cache the zip and skip (not fail) on network errors. Unit tests use a tiny synthetic fixture. Avoid N-CMAPSS (15.7 GB).
- **Caveats to state in the docs:** simulated data; no right-censoring in the train files; RUL is conventionally capped at 125, and 26-49% of rows sit on that plateau (49% in FD003), where almost every interval covers, so marginal coverage is inflated; within-engine rows are strongly dependent; licence unverified.

### What the feasibility prototype measured

A prototype script (about 500 lines, outside the repo) ran the flagship on the real data on 2026-09-25 with UEQ 1.0.2 from `main`, MAPIE 1.5.0 and crepes 0.9.1. Verdict: feasible with caveats. The whole script took 131 s on 4 CPU cores, of which 102 s was UEQ's default 100-model bootstrap. A follow-up ran 20 engine-grouped re-splits of the FD002 to FD004 shift in 30 s. `conformalize()`, delayed-feedback ACI, the outcome ledger, `coverage_report` with groups and clustered CIs, and the JSON evidence report were written as 10-40-line stand-ins because UEQ does not have them yet. Each needs its own validity test before it ships.

**Static audit, FD001** (engine-grouped 60 train / 20 calibration / 20 test engines, HistGradientBoostingRegressor, 90% target, 3 seeds; row counts 3,833-4,235 per test set):

| Producer | Coverage | Mean width (cycles) | Coverage without cap rows | Rows below the lower bound | True RUL 31-60 (outcome-conditional diagnostic) |
|---|---|---|---|---|---|
| UEQ split conformal, residual | 0.882 / 0.913 / 0.912 | 66.8-69.3 | 0.82-0.89 | 5-12% | 0.62 / 0.78 / 0.83 |
| UEQ split conformal, asymmetric | 0.912 / 0.910 / 0.906 | 63.7-70.5 | 0.88-0.91 | 3-7% | 0.71 / 0.83 / 0.88 |
| 1.0.1 default `UQ(HGB)` (bootstrap, 100 models) | 0.228 / 0.217 / 0.256 | 11.1-11.4 | 0.24-0.28 | 27-42% | 0.22 / 0.19 / 0.30 |
| Bootstrap + `conformalize()` stand-in | 0.886 / 0.911 / 0.922 | 61.0-65.2 | 0.82-0.89 | 5-11% | 0.66 / 0.79 / 0.85 |
| UEQ residual, row-random calibration (leaky) | 0.862 / 0.903 / 0.889 | 60.7-62.2 | 0.80-0.86 | 6-13% | 0.60 / 0.77 / 0.82 |
| MAPIE 1.5 CQR | 0.931 / 0.939 / 0.931 | 52.8-57.0 | 0.90-0.93 | 2-6% | 0.82 / 0.85 / 0.92 |
| crepes 0.9.1 normalized CPS | 0.910 / 0.906 / 0.935 | 53.5-64.8 | 0.88-0.93 | 2-6% | 0.83 / 0.85 / 0.94 |
| crepes Mondrian by predicted stage | 0.877 / 0.910 / 0.918 | 61.3-64.1 | 0.81-0.88 | 5-12% | 0.68 / 0.80 / 0.82 |

What this shows:
- The 1.0.1 default fails badly: 22-26% coverage here, and 36% in an earlier setup (30 models, calibration engines added to training, a cycle feature). Quote it as "about 0.2-0.36, setup-dependent". `UQ(HGB, alpha=0.1)` raises `TypeError`.
- Grouped split conformal lands near 0.90, but UEQ's own intervals are the widest and the least even. MAPIE CQR had the best Winkler score (58-71 against 84-100 for UEQ residual) and the most even coverage across RUL. UEQ is useful here as the auditor.
- Misses are almost all below the lower bound: the engine fails earlier than the interval allowed. That is the unsafe side for maintenance planning. The residual interval misses above the upper bound on only 0.2-3.7% of rows.
- The true-RUL 31-60 column conditions on the outcome, so it is not expected to reach 0.90 even without shift. It is reported only as a diagnostic, next to its in-distribution baseline.
- Row-random calibration costs 1-2.3 coverage points on these splits (an earlier 5-seed probe gave 0.879 against 0.910; the official last-cycle protocol shows no gap). Pooling rows gives only approximate validity: with engines exchangeable, split conformal is approximately valid, but it has no exact finite-sample guarantee under within-engine dependence.
- Engine-level inference is necessary. Row-level Clopper-Pearson intervals were about ±1 point (for example [0.872, 0.892]); an engine-cluster bootstrap on the same predictions gave about ±5 points ([0.825, 0.930]). Across 20 engine re-splits of FD001, a separate check found coverage SD 0.037, and only 40% of splits fell inside 0.88-0.92.
- `ueq.utils.metrics.coverage` and `interval_width` matched the independent audit exactly. MAPIE CQR logged "The predictions are ill-sorted" (quantile crossing) on every seed; adapter tests should expect it.

**Shift: marginal coverage holds.** The draft plan assumed the FD001 to FD003 shift would break coverage and that rolling windows or ACI would repair it. That did not happen.
- FD001 to FD003 fleet replay (20 held-out FD001 engines start at t=0-150, then all 100 FD003 train engines start at t=200-1200; labels resolve only at each engine's failure; about 28,500-29,000 predictions per seed): static split conformal covered 0.882-0.913 on FD001 before the shift and 0.892-0.906 on FD003 after it. Rolling (w=5000) gave 0.896-0.901, expanding 0.900-0.910, and an ACI stand-in with gamma=0.0005 about 0.90 at a narrower width of about 49 cycles. A separate 20-split check with a different setup gave FD001 0.894 (SD 0.037) against FD003 0.860 (SD 0.046): the drop is within split-to-split noise.
- FD002 to FD004 (per-regime z-scored sensors, 156 train / 52 calibration / 52 held-out FD002 engines, 20 re-splits): held-out FD002 covered 0.899 (SD 0.031) and FD004 0.871 (SD 0.019). Coverage by operating regime was flat (0.85-0.89 in every regime in the three prototype seeds), so a per-regime breakdown shows nothing.
- UEQ's `OnlineConformalUQ` with its default window of 500 held 0.887-0.900 on FD003 but inflated FD001 widths from 67 to 78-88 cycles: feedback arrives in bursts of about 250 scores per failed engine, so the window holds about two engines. An ACI stand-in with gamma=0.002 drove alpha_t as low as -0.73, which produces infinite intervals; single-stream ACI bounds assume immediate feedback.

**Where it fails: a fault-mode segment.** C-MAPSS does not label fault modes per engine. The prototype inferred them after the fact from the sign of each engine's end-of-life trend in sensor 12 (regime-normalised, last 20 against first 20 cycles). That split FD003 into 56 HPC-like and 44 fan-like engines and FD004 into 148 and 101. The label exists only after failure, so it is an audit segment recorded with the outcome, not a prediction-time group, and the inference rule is an assumption to document.
- FD004, 20 re-splits: inferred fan-like engines covered 0.830 (SD 0.027) against 0.909 (SD 0.018) for HPC-like engines. In all 20 splits the fan-like engine-cluster 95% upper bound was below the HPC-like coverage; the gap ranged from 2.6 to 13.1 points (mean 7.8). The marginal FD004 figure (0.871) hides this.
- Drilling down (outcome-conditional diagnostic): in the true-RUL 31-60 band, fan-like FD004 engines covered 0.38 on average (SD 0.12, range 0.22-0.59; engine-cluster CIs such as [0.19, 0.32] in single splits), against 0.76 (SD 0.06) for held-out FD002 engines in the same band. In the three prototype seeds, 57-75% of those rows were below the lower bound.
- FD003 shows the same pattern more weakly: fan-like engines covered 0.54-0.60 in the 31-60 band (40-46% below the lower bound) against 0.74-0.82 for HPC-like FD003 engines and 0.62-0.83 for held-out FD001 engines.
- Prediction-time segments do not isolate it. By regime the coverage is flat. In the predicted-RUL 61-100 band, all FD004 engines covered 0.61 / 0.74 / 0.61 against 0.64 / 0.83 / 0.67 for held-out FD002 engines (3 seeds), a drop inside the noise. A live monitor sees the failure only if the failure cause is recorded when the label resolves.
- No repair method fixed it. Rolling windows and ACI left the fan-like 31-60 band at 0.55-0.61 on FD003; one ACI run reached 0.75 only by going to infinite intervals. crepes Mondrian by predicted stage did not fix the band either. MAPIE CQR and crepes normalized CPS helped partly on the static FD001 audit.

**Delayed labels bias a naive monitor.** In the replay, a prediction made at cycle t resolves at failure, so its delay equals its true RUL. The first FD003 label arrived 179-219 cycles after the first FD003 prediction, and short-lived engines report first. At t=300, coverage computed on the labels resolved so far was 0.846 against an eventual truth of 0.885 (seed 0) and 0.884 against 0.919 (seed 2). Resolving capped labels early (RUL=125 is known once an engine survives 125 more cycles) adds mostly easy plateau rows and moves the estimate upward.

**Breach tests must be engine-level.** Over 200-cycle windows, a breach test using row-level Clopper-Pearson flagged 1, 4 and 3 windows for static split conformal in the three seeds. An engine-cluster bootstrap flagged none.

### Flagship demo: "Where do these RUL intervals fail?"

Punchline: marginal coverage says about 0.90. Engines with a fault mode that was absent from training get about 0.83 coverage on FD004 against 0.91 for the rest, and about 0.38 in the 31-60-cycle band where maintenance is planned, against 0.76 for comparable in-distribution engines, with most misses on the unsafe side. Only a segment-level, engine-clustered, delay-aware audit sees it. Rolling windows and ACI do not fix it, because they target marginal coverage.

1. **Load** (Phase 1). `ueq.benchmarks.load_cmapss("FD002", split="train")` returns per-cycle RUL capped at 125, engine ids, operating-regime columns and lifetimes. It needs numpy and pandas only, verifies size and SHA-256, and prints the citation.
2. **Old default vs new default** (Phase 1). Engine-grouped splits with fixed seeds. The 1.0.1 default covers about 0.2-0.36 at a nominal 90% (setup-dependent; the CI contrast uses n_models=30 for speed). Split conformal covers 0.88-0.91 at about 67 cycles wide. `UQ(regressor, alpha=0.1)` raises `TypeError`, because bootstrap takes alpha only at predict time; that is why one API contract matters.
3. **Leakage pitfall** (Phase 1). Row-random against engine-grouped calibration on the same splits, reported with engine-cluster CIs. The text says split conformal is approximately valid when engines are exchangeable, and that row-pooled split conformal has no exact finite-sample guarantee under within-engine dependence.
4. **Four producers, one audit** (Phase 1). UEQ split conformal, MAPIE CQR, crepes normalized CPS, and bootstrap repaired by `conformalize()`, passed to UEQ as plain arrays. `coverage_report(..., cluster=engine_id, groups=...)` reports coverage with engine-level CIs, lower and upper miss rates, width and Winkler score, by prediction-time groups (predicted-RUL band, operating regime, cycles since start). True-RUL bands appear only in a separate diagnostic table labelled "conditions on the outcome; not expected to be 1-alpha", next to the in-distribution baseline. Coverage excluding the cap plateau is reported alongside the marginal figure. The write-up says plainly where MAPIE and crepes are sharper.
5. **Repair** (Phase 1). `conformalize()` over the bootstrap intervals. The prototype's stand-in lifted coverage from 0.22-0.26 to 0.886-0.922, with widths rising to 61-65 cycles, close to split conformal. The shipped function is measured again against the same splits.
6. **Fleet replay with delayed labels** (Phase 2). Train on FD002 train engines and deploy on all FD004 train engines with staggered starts (FD004 train files run to failure; test files are never used here). FD001 to FD003 runs as a smaller secondary replay. Every prediction goes into the outcome ledger with its prediction time; each engine's labels, and its inferred failure cause, resolve at failure. The demo shows three things: naive resolved-so-far coverage against cohort-complete coverage; static, rolling-window and ACI intervals all holding marginal coverage (the null result for repair is published); and the failure-cause segment breaching under an engine-level test, which no prediction-time segment catches. The Phase 1 pilot confirms this design before Phase 2 starts.
7. **Evidence export** (Phase 2). A versioned JSON report records the declared target, realized coverage by segment and cohort with engine-level CIs, the fraction of labels still pending, breaches with the test used, and recalibration events.
8. **Limitations section.** Simulated data; no censoring in the train files; fault modes inferred after the fact with a stated rule; RUL cap inflates marginal coverage; marginal guarantee only, and only approximate under within-engine dependence; no cost claims; UEQ's own intervals are not the sharpest.

**What success looks like:**
- The audit (Phase 1) and the replay (Phase 2) run from `pip install ueq[mapie,crepes]` on CPU with no credentials in under 3 minutes; the UEQ-only parts run from a clean `pip install ueq`.
- Nightly: mean coverage of grouped split conformal over at least 20 engine-grouped re-splits lies within an engine-level tolerance of 0.90 (about 0.87-0.93, set from the measured re-split SD). Per PR: one pinned seed is checked against a stored snapshot, so CI does not flake.
- `conformalize()` repairs bootstrap to the same tolerance over the same re-splits, with its width reported.
- UEQ's coverage and width for MAPIE and crepes intervals match those libraries' own metrics to floating-point tolerance on identical arrays.
- The fault-mode finding is reproduced over at least 20 re-splits with engine-cluster CIs, and the prediction-time segments that fail to catch it are reported too.
- Negative findings are published, including the null result for rolling windows and ACI on marginal coverage.

## 4. Secondary use cases (later, each gated)

| Use case | Dataset (licence) | When | Condition |
|---|---|---|---|
| Classification validity benchmark: global LAC sets on imbalanced credit data, reporting per-class coverage, set size and singleton rate, with MAPIE and crepes class-conditional sets audited alongside. It states that marginal alpha does not bound the error of automatic decisions (that needs Learn-then-Test, already in MAPIE) and that declined applicants cause selection bias | UCI Default of Credit Card Clients (CC BY 4.0, corroborated) | Phase 3 | Download, size and licence verified from the primary source (for example via `ucimlrepo`) with a pinned checksum; tests stay network-marked and optional |
| Audit and ACI repair of a grid operator's published P10/P90 PV and wind forecasts, per lead time and month, without access to the model | Elia Open Data ods031/ods032/ods087 (CC BY 4.0 per snippets, unverified) | Phase 3 | Licence, fields and credential-free export confirmed from the primary source. If not, drop it; do not substitute HEFTCom2024 or PERFORM |
| Per-intent set-coverage monitoring for an intent router (sklearn TF-IDF + logistic regression, no API key), with an out-of-scope drift chapter | BANKING77 (CC BY 4.0); CLINC150 (CC BY 3.0) | Phase 3 | Prefit and label-encoding work done. Per-class guarantees need at least 19 calibration points per class at alpha=0.05, so CLINC150's 20 validation examples per intent are too few |
| Per-country coverage of crop-yield intervals as a teaching example of where group-conditional guarantees break down | HarvestStat Africa v1.2 (MIT) | Optional, community-contributed | Intervals come from MAPIE or crepes; UEQ only evaluates |

## 5. How the proposals and reviews were reconciled

| Question | Options on the table | Decision | Why |
|---|---|---|---|
| Flagship dataset | C-MAPSS vs Elia P10/P90 | C-MAPSS now, Elia gated to Phase 3 | Only C-MAPSS is verified, credential-free, has enough units, and runs on today's correct code. The prototype confirmed it end to end |
| What the flagship claims | "Static breaks, rolling/ACI repair" vs "marginal holds, a segment fails, the audit sees it" | The second | Measured: marginal coverage held under both shifts; the fault-mode segment failed in 20 of 20 re-splits; rolling windows and ACI did not fix it |
| UEQ as producer or auditor | Show UEQ's intervals winning vs audit everyone's | Auditor | MAPIE CQR and crepes normalized CPS were sharper and more even on the same splits |
| Statistical unit for CIs and breach tests | Rows vs engines | Engines (a `cluster=` argument) | Row-level CIs were about 5x too narrow and raised false breaches |
| Monitoring segments | True-RUL stage vs prediction-time quantities | Prediction-time groups for monitoring; outcome-conditional tables only as labelled diagnostics; label-time attributes recorded by the ledger | Coverage conditional on the outcome is uneven without any shift |
| When to switch the regressor default | 1.0.2 vs 1.1.0 | Warn in 1.0.2, switch in 1.1.0 | Bootstrap is the default in released 1.0.1. The switch widens intervals several-fold and uses part of the data for calibration, so it needs a minor version and migration notes |
| When to fix classifier routing and label mapping | 1.0.2 vs 1.1.0 | 1.0.2 | The released output (label +/- q, trivial sets for -1/+1) has no valid use, so fixing it breaks nothing anyone can rely on |
| Broken `normalized`, `margin`, `class_conditional` | Fix vs raise | Raise `NotImplementedError` in 1.0.2 and do not fix | Never on PyPI, so disabling them breaks no released user. Fixing Mondrian would duplicate crepes and MAPIE's `ConditionalSplitConformalClassifier` |
| Released but broken APIs (`CrossFrameworkEnsembleUQ`, interval ECE/MCE, `UQMonitor`, `UQ.monitor`) | Remove in 1.1 vs deprecate | Deprecate in 1.1, remove in 2.0 | CHANGELOG.md claims Semantic Versioning, and ECE/MCE are exported at the top level since 0.1.0 |
| Tests for the no-op recalibrators | Characterization tests vs strict xfail | Strict-xfail tests that encode the correct behaviour | Passing tests that pin a no-op would lock the bug in |
| Adapters | Seven library adapters vs three | Arrays, quantile columns and Gaussian in Phase 1; MAPIE and crepes in Phase 2; darts, Nixtla and sktime in Phase 3 after the gate | Phase 1 audit passes MAPIE and crepes outputs as arrays; the monitor's exit criterion needs the adapters |
| When the evaluation API lands | Phase 1 vs Phase 2 | Phase 1 | Lowest switching cost for new users, cheap to build, needed by the C-MAPSS audit, and what the month-6 gate measures |
| Bootstrap and torch methods | Remove in 2.0 vs keep | Keep as explicit opt-in "epistemic" producers under `ueq[torch]` (bootstrap stays in core) | They are useful inputs to `conformalize()`. Removal at 2.0 is decided on usage |
| Adoption targets | 150 stars and 2k downloads/month vs one external user | At least 3 external uses of the Phase 1 evaluation API or `conformalize()` by month 6, with planned outreach; stars and downloads tracked, not targeted | Evidence of real use matters more than popularity; the gate measures something released three months earlier |

## 6. Phased plan

Horizons assume 5-8 maintainer hours a week. When time runs short, the later deliverables in a phase slip, and each phase names what slips first. Validity work never slips.

**Versioning policy** (stated in the CHANGELOG in Phase 0): released APIs are deprecated with a warning for at least one minor release and removed only in a major release. Options that were never on PyPI can be disabled or removed at any time. Bug fixes that change a numerically wrong output (for example classifier routing) are not treated as breaking.

### Phase 0: an accurate 1.0.2 (weeks 0-4, October 2026, about 20-25 hours)

Goal: PyPI, the git tag, the code and the docs agree. Nobody who installs UEQ gets a silently invalid interval or set. No new features. Do not tag current `main` as it is.

Deliverables:
- **CI.** GitHub Actions running pytest on Python 3.10-3.14 (CPU torch only where wheels exist), required for merging to `main`. Add ruff with E722 (bare `except`) and T201 (`print`) so the "what we stop doing" list is enforced. Smoke-run every file in `examples/`; fix or remove the ones that fail, and declare or drop `yfinance`.
- **Release automation.** Publish to PyPI from a CI tag with trusted publishing, so tag, version and PyPI cannot drift again.
- **Classifier routing and labels.** Detect classifiers with `sklearn.base.is_classifier` and construct `ConformalUQ(task_type="classification", nonconformity="inverse_probability")`. When the model has no `predict_proba` (default `SVC()`, `RidgeClassifier`), raise a `ValueError` that suggests `SVC(probability=True)` or `CalibratedClassifierCV`. Map labels through `model.classes_` (`np.searchsorted`), or raise when y is not a subset of `classes_`. Return sets as class labels, not column indices. Tests for `LogisticRegression`, `SVC()`, `RidgeClassifier`, {-1, +1} and string labels.
- **Warnings on invalid defaults.** A `FutureWarning` whenever `method="auto"` picks bootstrap: it says these are confidence intervals of the mean, quotes the measured coverage, and says the default changes in 1.1. A `UserWarning` on explicit bootstrap, deep-ensemble and Bayesian-linear intervals. `UQ.predict` warns for MC dropout that it returns (mean, std), not intervals. `ConformalUQ` raises on `y.ndim > 1`.
- **Disable and quarantine.** The never-released `normalized`, `margin` and `class_conditional=True` raise `NotImplementedError` pointing to crepes and MAPIE. `ExperimentalWarning` on `AdaptiveConformalUQ`, `DriftAwareRecalibrator`, `UncertaintyInflator`, `UQMonitor`, `UQ.monitor` and `CrossFrameworkEnsembleUQ`.
- **Finite-sample edge cases, every branch.** Residual regression: +/-inf when `ceil((1-alpha)(n+1)) > n` (`conformal.py:70`). Asymmetric score: each side independently, with the lower side at -inf when `floor((alpha/2)(n+1)) = 0` (`conformal.py:67-68`). Classification, global and per-class paths (`conformal.py:88, 96, 101`): return all classes. Online (`online_conformal.py:113, 120`): +/-inf, not zero width, until the buffer holds `ceil(1/alpha) - 1` scores. Warn in every case, with one test per branch.
- **Tests.** Online conformal rolling-window recovery after a noise shift, as a statistical test that passes today. `benchmarks.synthetic`: shape and metadata tests plus RNG isolation; switch the generators to `np.random.default_rng` and make `shift="covariate"` shift X before y is computed. Strict-xfail spec tests saying recalibration must change q.
- **Imports.** Delete the unused torch imports (`core.py:7-8`, `cross_ensemble.py:2`).
- **Review gate.** A statistical review rule in CONTRIBUTING, plus CODEOWNERS and branch protection. Any PR that touches `ueq/methods` or the metrics needs a repeated-trial coverage test and maintainer sign-off. Bots may draft PRs but may not close issues end to end.
- **CHANGELOG `[1.0.2]`.** List every feature merged since 1.0.1, each marked stable, experimental or disabled: PR #28 (#9 metrics, including the broken interval ECE/MCE; #10 `check_calibration`; #17 plots), PR #30 (#5, #6, #8, #11, #12, #15, #18, #22, #27), and PRs #31 and #32. Add a Known Issues section with the measured coverage figures. Correct the "40 tests" claim. Drop the "Production-Ready" and "100% coverage" wording. State the versioning policy above.
- **README and package metadata.** Fix the version header and the licence line (Apache-2.0). Replace the README's "Project Status & Roadmap" section with a link to ROADMAP.md. Add a short banner about the default-interval issue and a "what is validated" table. Delete the broken monitoring and xgboost examples. Rewrite the `setup.py` description, and drop the "production", "phoenix", "auto-detection" and "cross-framework" keywords.
- **Docs.** Remove "production-ready" wording from `docs/API.md` and `docs/EXAMPLES.md`. Put an "experimental, see Known Issues" banner on `docs/PRODUCTION_GUIDE.md` or move it to `docs/releases/`. Move `IMPLEMENTATION_SUMMARY.md`, `ISSUE_RESOLUTION_SUMMARY.md`, `PHOENIX_RELEASE.md`, the release notes and the release checklist into `docs/releases/`.
- **Community.** Before closing #17, reply to @refexa: thank them, explain that `plot_intervals` shipped, and invite them to take the Phase 1 "Plot and diagnostics cleanup" issue, labelled good-first-issue.
- Tag v1.0.2 through the release workflow, close #17, and post the decisions in section 7 on every open issue.

Slips first: the docs moves and the examples clean-up beyond what CI requires.

Exit criteria:
- CI green and required, including the examples smoke run and ruff.
- PyPI, the tag, `setup.py`, `__version__` and the README all say 1.0.2, published by the release workflow.
- `UQ(LogisticRegression())` returns label sets with >= 0.88 marginal coverage at a 0.90 target; {-1, +1} and string labels give valid sets; `UQ(SVC())` raises a clear error instead of routing to bootstrap.
- Every known-invalid path warns or raises, with a test for each.
- The CHANGELOG lists every post-1.0.1 feature with its true status.
- All 10 open issues have a posted decision.

### Phase 1: v1.1.0, valid by default and torch-free, with an evaluation layer (months 1-3, November 2026 to January 2027, about 60-80 hours)

Goal: the default path and every stable estimator pass repeated-trial validity tests. scikit-learn users install without torch. Any library's intervals can be evaluated in a few lines with engine- or unit-level uncertainty. The C-MAPSS static audit proves the evaluation layer on real data.

Deliverables, in priority order:
1. **Statistical validity harness.** A `statistical` pytest marker with repeated-trial coverage tests (at least 50 seeds per PR, 200 or more nightly). Tolerances come from the Beta/binomial distribution of split-conformal coverage, not hand-picked ranges.
2. **Split conformal becomes the default for regressors,** with an internal calibration split (`calib_size`). Bootstrap, ensembles and Bayesian-linear run only on request and are documented as epistemic only. Write migration notes. `method="bootstrap"` keeps its warning.
3. **One result contract, the reframed #21.**
   - frozen `IntervalResult` and `SetResult` classes: point prediction, lower/upper or a boolean set matrix with `classes_`, alpha, and optional id, timestamp, segment, cluster and horizon;
   - `alpha` always means miscoverage and is set at construction;
   - `BayesianLinearUQ(alpha=...)` becomes `prior_precision` with a deprecation period;
   - constructors `from_arrays`, `from_quantiles`, `from_gaussian` (MAPIE and crepes adapters come in Phase 2);
   - estimators follow scikit-learn conventions (`BaseEstimator`, `get_params`, `clone`, pickling), with tests.
4. **Prefit and array API that never refits.** `ConformalUQ(model, prefit=True).calibrate(X_cal, y_cal)`. `ueq.conformalize(lower_cal, upper_cal, y_cal)` uses the CQR-style score `max(lower - y, y - upper)` to make bootstrap, ensemble, MC-dropout, Laplace or vendor intervals marginally valid. It has its own validity test.
5. **The evaluation API:** `coverage_report`, `coverage_by_group` and `coverage_over_time`. Each reports coverage, split lower and upper miss rates, width, Winkler and pinball scores, and set metrics (size, singleton rate, per-class coverage). A `cluster=` unit id gives CIs from a cluster bootstrap or from per-unit coverage averaged over units; without it the function warns that its Clopper-Pearson interval assumes independent rows. Outcome-conditional breakdowns are a separate, labelled diagnostic. Output is a tidy DataFrame. Feature-conditional metrics are left to MAPIE.
6. **C-MAPSS loader, static audit and shift pilot (part of #16).** `ueq.benchmarks.load_cmapss()` and `examples/cmapss_coverage_audit.py` (demo steps 1-5), reporting mean and SD over at least 20 engine-grouped re-splits. Include a two-hour pilot: measure FD001 to FD003 and FD002 to FD004 over at least 20 re-splits with engine-level CIs, fix the Phase 2 shift and the failure-cause segment rule before Phase 2 starts, and record the pilot numbers in the docs whatever they show. Pre-specified fallback: FD002 with one operating regime held out.
7. **Optional dependencies.** `install_requires` becomes numpy, scipy, scikit-learn and pandas, with extras `ueq[torch]`, `ueq[plot]`, `ueq[mapie]` and `ueq[crepes]`, lazy imports, a CI job without torch, and `python_requires>=3.10` with updated classifiers.
8. **Input validation.** `check_array`, pandas input everywhere (bootstrap's `X[idx]` fails on DataFrames today), and explicit multi-output handling.
9. **Deprecations, not removals.** `DeprecationWarning` on `CrossFrameworkEnsembleUQ`, the index-binned interval ECE/MCE (the warning says the value is not meaningful and points to `coverage_report`), `utils/performance.py`, `UQ.monitor` and `UQ.predict_large_dataset`. Move the monitoring and recalibration modules to `ueq.experimental` with warning import shims until Phase 2 replaces them. `check_calibration` stops flagging constant-width intervals (a bug fix). The duplicate `evaluate` is renamed with a deprecated alias.
10. **Plot and diagnostics cleanup** (good-first-issue, offered to @refexa). Plots return `fig, ax` and never call `plt.show()`. Keep one `plot_intervals`. Fix or delete the regression reliability diagram, which wrongly assumes width scales linearly with the nominal level. This absorbs the leftover items from #17.
11. **README** rewritten around a three-line quickstart: fit, `coverage_report` and `conformalize`.
12. **Outreach**, so the month-6 gate measures demand rather than obscurity: publish the C-MAPSS audit as a short write-up; answer relevant MAPIE, crepes and darts discussions with a link to a recipe; offer the audit to MAPIE as a gallery example.

Slips first: 10 (hand to a contributor), then 11 (trim), then 8 beyond what 3 and 5 need. Never slips: 1, 2, 4, 5.

Exit criteria:
- Default `UQ(LinearRegression())` and `UQ(RandomForestRegressor())` 95% intervals cover 93-97% on average over 200 synthetic trials (today 7.7% and 48.2%).
- Split conformal, LAC and `conformalize()` coverage lies within binomial tolerance of 1-alpha on synthetic data.
- `coverage_report` with `cluster=` gives CIs with close to nominal coverage in a simulation with clustered rows; without `cluster=` it warns.
- `pip install ueq` pulls in neither torch nor matplotlib, and `import ueq` works without them.
- Every public module has a behavioural test, and no test asserts only shapes or 0 <= ECE <= 1.
- The nightly C-MAPSS job meets the re-split tolerance in section 3; the per-PR job matches its snapshot. The pilot result and the chosen Phase 2 design are in the docs.
- v1.1.0 is on PyPI with migration notes.

### Phase 2: v1.2.0, coverage monitoring with delayed labels (months 3-6, January to April 2027, about 60-80 hours)

Goal: ship the part no incumbent packages. Monitor coverage per segment as labels arrive late, without the biases the prototype found, and keep marginal coverage on target with ACI over any upstream interval stream. Show it end to end on the C-MAPSS fleet replay.

Deliverables:
- **Adapters:** `from_mapie` and `from_crepes`, tested against pinned versions (MAPIE CQR's quantile-crossing log is expected).
- **`OutcomeLedger` and `CoverageMonitor`, the reframed #25.**
  - `log(result)` now and `resolve(ids, y, attrs=None)` later, allowing out-of-order, partial and never-arriving labels, with Parquet persistence. The ledger stores prediction time and resolution time, and `attrs` records segment attributes that arrive with the label, such as a failure cause.
  - By default the monitor reports coverage per prediction cohort, only once a cohort has fully resolved, or with the fraction still pending stated next to it. The docs explain that label-dependent delay biases "coverage so far", and that resolving capped labels early does too.
  - Coverage, width and Winkler score per segment and cohort with cluster-level CIs.
  - A stated breach test that holds up under autocorrelated, clustered misses: either a time-uniform confidence sequence (Hoeffding or empirical-Bernstein, as in MAPIE's `RiskMonitoring`) over one observation per resolved unit, or non-overlapping fixed windows with cluster-bootstrap CIs and a Holm correction across segments and horizons. The false-alarm rate is controlled family-wise across segments.
  - When mapie is installed, an optional `RiskMonitoring` backend that treats miscoverage as a binary risk.
- **Post-hoc ACI** (the Gibbs and Candes alpha_t update) over any score stream, including the `conformalize()` score. It is tested against the long-run bound `|mean(err) - alpha| <= (max(alpha_1, 1-alpha_1) + gamma) / (gamma * T)`. alpha_t < 0 gives infinite intervals by design, and the fraction of infinite intervals is reported. An experimental delayed-feedback variant updates once per resolved unit (or scales gamma by burst size), and ships only with a validity test on a synthetic stream with known, bursty lags.
- **Deprecate** `UQMonitor`, `PerformanceMonitor`, `AdaptiveConformalUQ`, `DriftAwareRecalibrator` and `UncertaintyInflator`, with warnings that name their replacements.
- **Evidence report v0:** a versioned JSON schema with declared alpha, realized coverage with cluster-level CIs by segment, cohort and time, pending-label fraction, breaches with the test used, recalibration events, data window and package version. The wording is "supports lifecycle documentation, for example of the kind EU AI Act Art. 15 describes". It makes no compliance claim.
- **`examples/cmapss_fleet_replay.py`** and a tutorial notebook, "Conformal RUL on C-MAPSS: what holds and what doesn't" (demo steps 6-8), using the design fixed by the Phase 1 pilot.
- **A recipe page,** "MAPIE CQR intervals into UEQ ACI and CoverageMonitor", offered as an answer on MAPIE #505.

Cut from the earlier draft: a label-free KS test on interval widths. UEQ's default producer has constant width, so the test has nothing to detect on the main use case. Revisit it only for set sizes in the optional BANKING77 example.

Slips first: the Parquet persistence (keep in-memory plus JSON), then the `RiskMonitoring` backend.

Exit criteria:
- On a synthetic abrupt-shift stream where static split conformal falls to 0.416, ACI keeps long-run coverage within 2 points of target.
- In null simulations that include autocorrelated and clustered misses, the monitor's family-wise false-alarm rate across segments is at most the test level.
- Ledger tests cover delayed, out-of-order, partial and missing labels. On a synthetic stream where delay depends on y, cohort-complete coverage is unbiased and naive resolved-only coverage is not.
- The monitor accepts UEQ, MAPIE and crepes intervals unchanged, with adapter tests in CI.
- The C-MAPSS replay runs in CI in under 3 minutes, and its results, negative ones included, are in the docs.

### Gate at month 6 (April 2027)

Continue to Phase 3 only if Phase 2's exit criteria are met, and at least 3 external issues, discussions, PRs or write-ups use the evaluation API or `conformalize()` (released in v1.1.0 around month 3) on intervals UEQ did not produce. The Phase 1 outreach items are tracked as leading indicators. If the gate fails, keep UEQ as a small, correct library. Fix bugs, and offer the evaluation functions upstream to MAPIE or crepes instead of building more.

### Phase 3: v1.3 to v2.0, forecasts and a second vertical (months 6-12, April to September 2027, about 80-100 hours)

Goal: extend the same evaluate-monitor-repair loop to multi-horizon forecasts and to prediction sets. Shrink the API, and bring the bus factor above one. If hours run short, cut the second vertical first.

Deliverables:
- The reframed #23: adapters for darts quantile `TimeSeries`, Nixtla forecast DataFrames and sktime `predict_interval`, tested against pinned upstream versions. Per-horizon post-hoc conformal and ACI over backtest residuals. No UEQ forecaster.
- Quantile evaluation: CRPS from quantiles, per-quantile (PIT) reliability, and per-horizon and per-group tables. The scope is a maintained successor to uncertainty-toolbox for non-Gaussian outputs.
- A second real-data demo: the Elia P10/P90 audit if its licence and access are verified, otherwise BANKING77 set monitoring. The UCI Taiwan classification benchmark, once its access is verified.
- Evidence report v1: the schema is validated in CI, with a static HTML render. Evidently, NannyML or ValidMind exporters come only on user request.
- v2.0 removes the deprecated modules. The public API is smaller than at 1.0.2.
- Recruit one co-maintainer or a regular statistical reviewer.

Exit criteria:
- One UEQ call evaluates and ACI-repairs the same backtest produced by darts and by Nixtla, with no library-specific code in the user's script.
- Two real-data demos run in CI without credentials.
- At least one statistical PR is approved by someone other than the maintainer.

## 7. Open-issue triage

| Issue | Current title | Action | New title or destination | Rationale |
|---|---|---|---|---|
| #13 | Evidential Regression Support (Normal-Inverse-Gamma) | Close | - | Torch-only with no coverage guarantee. Lightning-UQ-Box already ships deep evidential regression. Evidential outputs can be made marginally valid with `conformalize()` |
| #14 | Evidential Classification (Dirichlet-Based) | Close | - | Lightning-UQ-Box ships deep evidential methods (Dirichlet classification not checked in the research). Evidential outputs carry no coverage guarantee, and UEQ's own classification sets must be fixed first |
| #16 | Add Real-World UQ Benchmarks (Regression & Time-Series) | Prioritize | "Real-world coverage-audit benchmark: NASA C-MAPSS RUL (FD001; FD002 to FD004 and FD001 to FD003 shifts)" | The beachhead. One verified dataset, one protocol. No generic `load_benchmark` registry until a second verified dataset exists. The "energy_forecast" example is dropped. UCI Taiwan credit gets its own Phase 3 issue, because it is a classification benchmark |
| #17 | Prediction Interval & Uncertainty Visualizations | Reply to @refexa, then close with the 1.0.2 release | Leftovers move to the Phase 1 "Plot and diagnostics cleanup" issue, offered to @refexa as good-first-issue | `plot_intervals` shipped, and the release checklist already says "Closes #17". Still open: plots return nothing and call `plt.show()`, there are two `plot_intervals`, and the reliability diagram is wrong. @refexa is the only visible external contributor |
| #19 | Bayesian Neural Networks via Variational Inference | Close | - | Lightning-UQ-Box has BNN/VI, SWAG, Laplace and SNGP; TorchUncertainty is actively released (method list unverified). It would keep torch in the core |
| #20 | Laplace Approximation for Pretrained Models | Close | - | laplace-torch is the standard implementation. The "no retraining" goal is met by the prefit path and `conformalize()` |
| #21 | Design Plugin Architecture for UQ Methods | Reframe | "Define one result contract (IntervalResult / SetResult) and adapters for arrays, MAPIE and crepes" | The real problem is the missing contract: different fit and predict signatures, return types and meanings of alpha. That is what breaks `UQ.monitor`, the cross-ensemble and the metrics. Drop entry-point plugin discovery; no new method families are planned |
| #23 | Time-Series Conformal Prediction | Defer to Phase 3 | "Post-hoc per-horizon conformal and ACI over any forecaster's intervals (darts / Nixtla / sktime adapters)" | darts, Nixtla, sktime and MAPIE already produce forecast conformal intervals. The gap is evaluating and repairing their outputs, which depends on the Phase 1 contract and Phase 2 ACI |
| #24 | Structured Output Uncertainty (Sequences & Detection) | Close | - | UQLM, LM-Polygraph, TorchCP, puncc and MAPIE cover it. The one adjacent use case left (set monitoring for routers) is the Phase 3 BANKING77 example |
| #25 | Online / Continual Uncertainty Quantification | Reframe (Phase 2 centrepiece) | "Delayed-label coverage monitoring and ACI over any upstream interval" | This is the white space. Drop the continual-learning and forgetting scope. It replaces today's no-op pieces and builds on the rolling window, which works |
| #5, #6, #8, #9, #10, #11, #12, #15, #18 | Closed via PRs #28 and #30-#32 | Leave closed, add a comment | Link each to its new fix issue | Their implementations are partly wrong (see section 1): for example #15's generators reseed the global RNG and mislabel a concept shift as covariate shift, and #18's `plot_uncertainty_timeline` hardcodes targets and assumes half-widths. The CHANGELOG records each one's true status. Reopening would duplicate the new issues |

## 8. Proposed new issues

| Phase | Title | Summary |
|---|---|---|
| 0 | Add GitHub Actions CI, required for merge | pytest on 3.10-3.14, ruff E722/T201, examples smoke run; a no-torch job and a nightly statistical job come in Phase 1 |
| 0 | Release from CI with trusted publishing | Tag-triggered PyPI publish so tag, version and PyPI agree |
| 0 | Fix classifier auto-routing and label mapping | `is_classifier`; LAC sets; clear error without `predict_proba`; labels through `classes_`; sets returned as labels; tests for SVC, RidgeClassifier, {-1,+1} and strings |
| 0 | Warn on auto-selected bootstrap and other epistemic-only intervals | `FutureWarning` with measured coverage; MC-dropout and multi-output guards |
| 0 | Disable never-released broken options; mark no-op modules experimental | `normalized`, `margin` and `class_conditional` raise; the adaptive, recalibration, monitor, `UQ.monitor` and cross-ensemble paths warn |
| 0 | Return infinite bounds or full sets when the calibration set is too small | Every branch: residual, asymmetric per side, classification, online buffer; one test each |
| 0 | Tests for online conformal and benchmarks; strict-xfail specs for recalibration | Also fix the generators' global RNG seeding and the mislabelled covariate shift |
| 0 | Statistical review gate | CONTRIBUTING rule, CODEOWNERS and branch protection for `ueq/methods` and metrics, including bot PRs |
| 0 | Accurate 1.0.2 release: status per feature, known issues, tag and PyPI | CHANGELOG with PR #28 and #30-#32 items and a versioning policy; licence line; README roadmap replaced; `setup.py` description and keywords; docs banners; archived summary files; reply to @refexa; close #17 |
| 1 | Statistical validity test harness | Repeated-trial coverage with Beta/binomial tolerances for every stable estimator and for `conformalize()` |
| 1 | Make split conformal the default for regressors | `calib_size` split, migration notes, opt-out keeps its warning |
| 1 | Result contract and single alpha convention (reframed #21) | `IntervalResult` / `SetResult`, `from_arrays` / `from_quantiles` / `from_gaussian`, `prior_precision`, scikit-learn estimator conventions |
| 1 | Prefit `calibrate()` and array-level `conformalize()` | No refit; CQR-style repair of any upstream interval, with a validity test |
| 1 | Evaluation API with cluster-aware CIs | `coverage_report`, `coverage_by_group`, `coverage_over_time`; `cluster=`; tail miss rates, width, Winkler, pinball, set metrics; labelled outcome-conditional diagnostics |
| 1 | C-MAPSS loader, static audit and shift pilot (part of #16) | Checksum, cache, no redistribution; engine-grouped re-splits; UEQ vs MAPIE vs crepes vs repaired bootstrap as arrays; pilot fixes the Phase 2 design |
| 1 | Optional extras and lazy imports; require Python 3.10 | `ueq[torch]`, `ueq[plot]`, `ueq[mapie]`, `ueq[crepes]`; a CI job without torch |
| 1 | Input validation | `check_array`, pandas input, multi-output handling |
| 1 | Deprecate broken released APIs; add `ueq.experimental` | Cross-ensemble, interval ECE/MCE, `performance.py`, `UQ.monitor`, `UQ.predict_large_dataset`; `check_calibration` fix; rename duplicate `evaluate` |
| 1 | Plot and diagnostics cleanup (absorbs #17 leftovers; good-first-issue) | Return `fig, ax`; one `plot_intervals`; fix or delete the reliability diagram |
| 1 | Outreach: C-MAPSS audit write-up and recipe links | Leading indicators for the month-6 gate |
| 2 | MAPIE and crepes adapters | `from_mapie`, `from_crepes`, pinned-version tests |
| 2 | `OutcomeLedger` and `CoverageMonitor` (reframed #25) | Prediction and resolution time, label-time attributes, cohort-complete coverage, pending fraction, label-dependent-delay test |
| 2 | Breach test for clustered, autocorrelated misses | Confidence sequence over units, or fixed windows with cluster CIs and Holm correction; family-wise null simulations |
| 2 | Post-hoc ACI with a long-run bound test; experimental delayed-feedback ACI | Per-unit updates under bursty feedback; replaces `AdaptiveConformalUQ`, `DriftAwareRecalibrator` and `UncertaintyInflator` |
| 2 | Coverage evidence report v0 (versioned JSON) | Declared vs realized coverage by segment and cohort, pending labels, breaches, events; no compliance claims |
| 2 | C-MAPSS fleet replay and tutorial notebook | Train files only; design fixed by the pilot (FD002 to FD004 primary); publish negative results |
| 2 | Recipe: MAPIE CQR into UEQ ACI and CoverageMonitor | A worked answer to MAPIE #505 |
| 3 | Verify the Elia licence and fields, then the P10/P90 audit example | Drop the example if verification fails |
| 3 | Verify UCI Taiwan access, then a classification validity benchmark | Pinned checksum, network-marked tests; LAC plus audited MAPIE and crepes class-conditional sets |
| 3 | Forecast adapters and per-horizon post-hoc conformal/ACI (reframed #23) | darts, Nixtla, sktime; pinned-version CI |
| 3 | Quantile evaluation: CRPS, PIT reliability, per-horizon tables | Scope of an uncertainty-toolbox successor |
| 3 | BANKING77 set-coverage monitoring example | Only if the Elia demo is not possible or there is user demand |
| 3 | v2.0 API cleanup | Remove deprecated modules; the public API ends up smaller than at 1.0.2 |

## 9. What we stop doing

- Adding UQ method families: evidential, BNN/VI, Laplace, LLM and structured outputs, more nonconformity scores, Mondrian, or CQR/APS/RAPS/risk control from scratch. We ingest other libraries' outputs instead.
- Auto-selecting epistemic-only intervals (bootstrap, deep ensembles, fixed-noise Bayesian) as prediction intervals.
- Reporting row-level confidence intervals on clustered data, or segmenting a monitor by the outcome it is supposed to predict.
- Merging bot-authored statistical code without a repeated-trial validity test and human review. No more "fix all open issues one by one" PRs.
- Writing tests with hand-picked tolerances (0.7-1.0 coverage, 0 <= ECE <= 1), or tests that check only shapes.
- Shipping features that log instead of act, such as a recalibration that leaves q unchanged or a drift score that is always 0.
- "Production-ready", "Phoenix" and "auto-selects the optimal method" wording, README examples that don't run, and per-PR summary files in the repo root.
- Presenting domain products (clinical defer bands, maintenance-cost savings, trading revenue, guaranteed credit auto-decisions). The reviewers refuted or could not support each value claim.
- Competing on breadth with forecasting frameworks or observability platforms.
- Adding loaders for datasets that need a login or have non-commercial terms, or making CI depend on them.
- `print()` and bare `except:` in library code (enforced by ruff from Phase 0).

## 10. Risks

| Risk | Mitigation |
|---|---|
| Differentiation is thin. MAPIE released five times in 2026 and could add a delayed-label monitor first; `RiskMonitoring` plus a groupby already covers part of it | Interoperate (accept MAPIE outputs, offer `RiskMonitoring` as a backend). Compete on the ledger, cohort-complete coverage, clustered inference, multi-producer input and the evidence report. Apply the month-6 gate |
| Users inline 50 lines instead of adding a dependency (OpenSTEF PR #1060) | Keep the core dependency-light and small enough to vendor. Invest in the parts that are more than 50 lines: clustered CIs, slicing, delayed-label cohorts, breach tests, adapters |
| UEQ's own intervals lose the comparison (MAPIE CQR and crepes were sharper on C-MAPSS) | Position UEQ as the auditor; say so in the demo; recommend MAPIE or crepes producers where they win |
| The fault-mode segment is inferred after the fact and could look cherry-picked | Fix the inference rule in the Phase 1 pilot before the replay; report every segment examined (regime, predicted-RUL band, fleet, failure cause), each against an in-distribution baseline; publish the rule |
| Label delay equals the label on C-MAPSS, so "coverage so far" is biased | Cohort-complete coverage by default, pending fraction always shown, a synthetic test with y-dependent delay |
| Row-level statistics give false breaches on clustered data | `cluster=` in every CI; breach tests over units; null simulations with clustered, autocorrelated misses |
| The 1.1 default switch surprises users (C-MAPSS width goes from about 11 to about 67 cycles) | `FutureWarning` in 1.0.2, migration notes, documented opt-out |
| The C-MAPSS licence is unverified and the S3 URL could move; the data is simulated with no censoring | Runtime download with checksum, never redistribute, skip tests on network failure, and state the limits in the docs |
| CI flakes on coverage bands (only 40% of FD001 re-splits fell in 0.88-0.92) | Nightly re-split tolerance; per-PR snapshot on a pinned seed |
| Runtime: the 100-model bootstrap took 34 s per seed; `import ueq` takes 2 s with torch | n_models=30 or one seed for the bootstrap contrast in CI; lazy imports in Phase 1 |
| Delayed-feedback ACI is unstable under bursty labels (alpha_t reached -0.73 in the prototype) and comes from 2026 research whose content was not verified | Ship it as experimental, with per-unit updates and a bursty known-lag validity test |
| ACI guarantees long-run average coverage, not conditional coverage; autocorrelation breaks exchangeability | Say so in the docs; the flagship shows exactly this; grouped splits in every example; report width cost next to coverage |
| Maintainer bandwidth; the pull to hand work back to the bot | Cuts before additions, the review gate, named slip-first items, a good-first-issue for the external contributor, a reviewer recruited by month 12 |
| Statistical tests are slow or flaky | Fixed seeds, Beta/binomial tolerances, small trial counts per PR and large ones nightly |
| The evidence report is read as a compliance claim | Wording limited to "supports documentation". Art. 15 does not require UQ specifically |
| The Elia licence never verifies | Drop the demo. Do not substitute gated or simulated data |

## 11. Success metrics

| Metric | Baseline (Sept 2026) | Target | By |
|---|---|---|---|
| PyPI, tag, `setup.py` and README versions agree | 1.0.1 / none / 1.0.2 / v1.0.1 | All 1.0.2, published from CI | Week 4 |
| README licence line matches `LICENSE` | MIT vs Apache-2.0 | Apache-2.0 everywhere | v1.0.2 |
| `UQ(LogisticRegression())` output | label +/- q intervals | Label sets with >= 0.88 coverage at 0.90 | v1.0.2 |
| Classification labels other than 0..K-1 | crash or trivial sets | {-1,+1}, strings and {1,2,3} give valid sets, tested | v1.0.2 |
| Python versions tested in CI | none | 3.10-3.14 | v1.0.2 |
| Default 95% interval coverage, LinearRegression / RandomForest (200 trials) | 7.7% / 48.2% | 93-97% | v1.1.0 |
| Stable estimators with a repeated-trial validity test in CI | 0 | 100% | v1.1.0 |
| Clean install pulls torch | Yes; import 2-3 s | No torch or matplotlib; import under 1 s | v1.1.0 |
| Public API size | 1.0.2 surface | Never-released broken options disabled in 1.0.2; broken released APIs deprecated in 1.1; smaller overall at v2.0 | v1.0.2 / v1.1.0 / v2.0 |
| C-MAPSS audit in CI | none | Nightly mean over >= 20 engine re-splits within about 0.87-0.93; per-PR snapshot; < 3 min, no credentials; bootstrap repaired from about 0.2-0.36 to the same tolerance | v1.1.0 |
| C-MAPSS fault-mode finding | measured in the prototype (FD004 inferred fan-like 0.83 vs 0.91, 20/20 splits) | Reproduced in the replay with engine-level CIs and a stated breach test | v1.2.0 |
| ACI on an abrupt-shift stream | static falls to 0.416 | within 2 points of target, long run | v1.2.0 |
| Monitor false-alarm rate, null with clustered misses | row-level test flagged 1-4 windows per seed | at most the test level, family-wise | v1.2.0 |
| Open issues with a posted decision | 0 of 10 | 10 of 10 | Week 4 |
| Statistical PRs merged without a validity test and review | most recent ones | 0 | From Phase 0 |
| External uses of the evaluation API or `conformalize()` on non-UEQ intervals | 0 | >= 3 (gate) | Month 6 |
| Non-maintainer statistical reviewer | none | at least 1 | Month 12 |
| Tracked, not targeted: GitHub stars, forks, PyPI downloads | 11 stars, 1 fork | reported each release | - |
