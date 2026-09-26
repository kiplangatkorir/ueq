# Where UEQ could be used in practice: research evidence

This document holds the evidence behind [ROADMAP.md](ROADMAP.md). It records how the research was done, what each of eight domains showed, which datasets could be verified, which tools already cover what, and where a gap remains. It was compiled on 2026-09-25 and revised the same day after a critic review of the roadmap and a feasibility prototype on the flagship dataset.

Nothing here claims regulatory compliance. Where a regulation is mentioned, it is context for demand, not a requirement that UEQ meets.

## How the research was done

The work ran as a workflow of independent research agents, and every domain claim was checked by a second agent whose job was to refute it.

1. **Capability audit.** Every module in `ueq/` was read and probed with small experiments: coverage over repeated trials, edge cases, and input types. Each module was rated correct, usable prototype, fragile, or broken.
2. **Competitor landscape.** Other tools were compared using PyPI release metadata, READMEs and, where it mattered, the source inside downloaded wheels.
3. **Eight domain researchers**: energy, supply chain, healthcare, credit and finance, agriculture and climate, industrial, MLOps monitoring, and LLM/GenAI. Each proposed use cases, public datasets, the UEQ features a demo would need, and a flagship demo.
4. **Eight adversarial reviewers**, one per domain. Each tried to refute its researcher. They re-ran UEQ locally, inspected competitor wheels (MAPIE 1.5.0, crepes 0.9.1, TorchCP 1.2.1, NannyML 0.13.1, ValidMind 2.13.14, uqlm 0.6.6 and others), and tried to download each dataset. Each domain got a score out of 10 combining demand, fit with UEQ, differentiation from existing tools, and demo feasibility. Differentiation dominated the scores.
5. **Re-verification.** The headline numbers were re-run: the default-interval coverage, the classifier routing, and the C-MAPSS results.
6. **Three strategy proposals**, each written from a different angle: trust first, niche and adoption, and solo-maintainer realism. They were merged into the roadmap. Section 5 of the roadmap records how their disagreements were resolved.
7. **Critic review of the draft roadmap.** A reviewer checked every URL and figure against the research, re-read the repository, and ran its own spot checks on C-MAPSS (20 engine-grouped re-splits). It found the statistical design of the flagship too optimistic: row-level confidence intervals on clustered rows, segments defined by the label, label delay that depends on the label, and an unspecified breach test.
8. **Feasibility prototype.** A script downloaded C-MAPSS, ran the flagship audit and a delayed-label fleet replay with UEQ 1.0.2 from `main`, MAPIE 1.5.0 and crepes 0.9.1 (3 seeds), and checked the FD002 to FD004 shift. A follow-up ran 20 engine-grouped re-splits of that shift. The results changed the flagship's thesis; see "Why C-MAPSS became the beachhead".

**Evidence labels used below.**
- *Verified*: downloaded and parsed, read from a primary file or package source, or reproduced locally.
- *Corroborated*: confirmed through a secondary source, such as a README quoting the fact.
- *(unverified)*: rests on a search snippet, memory, or a page the reviewer could not open.

**Limits.** The research environment's proxy blocked many hosts, including arxiv.org, nature.com, zenodo.org, opendata.elia.be, kaggle.com, physionet.org, huggingface.co, archive.ics.uci.edu, fda.gov, data.nasa.gov, catalog.data.gov and the EU AI Act reference sites. Some reviewers also used up their search budget. A blocked source is marked unverified, not refuted.

## Headline findings

- **UEQ's defaults were not valid.**
  - `UQ(LinearRegression())` 95% bootstrap intervals covered 7.7% of outcomes and `UQ(RandomForestRegressor())` 48.2%; an independent probe gave 10.6% and 56%.
  - On C-MAPSS the default covered 0.22-0.36 of true RUL at a nominal 90%, depending on the setup (0.217-0.256 with UEQ's default of 100 bootstrap models).
  - `UQ(LogisticRegression())` ran conformal in regression mode.
  - Mondrian, normalized and margin conformal, the adaptive recalibration, `UQMonitor` drift and the cross-framework ensemble were broken or did nothing.
- **A small core was correct.** Split conformal regression with the residual score reached 0.906 at a 0.90 target and the asymmetric score 0.902, both over 200 trials. Global LAC classification reached 0.899. The rolling-window online conformal restored coverage from 0.416 to 0.869 after a shift. The basic interval metrics (coverage, width, Winkler) were correct.
- **Every domain scored 3 or 4 out of 10, mainly on differentiation.** MAPIE 1.5 (CQR, conditional conformal, ACI/EnbPI, risk control, `RiskMonitoring`, exchangeability tests), crepes (Mondrian, normalized, predictive systems, online), darts, Nixtla and sktime (forecast conformal), TorchCP, UQLM and LM-Polygraph already ship the methods the domains asked for.
- **What survived in every domain was a benchmark or a tutorial, not a product.** The one gap across domains is after the interval producers: evaluating and monitoring the coverage of deployed intervals and sets, per segment and horizon, with delayed labels, whichever library produced them. The reviewers judged even that gap thin.

## Results by domain

| Domain | Score | Strongest surviving use case | Best dataset (status) | Role in roadmap |
|---|---|---|---|---|
| Energy | 3/10 | Post-hoc audit and recalibration of existing day-ahead wind and solar quantile forecasts, as a benchmark | Elia P10/P90 (unverified); HEFTCom2024 submissions (licence unverified) | Phase 3 second demo, only if the Elia licence is verified |
| Supply chain | 4/10 | A per-segment coverage "SLA monitor" over any model's intervals, first on delivery promises | Olist (Kaggle, CC BY-NC-SA per search, unverified) | Informs the monitor design; no loader |
| Healthcare | 3/10 | Post-hoc calibration of a locked binary clinical risk model with correct label-conditional coverage, as a correctness benchmark | PhysioNet 2019 (unverified) | Not pursued |
| Credit and finance | 4/10 | Approve/decline/refer as an honest classification benchmark, not a product | UCI Taiwan credit (CC BY 4.0, corroborated) | Phase 3 benchmark, after access is verified |
| Agriculture and climate | 4/10 | A reproducible "UQ track" for subnational crop-yield intervals | HarvestStat Africa v1.2 (MIT, verified) | Optional community example |
| Industrial | 4/10 | Conformal RUL on NASA C-MAPSS as a benchmark and tutorial | C-MAPSS (verified) | **Beachhead** |
| MLOps monitoring | 3/10 | A thin layer on MAPIE for multiclass selective routing, online re-thresholding and audit sampling | Folktables (code MIT; loader last released 2023) | Informs the monitor; no routing product |
| LLM/GenAI | 3/10 | Calibrated auto-routing for an intent classifier, as a worked example | BANKING77 (CC BY 4.0, verified); CLINC150 (CC BY 3.0, verified) | Phase 3 optional example |

### Energy (3/10)

- **Demand.** Probabilistic forecasting and reserve sizing are really used and regulated. EU SOGL Art. 157 requires probabilistic FRR sizing that covers at least 99% of imbalances (snippet-verified). CAISO uses mosaic quantile regression, and ERCOT moved to a probabilistic net-load-error model. But the users (TSOs, ISOs, market monitors, vendors) build in-house. CAISO's market monitor already reviews quantile-regression coverage. OpenSTEF replaced a MAPIE-based pull request with about 50 lines of inlined conformal code (PR #1060, merged 2026-08-18).
- **Refuted or corrected.**
  - The HEFTCom2024 "calibration pays" newsvendor claim. Settlement uses a single imbalance price, so the bid is not a quantile. The best strategic-bidding gain was about 0.49%.
  - The idea that no array-in post-hoc calibrator exists. crepes, MAPIE and darts already provide one.
  - The ERCOT 3.7 GW error was a data-pipeline problem, not a calibration failure.
  - PERFORM's renewable "actuals" and forecasts are simulated.
- **Strongest surviving.** Recalibrate and audit, per quantile and per horizon, forecasts that already exist: HEFTCom2024 team submissions and Elia's published P10/P90. This works as a benchmark only.
- **Datasets.**
  - Elia ods031/ods032/ods087: CC BY 4.0 and the P10/P90 fields are known only from snippets (unverified); ods032 covers data before 22/05/2024.
  - HEFTCom2024 on Zenodo: licence unverified; the NWP training features are gated on IEEE DataPort.
  - epftoolbox: AGPL-3.0 (verified), day-ahead prices only.
  - PERFORM: simulated; licence unverified.
  - GEFCom2014: unverified.
- **Why not the beachhead.** The data could not be verified or is gated, the headline value claim was refuted, and incumbents cover the calibration. Elia remains the purest "audit someone else's intervals" demo, so it is kept as a gated Phase 3 option.

### Supply chain (4/10)

- **Demand.** High and verified: the M5 Uncertainty track, commercial probabilistic planning, and AWS closing Amazon Forecast to new customers on 29 July 2024.
- **Refuted.**
  - Forecasting libraries do not stop at a fixed interval. darts `ConformalNaiveModel`/`ConformalQRModel` calibrate any pre-trained model per horizon and at arbitrary quantiles.
  - mlforecast 1.1.0 has a per-series `scale_estimator`.
  - crepes returns one-sided Mondrian percentiles with `y_min=0`.
  - utilsforecast and darts ship pinball and coverage metrics per series.
  - MAPIE 1.5.0 `RiskMonitoring` fed with a miss indicator gives a valid coverage alarm. A per-segment monitor is therefore "a loop of monitors over groups".
- **UEQ fit.** `normalized` gave 19.3% coverage at a 90% target, adaptive recalibration was a no-op, `UQMonitor` never sees outcomes, and there was no prefit mode.
- **Strongest surviving.** A model-agnostic per-segment coverage monitor over intervals produced elsewhere, shown on Olist delivery promises. It is a narrow add-on.
- **Datasets.**
  - M5: Kaggle rules. The Nixtla mirror has no licence, was archived on 29 Nov 2025 and returned 403.
  - Favorita: Kaggle.
  - Olist: CC BY-NC-SA 4.0 per search (unverified); needs a Kaggle token.
  - LaDe: research use per its README.
- **Why not the beachhead.** Kaggle-gated or non-commercial data, and a heavy pipeline for one maintainer. The monitor idea itself went into the roadmap's Phase 2.

### Healthcare (3/10)

- **Demand.** Local validation of vendor models is a real need; Epic open-sourced seismometer for it. No practitioner evidence was found for set-valued or "defer" outputs. The regulatory drivers are secondary summaries (FDA draft guidance), not near-term (EU Annex I from 2028), or weakening (HTI-5 proposal).
- **Refuted.**
  - The "defer to clinician" band is not workable. In a binormal simulation at alpha=0.1, it deferred about 69% of patient-hours at AUC 0.63 and about 31% at AUC 0.85.
  - Label-conditional coverage does not guarantee alert sensitivity.
  - MAPIE 1.5.0's `BinaryClassificationController` already does PPV/NPV/abstention triage with Learn-then-Test.
- **UEQ fit.** Mondrian gave 60.7% positive-class coverage at a 90% target, and every conformal class refits the model.
- **Strongest surviving.** A locked-model calibration benchmark with correct label-conditional coverage, reported honestly with deferral rates.
- **Datasets.**
  - PhysioNet 2019: open access and licence unverified (host blocked).
  - SPARCS: unverified. It has discharge-time fields that leak the target.
  - MedMNIST: verified; CC BY 4.0 except DermaMNIST, which is CC BY-NC 4.0. It has no site metadata.
- **Why not chosen.** Unverified data, refuted value claims, and strong incumbents.

### Credit and finance (4/10)

- **Demand.** EU AI Act Annex III 5(b) covers credit scoring (corroborated). The Digital Omnibus moved Annex III duties to 2 Dec 2027 (corroborated secondhand). SR 26-2 replaced SR 11-7 on 17 Apr 2026 (corroborated secondhand). None of these requires UQ. A GitHub search for conformal credit scoring found 5 repositories, each with 4 stars or fewer.
- **Refuted.**
  - "Alpha sets a guaranteed error on automated decisions." Split conformal bounds only marginal coverage; bounding the error needs Learn-then-Test, which MAPIE 1.5.0 has.
  - Declined applicants have no outcomes, and the resulting selection bias undermines any auto-decline guarantee.
  - ValidMind's open library already ships credit validation and drift tests.
- **UEQ fit.** On 5%-minority data at alpha=0.1, minority coverage was 0.07 (marginal) and 0.287 (Mondrian). `UQ(classifier)` picks regression mode.
- **Strongest surviving.** An honest classification benchmark and a correctness fixture.
- **Datasets.**
  - UCI Taiwan: CC BY 4.0, 30,000 rows, DOI 10.24432/C55S3H (corroborated secondhand; page blocked).
  - Home Credit: Kaggle; terms unverified.
  - Zindi: login required; terms unverified.
  - BAF: Kaggle; licence unverified.
  - PaySim: synthetic; licence unverified.
  - freMTPL2: fetch_openml 41214, corroborated through a scikit-learn example; CC0 unverified.
- **Role.** UCI Taiwan becomes a published classification benchmark in Phase 3, once its download and licence are verified from the primary source. Phase 1 validity tests use synthetic data only, so CI does not depend on an unverified host.

### Agriculture and climate (4/10)

- **Demand.** Mostly public, humanitarian and academic. The benchmarks are real: CY-Bench-Modeling confirms 63 country-crop datasets and more than 12,000 regions. No practitioner source was found asking for calibrated intervals.
- **Refuted.**
  - MAPIE 1.5.0 already provides year-blocked CV+ (`groups=` with `LeaveOneGroupOut`), CQR, conditional conformal and `coverage_gap`.
  - crepes provides Mondrian regression and `predict_p` at a threshold.
  - The CY-Bench protocol is walk-forward, not leave-one-year-out.
  - Per-country guarantees are weak: the effective calibration size is about the number of years, and yield shocks are correlated across space.
- **Strongest surviving.** A reproducible "UQ track" for crop-yield intervals with per-country and per-lead-time coverage reporting. The only method nobody ships is contiguous ordinal prediction sets for sklearn, with unverified demand.
- **Datasets.**
  - HarvestStat Africa v1.2: MIT, verified; CSVs directly in git.
  - CY-Bench: code licence EUPL-1.2 (verified); data sources keep their own terms.
  - HFID, CHIRPS v3, JRC ASAP, FEWS NET: unverified.
- **Why not the beachhead.** Too few exchangeable units per group, labels arrive once a year, and incumbents cover the methods. It is kept as an optional community example.

### Industrial (4/10)

- **Demand.** Moderate and research-led. ProgPy's uncertainty propagation is verified. MATLAB's confidence-bound RUL estimators are a vendor claim (unverified). The standards claims (ISO 13381-1, FAA AC 43-218) are unverified and none mandates UQ.
- **Verified.**
  - The C-MAPSS zip returns HTTP 200 and is 12,429,152 bytes.
  - FD001 is 20,631 x 26 with 100/100 engines; FD002 260/259, FD003 100/100, FD004 249/248.
  - There are 1/6/1/6 operating conditions and 1/1/2/2 fault modes.
  - Engine-grouped split conformal reached 0.910 coverage against 0.879 for row-random calibration (5 seeds); on the official last-cycle protocol there was no gap.
  - The symmetric interval is 58-66 cycles wide, and the lower bound covers 0.97-0.98 for RUL <= 30, which is over-conservative.
  - Re-run for the roadmap draft (3 seeds, HistGradientBoosting, 30 bootstrap models, alpha passed at predict time, a cycle feature): split conformal 0.897/0.921/0.897 (width 63.8), asymmetric score 0.888-0.919, and the bootstrap default 0.356-0.358 (width 9.6).
- **Feasibility prototype** (FD001, engine-grouped 60/20/20 engines, 90% target, 3 seeds; UEQ's default 100 bootstrap models):
  - UEQ split conformal 0.882/0.913/0.912 (width 67-69); the 1.0.1 default `UQ(HGB)` 0.228/0.217/0.256 (width about 11); a 10-line `conformalize()` stand-in repaired the bootstrap to 0.886-0.922 (width 61-65). `UQ(HGB, alpha=0.1)` raises `TypeError`.
  - MAPIE 1.5 CQR covered 0.931-0.939 at width 53-57 with the best Winkler score; crepes normalized CPS covered 0.906-0.935 at width 53-65. UEQ's own intervals were the widest and the least even across RUL.
  - Misses fall almost entirely below the lower bound, the unsafe side. In the true-RUL 31-60 band, UEQ split conformal covered 0.62-0.83. That band conditions on the outcome, so it is a diagnostic, not a target.
  - Row-level Clopper-Pearson intervals were about ±1 point; engine-cluster bootstrap intervals about ±5 points.
  - 26-49% of rows sit on the RUL=125 cap, where almost every interval covers, which inflates marginal coverage.
  - Shift did not break marginal coverage: FD001 to FD003 gave 0.892-0.906 in a delayed-label replay, and FD002 to FD004 gave 0.871 (SD 0.019) against 0.899 (SD 0.031) on held-out FD002 engines over 20 re-splits. Coverage by operating regime was flat.
  - The failure is conditional. With fault modes inferred after the fact from each engine's end-of-life sensor-12 trend (the data does not label them), FD004's 101 inferred fan-like engines covered 0.830 (SD 0.027) against 0.909 for the other 148, with the engine-cluster upper bound below the HPC-like figure in 20 of 20 re-splits. In the true-RUL 31-60 band they covered 0.38 on average against 0.76 for held-out FD002 engines. Rolling windows, ACI and crepes Mondrian by predicted stage did not fix it.
  - Label delay equals the label in a fleet replay. At t=300, coverage on the labels resolved so far was 0.846 against an eventual 0.885 (one seed).
  - The archive contains no licence or terms text.
- **Refuted.**
  - A working normalized score (it gave 4.2% coverage).
  - `load_cmapss` as a differentiator: rul-datasets already loads C-MAPSS, N-C-MAPSS, FEMTO and XJTU-SY.
  - Drift handling: `DriftAwareRecalibrator` reported drift 0.0 under a 3-sigma shift while coverage fell to 0.47.
  - The flotation soft-sensor runner-up: the cited repository reports a sensor-only R^2 of 0.071, and persistence beats sensors.
- **Strongest surviving.** Conformal RUL on C-MAPSS as a benchmark and tutorial, "what goes wrong when you conformalize RUL". It is not a market position.
- **Datasets.**
  - C-MAPSS: verified; licence unverified (likely US-government work; the archive contains no licence or terms text).
  - N-CMAPSS: HTTP 200, 15.76 GB.
  - NASA Milling: verified, but only 16 tools, too few groups.
  - FEMTO, IMS: HTTP 200; contents not inspected.
  - XJTU-SY, SECOM, flotation: unverified.
- **Why chosen.** See "Why C-MAPSS became the beachhead" below.

### MLOps monitoring (3/10)

- **Demand.** Real for routing to human review and for label-free monitoring. NannyML has about 2.2k stars, but its last release was 0.13.1 on 2025-07-12. MAPIE ships human-verification and LLM-judge abstention examples.
- **Refuted.**
  - MAPIE already has two-threshold abstention with joint PPV/NPV and abstention-rate control, FWER-controlled Learn-then-Test, `RiskMonitoring` with anytime-valid bounds, and online exchangeability tests.
  - ppi_py covers PPI confidence intervals and audit power analysis.
  - NannyML CBPE auto-calibrates.
  - Evidently 0.7.23 contains no conformal code (verified in the wheel).
- **Strongest surviving.** A thin layer on MAPIE: multiclass selective routing, online re-thresholding from selection-biased reviewer labels, and risk-coverage reports. For the regression "coverage SLA" case, what is left is a delayed-label ledger and a report formatter: "useful glue but easy for users to write and hard to defend".
- **Datasets.**
  - Folktables: code MIT (verified); Census terms; last PyPI release Feb 2023.
  - Shifts Weather: CC BY-NC-SA 4.0 (verified from README).
  - UCI Bike Sharing: unverified.
- **Role.** The delayed-label ledger and the stated breach test go into the roadmap's Phase 2. Routing and risk control are left to MAPIE.

### LLM/GenAI (3/10)

- **Demand.** Moderate to high. uqlm has 71 releases through 2026-09-03 and ships calibration and threshold tuning. Cleanlab TLM is a commercial product.
- **Refuted.**
  - crepes (torch-free, Mondrian, online), TorchCP and MAPIE (`prefit=True`) already do score-only conformal classification.
  - MAPIE's `BinaryClassificationController` does hallucination-risk abstention. A reviewer ran it torch-free on precomputed scores: at a precision target of 0.9, 17.2% of queries were answered, with a 4.7% hallucination rate among them.
  - ppi_py covers PPI.
- **UEQ fit.** Mondrian reached 0.759 overall (worst class 0.544) at a 0.90 target, and margin sets were trivial.
- **Strongest surviving.** A worked example of calibrated intent routing on BANKING77 or CLINC150.
- **Datasets.**
  - BANKING77: CC BY 4.0, 10,003 train and 3,080 test rows, 77 intents (verified).
  - CLINC150: CC BY 3.0, 150 intents, 1,000 out-of-scope test queries (verified).
  - MMLU: MIT (verified; counts unverified).
  - RAGTruth: MIT (verified), but it has no confidence scores.
  - TriviaQA: Apache-2.0 (verified).
  - MT-Bench human judgments and Chatbot Arena: terms unverified.
- **Role.** At most one Phase 3 example.

## Dataset register

### Verified or clearly public

| Dataset | Domain | Licence | Access | Notes |
|---|---|---|---|---|
| NASA C-MAPSS FD001-FD004 | Industrial | Unverified (likely US-government work); no licence text in the archive | Public S3 zip, HTTP 200, 12,429,152 bytes, SHA-256 pinned | Beachhead. Simulated; train files run to failure, test files end before failure; no censoring; 26-49% of rows on the RUL cap |
| NASA N-CMAPSS | Industrial | Unverified | HTTP 200, 15.76 GB | Too large for CI |
| NASA Milling | Industrial | Unverified | HTTP 200, 14,731,306 bytes; parsed | 16 tools, too few groups |
| NASA FEMTO, IMS bearings | Industrial | Unverified | HTTP 200 | Contents not inspected |
| HarvestStat Africa v1.2 | Agriculture | MIT (verified) | CSVs in git | 18 countries at Admin-1 |
| CY-Bench | Agriculture | Code EUPL-1.2 (verified); data terms vary | Zenodo DOI | Walk-forward protocol |
| BANKING77 | LLM | CC BY 4.0 (verified) | GitHub | 10,003 / 3,080 rows, 77 intents |
| CLINC150 | LLM | CC BY 3.0 (verified) | GitHub | 20 validation examples per intent |
| MMLU | LLM | MIT (verified) | GitHub data.tar | Counts unverified |
| RAGTruth | LLM | MIT (verified) | GitHub | No confidence scores |
| TriviaQA | LLM | Apache-2.0 (verified) | GitHub | Size unverified |
| MedMNIST | Healthcare | CC BY 4.0, DermaMNIST CC BY-NC 4.0 (verified) | pip | No site metadata |
| epftoolbox | Energy | AGPL-3.0 (verified) | GitHub downloader | Day-ahead prices only; do not vendor |
| UCI Default of Credit Card Clients | Finance | CC BY 4.0 (corroborated) | No login | Page blocked here; 30,000 rows |
| Folktables ACS | MLOps | Code MIT; Census terms | Programmatic | Loader unmaintained since 2023 |
| Shifts Weather | MLOps | CC BY-NC-SA 4.0 (verified) | Tarballs | Non-commercial |

### Unverified, gated or restricted

| Dataset | Domain | Problem |
|---|---|---|
| Elia ods031/ods032/ods087 | Energy | Host blocked; CC BY 4.0 and fields known from snippets only (unverified) |
| HEFTCom2024 (Zenodo 13950764) | Energy | Licence unverified; NWP features gated on IEEE DataPort |
| ARPA-E PERFORM | Energy | Simulated actuals and synthetic forecasts; licence unverified |
| GEFCom2014 | Energy | Download not checked (unverified) |
| M5, Favorita | Supply chain | Kaggle login and rules; the M5 mirror is unlicensed and archived |
| Olist | Supply chain | Kaggle; CC BY-NC-SA 4.0 per search (unverified) |
| PhysioNet 2019, MIMIC-IV-ED | Healthcare | Host blocked (unverified); MIMIC needs credentials |
| NY SPARCS | Healthcare | Host blocked (unverified); discharge-time fields leak the target |
| Home Credit, Zindi, BAF, PaySim | Finance | Login needed; terms unverified |
| HFID, CHIRPS v3, JRC ASAP, FEWS NET | Agriculture | Hosts blocked (unverified) |
| XJTU-SY, SECOM, iron-ore flotation | Industrial | Hosts blocked or Kaggle (unverified) |
| UCI Bike Sharing | MLOps | Host blocked (unverified) |
| MT-Bench judgments, Chatbot Arena | LLM | Terms unverified; possibly gated |

## Why C-MAPSS became the beachhead

It is the only verified, credential-free regression dataset in the research with enough independent units and with delayed labels, segments and a shift built in:
- the download is verified twice, small (12.4 MB) and pinned by SHA-256;
- the contents were parsed and match the documentation;
- it needs no credentials and runs on scikit-learn alone;
- UEQ's correct core (residual split conformal) already works on it end to end.

Other verified datasets fall short on one of these. NASA Milling was downloaded and parsed but has only 16 tools. HarvestStat Africa gives roughly one exchangeable unit per year per country. BANKING77 and CLINC150 are classification tasks that need prefit and label-encoding work before UEQ can use them.

What C-MAPSS offers a coverage audit:
- **Labels that arrive late.** In a fleet replay, RUL is known when the engine fails, so the delay of each label equals the label. That is realistic, and it is also a trap: coverage computed on the labels resolved so far over-represents short-lived engines. The prototype measured this bias.
- **Many rows per unit.** Rows within an engine are strongly dependent, so confidence intervals and breach tests must work at the engine level. The prototype found row-level intervals about 5x too narrow.
- **Segments known at prediction time**: operating regime (six in FD002/FD004), predicted-RUL band, cycles since start, and fleet.
- **A fault-mode shift.** FD003 and FD004 add fan degradation to the HPC degradation seen in FD001 and FD002. The data does not label which engine has which fault, so any fault-mode segment has to be inferred after failure.

What the feasibility prototype changed. The draft expected the FD001 to FD003 shift to break coverage and rolling windows or ACI to repair it. On the real data, marginal coverage held near 0.90 under both shifts, and every repair method landed at about 0.89-0.91. The failure was conditional: engines with the inferred new fault mode lost about 8 coverage points on FD004 (in 20 of 20 re-splits), and far more in the mid-RUL band where maintenance is planned, with misses below the lower bound. Prediction-time segments did not isolate it. Rolling windows and ACI did not fix it. MAPIE CQR and crepes normalized CPS produced sharper, more even intervals than UEQ. The flagship therefore became an engine-clustered, delay-aware audit of several producers' intervals, with UEQ as the auditor.

It also shows ordinary failure modes honestly: row leakage in calibration (1-2.3 coverage points), a RUL cap that inflates marginal coverage, and a symmetric interval whose lower bound is over-conservative near end of life.

The domain itself scored 4/10. The benchmark is a proving ground for UEQ's evaluate, monitor and repair layer, not a claim that UEQ wins in predictive maintenance. Elia's published P10/P90 bands would be the purest demonstration of auditing someone else's intervals, and they stay as a gated Phase 3 demo until their licence and fields are verified.

## Competitor landscape

| Tool | Focus | Already covers (relevant to UEQ) | Activity |
|---|---|---|---|
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | sklearn-compatible conformal prediction and risk control | Split, CV+, jackknife+, CQR (`prefit=True` default in 1.5), LAC/APS/RAPS/top-k, conditional conformal via `feature_map`, ACI and EnbPI, `BinaryClassificationController` (Learn-then-Test with PPV/NPV/abstention), `RiskMonitoring`, exchangeability martingales, `coverage_gap`, Venn-ABERS, an LLM-as-judge abstention example | Very active: 1.2.1 to 1.5.0 in 2026 (1.5.0 on 2026-08-05) |
| [crepes](https://github.com/henrikbostrom/crepes) | Light conformal classifiers, regressors and predictive systems | Array-in `fit(residuals, sigmas, bins)`, Mondrian, normalized (DifficultyEstimator), `predict_percentiles`, `predict_p(t)`, online variants, martingales; numpy/pandas/scipy only | Steady: 0.9.1 on 2026-06-12; single academic maintainer |
| [TorchCP](https://github.com/ml-stat-Sustech/TorchCP) | PyTorch conformal research toolbox | 25+ classification scores, ACI/AGACI/CQR, class-conditional and weighted predictors, logits-in thresholds, LLM module | 1.2.1 (2025-10-14); no 2026 release |
| [puncc](https://github.com/deel-ai/puncc) | Conformal for critical systems | Split, CQR, CV+, EnbPI, AdaptiveEnbPI, APS/RAPS, anomaly and object detection | 0.9.1-0.9.3 in 2026 |
| [Fortuna](https://github.com/awslabs/fortuna) | Deep-learning UQ | Archived 2025-04-23 | Dead; sktime's EnbPI depends on it |
| [uncertainty-toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox) | Regression UQ metrics and plots | Calibration, sharpness, scoring rules; assumes Gaussian outputs | Dormant since 0.1.1 (2023-01-18); about 2k stars |
| [Lightning-UQ-Box](https://github.com/lightning-uq-box/lightning-uq-box) | Deep-learning UQ | 30+ methods incl. evidential, MC dropout, SWAG, Laplace, BNN/VI, SNGP, ensembles, CQR | 0.3.0 (2026-08-23) |
| [laplace-torch](https://github.com/aleximmer/Laplace) | Post-hoc Laplace for PyTorch | The standard Laplace implementation | 0.2.2.2 (2024-11-27) |
| [TorchUncertainty](https://pypi.org/project/torch-uncertainty/) | PyTorch UQ | Method list not checked (unverified) | 0.13.0 (2026-07-22) |
| [Nixtla statsforecast/mlforecast](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html) | Forecasting | `ConformalIntervals` / `PredictionIntervals` with `h`, `n_windows`, per-series scale; utilsforecast metrics per series | statsforecast 2.1.1, mlforecast 1.1.0 (2026-07) |
| [darts](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.conformal_models.html) | Forecasting | `ConformalNaiveModel`/`ConformalQRModel` on pre-trained models, per horizon, rolling `cal_length`; interval metrics | 0.47.0 (2026-09-04); roughly monthly |
| [sktime](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.conformal.ConformalIntervals.html) | Time-series framework | `ConformalIntervals`, EnbPI (via archived Fortuna) | 1.2.0 (2026-09-22) |
| [GluonTS](https://github.com/awslabs/gluonts) | Probabilistic deep forecasting | No conformal wrapper found (unverified) | 0.17.0 (2026-07-31) |
| [NannyML](https://github.com/NannyML/nannyml) | Label-free performance monitoring | CBPE/DLE, drift detection; point-prediction performance, not interval coverage | 0.13.1 (2025-07-12); acquired by Soda |
| [Evidently](https://github.com/evidentlyai/evidently) | ML/LLM evaluation and observability | 100+ evals, 20+ drift tests; no conformal code (verified in the 0.7.23 wheel) | 0.7.23 (2026-09-11) |
| [UQLM](https://github.com/cvs-health/uqlm) | LLM hallucination scoring | Consistency, token-probability and judge scorers; `ScoreCalibrator`; no finite-sample guarantee; requires torch | 0.6.6 (2026-09-03) |
| [LM-Polygraph](https://github.com/IINemo/lm-polygraph) | LLM UQ benchmark | 50+ methods | 0.7.0 (2026-05-04) |
| [ValidMind](https://pypi.org/project/validmind/2.13.14/) | Model validation library | Credit validation tests, calibration drift, documentation export; AGPL/commercial | 2.13.14 (2026-09-03) |
| [seismometer](https://github.com/epic-open-source/seismometer) | Local validation of healthcare AI | Cohort and fairness tables, monitoring over time, HTML reports; BSD-3 | Not checked |
| [ppi_py](https://github.com/aangelopoulos/ppi_py) | Prediction-powered inference | PPI CIs, power analysis; no stratified PPI | 0.2.3 |
| [rul-datasets](https://github.com/tilman151/rul-datasets) | RUL dataset loaders | C-MAPSS, N-C-MAPSS, FEMTO, XJTU-SY, domain adaptation | Not checked |
| [OpenSTEF](https://github.com/OpenSTEF/openstef/pull/1060) | Energy forecasting | Inlined conformalized quantile calibration (PR #1060) | 4.4.3 (2026-09-21) |
| [venn-abers](https://pypi.org/project/venn-abers/) | Venn-ABERS calibration | Calibrated probability intervals | 1.5.4 (2026-08-12) |
| netcal, conformal-tights, probly, pytorch-ood | Narrow calibration, conformal and OOD tools | Each covers one slice; none monitors coverage (unverified in depth) | Active in 2026 |

Strategy proposals also named online-cp (delayed-feedback online conformal for its own predictors) and covmetrics (conditional coverage metrics). Neither was checked in the research (unverified).

For comparison, UEQ was at 1.0.1 on PyPI (2025-09-29) with 11 stars and 1 fork.

## White space

Each item lists the evidence for the gap and the reviewers' caveat.

1. **Coverage monitoring of deployed intervals and sets with delayed labels**, per segment and horizon, with a pending-outcome ledger.
   - Evidence: conformal libraries stop at producing intervals, and NannyML estimates point-prediction performance, not interval coverage. Delayed-feedback ACI appears only in 2026 research (arXiv 2609.07251, 2603.08578; contents unverified).
   - Caveat: MAPIE `RiskMonitoring` plus a groupby covers part of this. The reviewers found no evidence that "coverage SLAs" are a named practice.
   - What the C-MAPSS prototype added: when label delay depends on the label, naive "coverage so far" is biased (0.846 against an eventual 0.885 in one seed); row-level breach tests on clustered rows raise false alarms; and the segment that failed was only identifiable from an attribute that arrives with the label. A monitor has to handle all three.
2. **A production recipe for recalibrating conformal forecasts.**
   - Evidence: MAPIE issue #505 asked for CQR plus production recalibration for energy load and was closed as "Discussion in progress".
   - Caveat: serious teams recalibrate in-house (the HEFTCom2024 winner, OpenSTEF).
3. **Library-agnostic evaluation of time-series conformal methods.**
   - Evidence: a 2026 benchmark (arXiv 2601.18509) had to implement MSCP, ACI and AcMCP itself. Implementations are tied to darts, Nixtla, sktime, MAPIE and TorchCP data models.
4. **A maintained successor to uncertainty-toolbox** for interval, quantile and set outputs, with calibration over time, per group and per horizon.
   - Evidence: uncertainty-toolbox has been dormant since January 2023 and is Gaussian-only.
   - Caveat: MAPIE and crepes have metrics, but not over time or segments.
5. **Coverage evidence reports over a model's lifecycle**: declared metrics, subgroup coverage over time, drift and recalibration events.
   - Evidence: EU AI Act Art. 15 asks high-risk systems to declare accuracy metrics and perform consistently over their lifecycle, but it does not require UQ.
   - Caveat: ValidMind and seismometer already produce validation reports. UEQ's report would support documentation and must not claim compliance.
6. **Reliability audits of tabular foundation models on real data** (arXiv 2605.28554).
   - Not pursued in the roadmap.

## Where UEQ cannot win

- **Breadth of conformal methods.** MAPIE had five releases in 2026. crepes has predictive systems and martingales, and puncc covers detection.
- **Deep-learning and Bayesian UQ.** Lightning-UQ-Box, laplace-torch and TorchUncertainty cover this ground (open issues #13, #14, #19, #20).
- **LLM and structured-output uncertainty.** UQLM, LM-Polygraph, TorchCP and MAPIE cover this ground (#24).
- **Forecasting-native conformal intervals.** darts, Nixtla and sktime cover this ground (#23 as originally framed).
- **General observability and drift.** Evidently and NannyML cover this ground.
- **Domain-specific conformal for graphs, vision and detection.** TorchCP, puncc and Lightning-UQ-Box cover this ground.

## Regulatory context (not compliance claims)

| Instrument | What it asks | Relevance to UQ | Verification |
|---|---|---|---|
| EU AI Act Art. 15 | Declared accuracy metrics and consistent performance over the lifecycle for high-risk systems | Indirect; UQ not required | Page blocked; secondary sources |
| EU AI Act Art. 14 | Human oversight | Any review queue satisfies it; UQ not required | Page blocked (unverified) |
| EU AI Act Annex III 5(b) | Credit scoring is high-risk; fraud detection excluded | Context for credit | Corroborated |
| Digital Omnibus | Annex III duties moved to 2 Dec 2027; Annex I medical-device AI to Aug 2028 | Timing | Secondary sources |
| SR 26-2 / OCC Bulletin 2026-13 | Replaces SR 11-7 (17 Apr 2026); ongoing monitoring | Indirect | Corroborated secondhand |
| PRA SS1/23, EBA GL/2017/16 | Post-model adjustments; margin of conservatism | Estimation-error evidence | Unverified (PDFs not opened) |
| EU SOGL Art. 157 | Probabilistic FRR sizing covering >= 99% of imbalances | Falls on TSOs | Snippet-verified |
| FDA draft AI device guidance (Jan 2025) | Recommends showing confidence or uncertainty | Manufacturers' submissions | Unverified (secondary summaries) |

None of these requires conformal prediction or any specific UQ method. UEQ will describe its evidence report as supporting documentation only.

## Sources

URLs below are those cited in the research brief. Some could not be opened in the research environment; claims resting on them are marked (unverified) above.

**UEQ and competitors**
- https://pypi.org/pypi/ueq/json
- https://github.com/scikit-learn-contrib/MAPIE
- https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/HISTORY.md
- https://github.com/scikit-learn-contrib/MAPIE/issues/505
- https://github.com/scikit-learn-contrib/MAPIE/tree/master/examples/risk_control/2-advanced-analysis
- https://pypi.org/pypi/mapie/json
- https://github.com/henrikbostrom/crepes
- https://pypi.org/pypi/crepes/json
- https://github.com/ml-stat-Sustech/TorchCP
- https://pypi.org/project/torchcp/
- https://github.com/deel-ai/puncc
- https://github.com/awslabs/fortuna
- https://github.com/uncertainty-toolbox/uncertainty-toolbox
- https://pypi.org/pypi/uncertainty-toolbox/json
- https://github.com/lightning-uq-box/lightning-uq-box
- https://github.com/aleximmer/Laplace
- https://pypi.org/project/torch-uncertainty/
- https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html
- https://github.com/Nixtla/mlforecast
- https://unit8co.github.io/darts/generated_api/darts.models.forecasting.conformal_models.html
- https://github.com/unit8co/darts/blob/master/CHANGELOG.md
- https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.conformal.ConformalIntervals.html
- https://www.sktime.net/en/v0.35.0/api_reference/auto_generated/sktime.forecasting.enbpi.EnbPIForecaster.html
- https://github.com/awslabs/gluonts
- https://github.com/NannyML/nannyml
- https://pypi.org/pypi/nannyml/json
- https://github.com/evidentlyai/evidently
- https://github.com/cvs-health/uqlm
- https://github.com/IINemo/lm-polygraph
- https://pypi.org/project/validmind/2.13.14/
- https://github.com/epic-open-source/seismometer
- https://github.com/aangelopoulos/ppi_py
- https://github.com/tilman151/rul-datasets
- https://github.com/nasa/progpy
- https://github.com/OpenSTEF/openstef/pull/1060
- https://github.com/OpenSTEF/openstef/pull/1058
- https://pypi.org/project/venn-abers/
- https://skforecast.org/latest/user_guides/probabilistic-forecasting-conformal-calibration.html

**Datasets**
- https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
- https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip
- https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip
- https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
- https://opendata.elia.be/explore/dataset/ods031/
- https://opendata.elia.be/explore/dataset/ods032/
- https://opendata.elia.be/explore/dataset/ods087/
- https://www.elia.be/en/grid-data/elia-open-data-license
- https://zenodo.org/records/13950764
- https://github.com/jbrowell/HEFTcom24-Analysis
- https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition
- https://data.openei.org/submissions/5772
- https://github.com/PERFORM-Forecasts/documentation
- https://github.com/jeslago/epftoolbox
- http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html
- https://www.kaggle.com/c/m5-forecasting-uncertainty/data
- https://github.com/Nixtla/m5-forecasts
- https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- https://github.com/wenhaomin/LaDe
- https://physionet.org/content/challenge-2019/1.0.0/
- https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/46xm-urtu
- https://github.com/MedMNIST/MedMNIST
- https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- https://www.kaggle.com/competitions/home-credit-default-risk
- https://github.com/feedzai/bank-account-fraud
- https://github.com/feedzai/fifar-dataset
- https://www.openml.org/d/41214
- https://github.com/HarvestStat/HarvestStat-Africa
- https://raw.githubusercontent.com/HarvestStat/HarvestStat-Africa/main/LICENSE
- https://github.com/wur-ai/agml-cy-bench
- https://raw.githubusercontent.com/WUR-AI/AgML-CY-Bench/main/LICENSE
- https://raw.githubusercontent.com/WUR-AI/AgML-CY-Bench-Modeling/main/README.md
- https://data.humdata.org/dataset/harmonized-food-insecurity-dataset-hfid
- https://www.chc.ucsb.edu/data/chirps3
- https://github.com/socialfoundations/folktables
- https://github.com/Shifts-Project/shifts
- https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- https://github.com/PolyAI-LDN/task-specific-datasets
- https://github.com/clinc/oos-eval
- https://github.com/hendrycks/test
- https://github.com/ParticleMedia/RAGTruth
- https://github.com/mandarjoshi90/triviaqa
- https://github.com/BrandonKaza32/Silica-flotation-prediction

**Domain evidence**
- https://arxiv.org/abs/2507.01579
- https://arxiv.org/abs/2505.10367
- https://github.com/BigdogManLuo/HEFTcom24
- https://www.caiso.com/documents/review-of-the-mosaic-quantile-regression-nov-20-2023.pdf
- https://ionanalytics.com/insights/infralogic/ercot-error-spurs-uncertainty-frustration-for-texas-asset-owners/
- https://www.statnett.no/globalassets/for-aktorer-i-kraftsystemet/systemansvaret/metoder---innsendt-til-godkjenning/metode-iht.-sogl-art-157-29.06.2022.pdf
- https://aws.amazon.com/blogs/machine-learning/transition-your-amazon-forecast-usage-to-amazon-sagemaker-canvas
- https://github.com/Mcompetitions/M5-methods
- https://www.nature.com/articles/s41598-026-40637-w
- https://github.com/physionetchallenges/evaluation-2019
- https://github.com/CommonAccord/Cmacc-Org/blob/75f2c6f8c4a7dccd35ba088f581ae7e21ebac87c/Doc/G/EU/Artificial_Intelligence_Act/Annex/III.md
- https://github.com/MaximilianSuliga/Conformal-Active-Learning-for-Reject-Inference
- https://github.com/feedzai/Uncertainty-Aware-Systems-for-Human-AI-Collaboration
- https://github.com/OCHA-DAP/pa-anticipatory-action
- https://aws.amazon.com/blogs/machine-learning/preserve-access-and-explore-alternatives-for-amazon-lookout-for-equipment/
- https://www.mathworks.com/help/predmaint/ug/rul-estimation-using-rul-estimator-models.html
- https://soda.io/blog/soda-acquires-nannyml
- https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-use-augmented-ai-a2i-human-review-loops.html
- https://cfp.pydata.org/pydataglobal2025/talk/8U7WLS/
- https://torchcp.readthedocs.io/en/latest/torchcp.llm.html

**Research**
- https://arxiv.org/pdf/2609.07251
- https://arxiv.org/pdf/2603.08578
- https://arxiv.org/abs/2601.18509
- https://arxiv.org/html/2605.28554
- https://arxiv.org/abs/2412.13159
- https://arxiv.org/abs/2502.04935
- https://arxiv.org/abs/2301.09633

**Regulation**
- https://artificialintelligenceact.eu/article/15/
- https://artificialintelligenceact.eu/article/14/
- https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/
- https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html
- https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm
- https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2023/ss123.pdf
- https://www.kslaw.com/news-and-insights/fda-releases-draft-guidance-on-submission-recommendations-for-ai-enabled-device-software-functions
- https://www.covingtondigitalhealth.com/2026/01/hhs-proposes-changes-to-the-health-it-certification-program-and-information-blocking-regulations-in-hti-5-proposed-rule/
