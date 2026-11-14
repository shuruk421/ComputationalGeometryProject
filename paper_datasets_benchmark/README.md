# Consent-aware MEB on the datasets of "Selecting Sub-tables for Data Exploration"

This folder benchmarks the project's two consent-aware minimum-enclosing-ball
algorithms (`db_consents.incremental_distance_based` and
`db_consents.decremental_distance_based`) on public copies of the datasets used in

> Y. Amsterdamer, S. B. Davidson, T. Milo, K. Razmadze, A. Somech.
> *Selecting Sub-tables for Data Exploration* (2023).
> PDF: https://u.cs.biu.ac.il/~amstery/files/amsterdamer2023selecting.pdf
> (extended version: arXiv:2203.02754).

## 1. What the paper uses

The paper is about **SubTab**: picking a small, informative k x l sub-table
(rows + columns) of a large table for exploratory data analysis. It has no
consent, oracle, or geometry component.

**Datasets (paper Section 5.1).** Sizes are as the paper states them.

| Code | Paper's description | Paper's source | Public copy used here | Fidelity |
|---|---|---|---|---|
| FL | Flights, 6M rows x 32 cols | Kaggle `usdot/flight-delays` (flights.csv, 2015 US DOT on-time data) | BTS TranStats *On-Time Reporting* table, **January 2015** (469,968 flights x 110 cols) + OpenFlights airport coordinates | **Substitute**: same underlying BTS table, one month instead of the full year. Kaggle downloads need a login. |
| SP | Spotify, 42K x 15 | Kaggle competition `bfh-spotify-challenge` | `spotify_all.csv` from the authors' group's `pd-explain` repo (its notebook cites the same competition), 174,389 x 21 | **Closest public version**: same audio-feature columns. It is larger than the paper's table, which is probably a split cut from it (not verifiable). |
| CY | Cyber-security, 30K x 15 (12 columns in Sec. 5.3.2) | Honeynet challenges, as used in REACT [46] | The four REACT packet tables published by the authors' group in `TAU-DB/ATENA-A-EDA` (`benchmark/atena/datasets/cyber/1-4.tsv`), 23,366 packets x 17 | **Closest public version**: same Honeynet captures. The row count differs from the paper's. |
| CC | Credit card frauds, 250K x 31 | Kaggle `mlg-ulb/creditcardfraud` | Identical file from TensorFlow's public mirror, 284,807 x 31 | Exact |
| USF | US Funds, 23.5K x 298 | Kaggle `stefanoleone992/mutual-funds-and-etfs` (MutualFunds.csv) | GitHub mirror of MutualFunds.csv, 23,783 x 298 | Exact |
| BL | Bank Loans, 110K x 19 | Kaggle `panamby/bank-loan-status-dataset` | GitHub mirror of credit_train.csv + credit_test.csv, 110,867 x 19 | Exact |

The paper also replays 122 recorded EDA sessions over CY (from REACT [46]).
Those are query logs, not a dataset that can be geometrised, so they are not used here.
Its reference list mislabels some links ([1] "Cyber Dataset" points to the
credit-card URL, [3] "Flights Dataset" points to Honeynet, [6] "Funds Dataset" points to the
bank-loan URL). The URLs themselves are unambiguous.

**Airbnb, Chicago crimes, ADS-B.** The paper uses none of them. Its FL is the
BTS on-time *table*, not ADS-B trajectories. I also grepped the author's two consent
papers (Drien, Amarilli, Amsterdamer, ICDE 2021, *Managing Consent for Data Access in
Shared Databases*; Drien et al. 2023, *Query-Guided Resolution in Uncertain
Databases*). Neither mentions Airbnb, Chicago or ADS-B. They use a synthetic
generator, NELL, and TPC-H.

**Consent assignment in the paper.** None, for any dataset. The closest precedent is in
the author's related work. Drien et al. 2023 draws a per-tuple Boolean (correctness)
either at a **fixed probability (0.5)** or from probabilities produced by a **random
decision tree over the tuple's metadata**. The random-consent contrast below uses the
first mechanism at varying rates.

**Metrics in the paper.** Sub-table quality (cell coverage, diversity, combined
score), wall-clock time (pre-processing up to 85 s on CC, selection a few
seconds), and user-study measures. None of them is a consent-request count, so no
number here is directly comparable to the paper. Like the rest of this project, we
report **oracle calls** (unique consent requests, `Oracle.get_call_count()`), also
normalised by n, and **run time**.

## 2. Setting it up as an MEB problem

| Data | Points (after cleaning) | 2-D coordinates passed to the algorithms |
|---|---|---|
| CC | 275,663 transactions (9,144 duplicate points dropped) | V1, V2 (the two leading PCA components), z-scored |
| SP | 151,617 tracks (22,772 duplicates dropped) | valence, energy (the "mood plane"), z-scored |
| CY | 5,664 TCP connections (7,672 streams; 2,008 duplicates dropped) | source port / 65535, destination port / 65535 of the first packet (the "port plane") |
| USF | 20,817 funds | 3-year volatility, 3-year annualised return (risk-return plane), z-scored |
| BL | 59,208 loans (repeated Loan IDs, missing income/debt, the 99,999,999 loan-amount sentinel and DTI >= 100% dropped) | log10 annual income, debt-to-income %, z-scored |
| FL | 284 origin airports in the contiguous US (aggregated from 469,968 flights) | equirectangular projection around the airport centroid, in **units of 1000 km** |

* **Duplicates.** `Oracle` caches answers by coordinate tuple, so two rows at the same
  point would share one consent answer. Exact coordinate duplicates are therefore
  dropped (the first row is kept).
* **Numerical scale.** This matters; see section 6. All algorithm inputs are O(1):
  across all runs the ground-truth radii range from 0.53 to 27 units.
* **Why FL is one point per airport.** A flight's location is its airport, so
  per-flight points would be massive duplicates. Alaska, Hawaii and the territories are
  excluded because an equirectangular projection is meaningless at trans-Pacific scale.
  The figures reproject the Web-Mercator basemap tiles to the same equirectangular km
  grid, so points, circles and map line up.

## 3. Consent column analysis

The paper specifies no consent mechanism, so each dataset uses a **threshold on one
column** ("consent <=> value >= t" or "<= t"). Thresholds are quantiles over the whole
cleaned dataset, so the consenting fraction can be swept from 5% to 95%. As a contrast,
**random (Bernoulli) consent at the same rate** is run on the same points.

`consent_analysis.py` scores every candidate on a 20,000-point sample. Two numbers
compare the consenting MEB with the all-points MEB:

* `r/r0` = the radius ratio (consenting / all points);
* `shift` = the centre displacement divided by the all-points radius.

Each is compared with the same numbers under random consent at that rate. Full tables
are in `results/<dataset>_consent_candidates.csv`.

| Data | Chosen consent rule | Story | r/r0, shift at 10% / 50% / 90% (random consent in brackets) | Other candidates and why not |
|---|---|---|---|---|
| CC | **Amount <= t** | Cardholders share low-value transactions; large purchases are sensitive and identifying. | 0.41, 0.73 / 0.41, 0.73 / 0.61, 0.43 (0.55, 0.47 / 0.85, 0.19 / 1.00, 0.00) | Time >= t ("opt-in, not retroactive") is weaker at high rates. Class (fraud) is binary at 99.8%. |
| SP | **popularity >= t** (ties broken by release year, newer first) | Rights holders of popular tracks opt in for promotion. | 0.99, 0.02 / 1.00, 0.01 / 1.00, 0.00 (0.98, 0.01 / 1.00, 0.01 / 1.00, 0.00) | Year, duration and explicit are equally random-looking. **No column moves the MEB in this plane** (it is a dense box whose corners are always populated), so SP is the **"random-looking" contrast case**. Other planes (loudness-based) only looked structured because of a few -60 dB outliers. |
| CY | **connection start >= t** (fraction of the capture) | Users accepted the monitoring banner at time t; consent is not retroactive. | 0.85, 0.20 / 0.96, 0.06 / 1.00, 0.00 (0.97, 0.04 / 0.99, 0.01 / 1.00, 0.00) | Bytes <= t has only 51 distinct values, so heavy ties make the sweep impossible. Time is genuinely spatial: OS ephemeral source ports increase over time. |
| USF | **total net assets >= t** | Large funds have compliance teams that sign the data-sharing agreement. | 0.41, 0.40 / 0.61, 0.28 / 0.80, 0.21 (0.83, 0.10 / 0.97, 0.03 / 1.00, 0.00) | Expense ratio is similar but weaker at 90%. Age is near-random. Morningstar rating has only 5 values. |
| BL | **loan amount <= t** | Small loans use a standard digital contract with a consent box; large loans are negotiated. | 0.80, 0.20 / 0.88, 0.12 / 0.88, 0.12 (0.80, 0.14 / 0.90, 0.08 / 0.98, 0.04) | Credit score (the obvious story) and credit-history length behave like random consent: the MEB is pinned by a few extreme-income borrowers, whose consent is unrelated to either column. Even loan amount is only **weakly** structured, because it depends on whether those few borrowers have small loans. In the seed-0 benchmark subsample they do, so from 20% upwards the consenting MEB equals the all-points MEB. |
| FL | **cancellation rate <= t** (January 2015) | Airports hit by the late-January Northeast blizzard decline to publish (reputational risk). | 0.89, 0.14 / 0.95, 0.20 / 1.00, 0.00 (0.94, 0.15 / 0.98, 0.04 / 1.00, 0.00) | On-time rate, volume and taxi-out are weaker at 25-75%. |

**Better mechanism (proposal).** For datasets whose MEB is pinned by a few extreme
records (CC, BL), whether those few records consent decides almost everything. A
consent model that makes extremes systematically less willing (e.g. a logistic in a
sensitive attribute) is the realistic hard case. A threshold on such an attribute (CC
Amount, USF assets) is the deterministic version of that, and is what is used here.
The paper's group's decision-tree-over-metadata mechanism (Drien et al. 2023) would be a
natural stochastic generalisation.

## 4. Benchmark protocol

* For each target fraction 5, 10, 20, ..., 90, 95% and each of **5 seeds**:
  1. draw a fresh random subsample of **n = 5,000** points (FL: all 284 airports; CY:
     5,000 of 5,664);
  2. run both algorithms on the same points, under the column rule and under random
     consent. **Incremental runs first on even seeds and decremental first on odd
     seeds**, because other jobs share the machine.
* **Scaling with n.** The column rule at ~50% consent, for n = 250 ... 32,000 (capped at
  the dataset size), 5 seeds each.
* **Ground truth.** `incremental_distance_based` on the consenting points only, with a
  throwaway Oracle. It is independently **certified**: every consenting point is
  enclosed, and the centre lies in the convex hull of the points on the sphere (the MEB
  optimality condition), checked with NNLS.
* **A run "matches"** if |r - r_gt| <= 1e-7 r_gt and its centre is within 1e-6 r_gt of
  the true centre.
* **Lower-bound reference.** Every run also records how many non-consenting points lie
  outside the consenting MEB. Decremental must query each of these before it can drop
  it, so this count is a lower bound on its oracle calls. It is drawn as a grey line.

## 5. Results

The numbers below are means over 5 seeds (± SD for calls) at n = 5,000. Raw rows are in
`results/<dataset>_sweep.csv` and `results/<dataset>_vs_n.csv`; aggregates are in
`results/<dataset>_summary.json`.

#### CC Credit card frauds (n=5000, 5 seeds; runs=300, mismatches=0, cert failures=0, sweep 99.4s + vs_n 18.2s)

| consent rule | % consenting | oracle calls inc | oracle calls dec | time inc (s) | time dec (s) | random: calls inc / dec | random: time inc / dec (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amount (EUR) <= 0.98 | 5% | 724 ± 139 | 350 ± 212 | 0.020 | 7.787 | 406 / 69 | 0.025 / 1.391 |
| Amount (EUR) <= 3.99 | 20% | 213 ± 67 | 50 ± 49 | 0.025 | 1.502 | 134 / 14 | 0.035 / 0.226 |
| Amount (EUR) <= 23.7 | 50% | 69 ± 18 | 13 ± 6 | 0.030 | 0.331 | 52 / 4 | 0.019 / 0.074 |
| Amount (EUR) <= 103 | 80% | 53 ± 11 | 10 ± 5 | 0.026 | 0.197 | 30 / 4 | 0.021 / 0.074 |
| Amount (EUR) <= 374 | 95% | 38 ± 5 | 6 ± 2 | 0.042 | 0.117 | 41 / 2 | 0.052 / 0.052 |

*Scaling with n (~50% consent):* n=250: inc 35 calls/0.004s, dec 7 calls/0.022s; n=4000: inc 75 calls/0.024s, dec 8 calls/0.156s; n=32000: inc 127 calls/0.202s, dec 9 calls/1.245s

#### SP Spotify tracks (n=5000, 5 seeds; runs=300, mismatches=0, cert failures=0, sweep 44.2s + vs_n 8.7s)

| consent rule | % consenting | oracle calls inc | oracle calls dec | time inc (s) | time dec (s) | random: calls inc / dec | random: time inc / dec (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity > 64, or = 64 and year >= 2016 | 5% | 537 ± 226 | 111 ± 133 | 0.028 | 2.385 | 346 / 49 | 0.035 / 1.085 |
| popularity > 47, or = 47 and year >= 1999 | 20% | 114 ± 29 | 14 ± 7 | 0.021 | 0.380 | 102 / 16 | 0.031 / 0.364 |
| popularity > 26, or = 26 and year >= 1985 | 50% | 49 ± 16 | 6 ± 3 | 0.035 | 0.140 | 42 / 6 | 0.022 / 0.147 |
| popularity > 0, or = 0 and year >= 2016 | 80% | 31 ± 8 | 4 ± 1 | 0.043 | 0.073 | 28 / 3 | 0.027 / 0.041 |
| popularity > 0, or = 0 and year >= 1935 | 95% | 26 ± 4 | 3 ± 1 | 0.034 | 0.039 | 28 / 3 | 0.045 / 0.050 |

*Scaling with n (~50% consent):* n=250: inc 31 calls/0.005s, dec 6 calls/0.013s; n=4000: inc 50 calls/0.023s, dec 7 calls/0.135s; n=32000: inc 45 calls/0.132s, dec 5 calls/0.511s

#### CY Cyber-security packets (TCP connections) (n=5000, 5 seeds; runs=280, mismatches=0, cert failures=0, sweep 977.5s + vs_n 9.1s)

| consent rule | % consenting | oracle calls inc | oracle calls dec | time inc (s) | time dec (s) | random: calls inc / dec | random: time inc / dec (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| connection start (fraction of capture) >= 0.949 | 5% | 1830 ± 112 | 1436 ± 7 | 0.023 | 58.800 | 294 / 70 | 0.028 / 2.279 |
| connection start (fraction of capture) >= 0.723 | 20% | 1116 ± 142 | 833 ± 6 | 0.034 | 34.589 | 100 / 15 | 0.032 / 0.366 |
| connection start (fraction of capture) >= 0.369 | 50% | 58 ± 17 | 16 ± 1 | 0.035 | 0.377 | 46 / 6 | 0.033 / 0.153 |
| connection start (fraction of capture) >= 0.148 | 80% | 38 ± 5 | 7 ± 1 | 0.049 | 0.205 | 33 / 4 | 0.042 / 0.094 |
| connection start (fraction of capture) >= 0.0369 | 95% | 28 ± 4 | 4 ± 0 | 0.042 | 0.095 | 25 / 3 | 0.035 / 0.063 |

*Scaling with n (~50% consent):* n=250: inc 31 calls/0.006s, dec 10 calls/0.045s; n=2000: inc 61 calls/0.022s, dec 9 calls/0.117s; n=5664: inc 63 calls/0.057s, dec 18 calls/0.831s

#### USF US mutual funds (n=5000, 5 seeds; runs=300, mismatches=0, cert failures=0, sweep 86.8s + vs_n 106.4s)

| consent rule | % consenting | oracle calls inc | oracle calls dec | time inc (s) | time dec (s) | random: calls inc / dec | random: time inc / dec (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| total net assets (USD) >= 16.9B | 5% | 602 ± 188 | 210 ± 70 | 0.020 | 4.766 | 315 / 61 | 0.021 / 1.201 |
| total net assets (USD) >= 3.6B | 20% | 151 ± 26 | 66 ± 13 | 0.028 | 1.575 | 95 / 12 | 0.016 / 0.197 |
| total net assets (USD) >= 721.3M | 50% | 83 ± 10 | 40 ± 4 | 0.037 | 0.958 | 38 / 3 | 0.017 / 0.035 |
| total net assets (USD) >= 129.1M | 80% | 47 ± 5 | 16 ± 2 | 0.037 | 0.305 | 36 / 4 | 0.026 / 0.064 |
| total net assets (USD) >= 22.0M | 95% | 29 ± 12 | 4 ± 5 | 0.026 | 0.075 | 24 / 2 | 0.018 / 0.020 |

*Scaling with n (~50% consent):* n=250: inc 29 calls/0.002s, dec 8 calls/0.020s; n=4000: inc 73 calls/0.024s, dec 32 calls/0.637s; n=20817: inc 205 calls/0.126s, dec 144 calls/10.524s

#### BL Bank loans (n=5000, 5 seeds; runs=300, mismatches=0, cert failures=0, sweep 40.2s + vs_n 10.2s)

| consent rule | % consenting | oracle calls inc | oracle calls dec | time inc (s) | time dec (s) | random: calls inc / dec | random: time inc / dec (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| loan amount <= 69K | 5% | 424 ± 59 | 80 ± 82 | 0.031 | 2.108 | 342 / 57 | 0.022 / 1.349 |
| loan amount <= 146K | 20% | 104 ± 19 | 11 ± 7 | 0.019 | 0.290 | 98 / 13 | 0.028 / 0.284 |
| loan amount <= 266K | 50% | 59 ± 9 | 7 ± 6 | 0.035 | 0.238 | 41 / 7 | 0.026 / 0.170 |
| loan amount <= 450K | 80% | 35 ± 6 | 3 ± 2 | 0.033 | 0.066 | 29 / 3 | 0.029 / 0.041 |
| loan amount <= 675K | 95% | 37 ± 12 | 3 ± 2 | 0.032 | 0.071 | 35 / 2 | 0.033 / 0.046 |

*Scaling with n (~50% consent):* n=250: inc 29 calls/0.004s, dec 6 calls/0.021s; n=4000: inc 68 calls/0.024s, dec 9 calls/0.189s; n=32000: inc 90 calls/0.146s, dec 5 calls/0.515s

#### FL US flights (airports) (n=284, 5 seeds; runs=270, mismatches=0, cert failures=0, sweep 6.4s + vs_n 1.3s)

| consent rule | % consenting | oracle calls inc | oracle calls dec | time inc (s) | time dec (s) | random: calls inc / dec | random: time inc / dec (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cancellation rate <= 0 | 7% | 98 ± 10 | 55 ± 0 | 0.002 | 0.104 | 86 / 25 | 0.002 / 0.059 |
| cancellation rate <= 0.0119 | 20% | 60 ± 9 | 27 ± 0 | 0.003 | 0.128 | 54 / 15 | 0.004 / 0.042 |
| cancellation rate <= 0.026 | 50% | 37 ± 7 | 20 ± 0 | 0.005 | 0.099 | 29 / 7 | 0.005 / 0.025 |
| cancellation rate <= 0.0551 | 80% | 22 ± 3 | 9 ± 0 | 0.005 | 0.033 | 20 / 3 | 0.008 / 0.007 |
| cancellation rate <= 0.101 | 95% | 21 ± 5 | 3 ± 0 | 0.008 | 0.009 | 14 / 3 | 0.004 / 0.005 |

*Scaling with n (~50% consent):* n=35: inc 13 calls/0.001s, dec 6 calls/0.004s; n=142: inc 28 calls/0.003s, dec 13 calls/0.031s; n=284: inc 35 calls/0.005s, dec 20 calls/0.106s

### Key findings

* **Consent requests: decremental always wins.** In every one of the 875 paired runs
  (660 sweep pairs = 6 datasets x 11 rates x 2 mechanisms x 5 seeds, plus 215 scaling
  pairs), decremental made **no more** oracle calls than incremental.
  * Typically it needs 2-10x fewer calls.
  * At high consent it asks only for the 2-5 support points, while incremental still
    asks ~20-50 ("learning" queries for points that fall outside the growing ball in
    shuffled order).
  * Its count tracks the grey lower bound (non-consenters outside the consenting MEB)
    almost exactly, so it asks about little beyond what is unavoidable.
  * The advantage nearly vanishes when consent is a thin cluster and most points lie
    outside it. On CY at 5-30% consent it is only 1.2-1.3x (1,436 vs 1,830 calls at 5%):
    both algorithms must then ask about ~30% of all points.
* **Run time: incremental always wins**, by a large factor at low consent.
  * At 5% consent decremental is 50-400x slower on CC, SP, USF, BL and FL (CC: 7.8 s vs
    0.02 s), and 2,500x slower on CY (59 s vs 0.02 s).
  * Why: each decremental round re-runs Welzl on every active point but removes only
    the <= 3 non-consenting boundary points, so the work is ~ (points to peel / 3) x n.
  * The gap closes towards 95% consent (1-3x). Decremental was faster in only 35 of the
    660 sweep pairs: 29 of them at >= 70% consent, and all within timing noise.
* **No crossover in oracle calls.** In time, the two curves only meet at 90-100%
  consent (FL 0.009 s vs 0.008 s at 95%; SP 0.039 s vs 0.034 s).
* **Structured consent costs more than random consent at the same rate**, for both
  algorithms. Decremental is hit hardest:
  * CC at 5%: 350 calls under the Amount rule vs 69 under random consent;
  * CY at 20%: 833 vs 15 calls, and 35 s vs 0.4 s.

  When consent is clustered, many non-consenters lie outside the small consenting ball,
  and all of them must be asked about and peeled.
* **Scaling with n at ~50% consent.**
  * Incremental calls grow slowly, roughly logarithmically: CC 35 -> 127 from
    n = 250 to 32,000.
  * Decremental calls stay flat when few non-consenters lie outside the consenting MEB
    (CC 7 -> 9, SP 6 -> 5, BL 6 -> 5).
  * When a fixed fraction of the data lies outside it, decremental calls grow linearly
    (USF: 8 -> 144 at n = 20,817, against incremental's 29 -> 205). The two then
    converge, so decremental's call advantage shrinks with n while its time penalty
    grows (USF at n = 20,817: 10.5 s vs 0.13 s).
  * Incremental's run time grows roughly linearly with n. Decremental's does too when
    the number of peeling rounds stays flat (CC, SP, BL), and super-linearly when the
    rounds grow with n (USF: 525x the time for 83x the points).
* **Spatial pictures** (`figures/*_spatial.png`).
  * **CC.** Only small purchases consent, and the extreme-V2 transactions (large
    amounts) never do. The consenting MEB stays at 0.45-0.67x the all-points radius
    even at 80-95% consent, whereas random consent returns to 1.0x by 50%.
  * **USF.** Big funds sit in the middle of the risk-return cloud: 0.34x at 5%, 0.78x
    at 80%.
  * **FL.** The circle moves south-west and shrinks (0.80x at 7%) as blizzard-hit
    north-eastern airports drop out.
  * **CY.** At 5-20% the consenting connections are mostly one late port scan (a
    vertical line at a single source port) plus one low-port outlier. The ball shrinks to
    0.85x, and the dense block of earlier connections must be peeled.
  * **BL.** Weak: the effect is visible only at 5%.
  * **SP.** Does not move at all. It is the random-looking contrast case.

## 6. Correctness check and a bug in `db_consents`

* **All 1,750 benchmark runs match the certified ground truth** (0 mismatches,
  0 certificate failures; see `mismatches` / `gt_certificate_failures` in the summary
  JSONs).
* **Bug (not fixed here, `db_consents.py` untouched).** `decremental_distance_based`
  recognises boundary points with an **absolute** tolerance:
  `abs(dist_sq - radius_sq) <= GEOMETRY_TOLERANCE` (1e-10).
  * Once radius² is large (around 1e5 and up), rounding error in the squared distances
    exceeds 1e-10. The true support points are then not recognised, the loop stops at
    once, and it returns a ball that still rests on non-consenting points.
  * `scale_check.py` runs the *same* point sets at different scales
    (`results/scale_check.csv`):

    | data, units | incremental matches | decremental matches | decremental mean calls |
    |---|---|---|---|
    | FL, 1000 km (used here) | 15/15 | 15/15 | 18.7 |
    | FL, km | 15/15 | **0/15** | 0.2 |
    | FL, m | 15/15 | **1/15** | 6.3 |
    | CC, z-score (used here) | 15/15 | 15/15 | 22.7 |
    | CC, z x 1e3 | 15/15 | **2/15** | 1.9 |
    | CC, z x 1e5 | 15/15 | **2/15** | 2.4 |

  * Wrong radii are up to 2.6x the truth.
  * Incremental is unaffected in these tests: its in-ball test errs on the side of
    querying.
  * Fix suggestion: use a relative tolerance (e.g. `<= 1e-9 * max(1, radius_sq)`) in
    `decremental_distance_based`, `is_point_in_sphere` and the Welzl base cases.
  * **Until then, any geographic benchmark that projects to km (Airbnb, Chicago crimes,
    ADS-B) must rescale its coordinates to O(1) before calling the algorithms.**
* **Oracle caching by coordinates.** Rows with identical coordinates share one cached
  consent answer, which is wrong if their consent differs. Handled here by
  de-duplicating points.
* **Basemap.** The CARTO Positron endpoint used by `airbnb_benchmark/plot_amsterdam_meb.py`
  now returns an "API KEY REQUIRED" placeholder for every tile. This folder uses Esri's
  keyless World Light Gray Canvas instead.

## 7. Files and how to reproduce

```
download_data.py     public copies -> data/<dataset>/   (~310 MB, gitignored)
datasets.py          loaders: cleaning, coordinates, consent candidates, de-duplication
consent_analysis.py  candidate scoring -> results/<dataset>_consent_candidates.csv
benchmark.py         sweeps + scaling + correctness -> results/<dataset>_{sweep,vs_n}.csv, _summary.json
scale_check.py       numerical-scale experiment -> results/scale_check.csv
plots.py             figures -> figures/*.png (PNG is gitignored repo-wide)
meb_common.py        shared helpers (thresholds, certified ground truth, map tiles)
run_all.py           one entry point for all of the above
```

```bash
cd paper_datasets_benchmark
python -I run_all.py                                   # everything, ~25 min (CY sweep alone ~16 min)
python -I run_all.py --steps plot                      # redraw from results/
python -I run_all.py creditcard --steps bench,plot --n 2000 --seeds 3
python -I download_data.py --list                      # show the source URLs
```

Python 3.9 + numpy, scipy, matplotlib and Pillow; no pandas. Downloads are parsed as
CSV/TSV text only. The BTS zip is read member-by-member in memory, never extracted.

Figures: `figures/<dataset>_spatial.png` (small multiples over thresholds, column rule
vs random), `figures/<dataset>_performance.png` (calls, calls/n and time vs %
consenting), `figures/<dataset>_vs_n.png`, `figures/overview_calls.png`,
`figures/overview_time.png`.
