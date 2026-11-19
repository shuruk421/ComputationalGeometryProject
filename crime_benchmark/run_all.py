"""
Regenerate everything for the Chicago crimes consent-MEB study from the CSV.

    python crime_benchmark/run_all.py --csv /path/to/Crimes_Ordered.csv

Steps (each writes into crime_benchmark/):
  1. sample      one streaming pass, reservoir sample cached in sample_cache/ (gitignored)
  2. candidates  consent-column analysis          -> results/consent_candidates.json
  3. benchmark   incremental vs decremental runs  -> results/benchmark_{runs,summary}.json
     ablations   duplicates kept / feet units     -> results/correctness_ablations.json
  4. figures     maps + performance charts        -> *.png (gitignored)

Use --skip-benchmark to redraw figures from existing results, --quick for a
smoke test (writes the same result files, with a tiny configuration).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import benchmark_crimes  # noqa: E402
import consent_analysis  # noqa: E402
import crime_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Chicago crimes consent-MEB study: data, benchmark, figures.")
    ap.add_argument("--csv", default=crime_data.DEFAULT_CSV, help="path to Crimes_Ordered.csv")
    ap.add_argument("--reservoir", type=int, default=400_000, help="rows in the cached uniform sample")
    ap.add_argument("--rebuild-sample", action="store_true", help="ignore the cached sample")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--skip-benchmark", action="store_true", help="reuse results/*.json, only redraw")
    ap.add_argument("--skip-figures", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    records, meta, proj = crime_data.load(a.csv, k=a.reservoir, force=a.rebuild_sample)
    print(f"[run_all] {meta['clean_rows']:,} sampled rows after cleaning "
          f"(of {meta['source_data_rows']:,} in the file); reference point "
          f"{meta['ref_lat']:.4f}, {meta['ref_lng']:.4f}", flush=True)

    if not a.skip_benchmark:
        consent_analysis.run(records, proj, n=5000 if a.quick else 20000, meta=meta)
        benchmark_crimes.run_benchmark(records, proj, seeds=a.seeds, quick=a.quick)
        if not a.quick:
            benchmark_crimes.run_ablations(records, proj)

    if not a.skip_figures:
        import plot_chicago_crimes
        import plot_performance

        outs = plot_chicago_crimes.make_all(records, proj)
        outs += plot_performance.make_all()
        print("[run_all] figures:\n  " + "\n  ".join(outs))
    print(f"[run_all] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
