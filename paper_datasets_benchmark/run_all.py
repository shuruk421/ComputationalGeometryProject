"""
Regenerate everything for the paper-datasets benchmark.

    python -I run_all.py                       # download + analyse + benchmark + scale check + plots
    python -I run_all.py --steps plot          # only redraw figures from results/
    python -I run_all.py creditcard funds --steps bench,plot --n 1000 --seeds 3

Steps: download, analyse, bench, scale, plot (comma separated).
Datasets: creditcard, spotify, bankloans, funds, flights, cyber (default: all).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets  # noqa: E402

ALL_STEPS = ("download", "analyse", "bench", "scale", "plot")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    ap.add_argument("--n", type=int, default=5000, help="points per benchmark run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--vs-n", default="250,500,1000,2000,4000,8000,16000,32000")
    args = ap.parse_args()
    keys = args.datasets or list(datasets.LOADERS)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        sys.exit(f"unknown steps {bad}; choose from {ALL_STEPS}")
    t0 = time.time()
    if "download" in steps:
        import download_data
        sys.argv = ["download_data.py"] + keys
        download_data.main()
    if "analyse" in steps:
        import consent_analysis
        consent_analysis.main(keys)
    if "bench" in steps:
        import benchmark
        benchmark.main(keys + ["--n", str(args.n), "--seeds", str(args.seeds), "--vs-n", args.vs_n])
    if "scale" in steps:
        import scale_check
        scale_check.main()
    if "plot" in steps:
        import plots
        plots.main(keys + ["--n", str(args.n)])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
