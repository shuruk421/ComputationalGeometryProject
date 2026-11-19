"""Map figure for consent = UTC time of day (positions recorded before a cut-off hour).

Runs the same n=10,000 sweep as the main benchmark for the hour_of_day column,
saves results/sweep_hour_of_day.json and writes the small-multiples map.

    python flight_benchmark/plot_hour_of_day_map.py --csv .../flight.csv [--out FILE] [--seeds 3]
"""
import argparse, json, os
import flight_data as fd
import flight_meb_benchmark as bm
import plot_flight_figures as pf

HERE = os.path.dirname(os.path.abspath(__file__))
COL = "hour_of_day"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=fd.DEFAULT_CSV)
    ap.add_argument("--out", default=os.path.join(HERE, f"flight_map_{COL}.png"))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    fs = fd.FlightSet(a.csv)
    recs = bm.sweep(fs, a.seeds, columns=(COL,))
    bad = [r for r in recs if not (r["incremental_ok"] and r["decremental_ok"])]
    print(f"correctness: {len(recs) - len(bad)}/{len(recs)} instances match the exact MEB")
    sweep = {"n": bm.SWEEP_N, "runs": recs}
    json.dump(sweep, open(os.path.join(HERE, "results", f"sweep_{COL}.json"), "w"), indent=1, default=float)
    pf._style()
    pf.COLUMN_TITLE[COL] = "UTC time of day (positions recorded before the cut-off hour consent)"
    print("wrote", pf.plot_maps(fs, sweep, COL, a.out))
