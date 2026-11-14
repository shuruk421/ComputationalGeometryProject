"""
Regenerate the whole flight consent-MEB study from flight.csv.

    python flight_benchmark/run_flight_suite.py --csv C:/path/to/flight.csv

Steps (each also runnable on its own):
  1. flight_data.py          one streaming pass over the 9.3 GB CSV keeping every
                             60th row (~850k rows, ~20 s), cached as
                             cache/flight_sample_*.npz (gitignored); later runs
                             load the cache.
  2. consent_analysis.py     candidate consent columns -> results/consent_candidates.json
  3. flight_meb_benchmark.py incremental vs decremental sweep, n-scaling and the
                             km-units check -> results/{sweep,scaling,correctness}.json
                             (~10 min of pure-Python compute at the defaults)
  4. plot_flight_figures.py  PNG figures next to this file (gitignored)

Design decisions
----------------
Data.  The file is a single aircraft (ICAO24 471f63, a Wizz Air A320) from
2016-03-09 to 2019-11-27, ~51M rows at one row per second, sorted by time.
Reading only the head would cover a few flights from March 2016, so the
sampler streams the whole file.  Cleaning drops stale (repeated) positions,
garbled position jumps (implied speed > 350 m/s to every neighbour),
implausible altitudes/velocities, and collapses exact duplicate (LAT, LON)
pairs to their earliest observation, because db_consents.Oracle caches by
coordinate tuple (identical coordinates would be ONE consent request answered
with whichever label it saw first).  Oracle calls therefore count distinct
locations.

Consent.  Per-aircraft (ICAO24) and per-operator (CALLSIGN prefix) consent are
degenerate here (100% and 99.5%).  The entity that remains is the flight leg.
Primary column: opt-in date - every leg flown on/after date T shares its
positions; T is swept by quantile from 5% to 95% consenting.  Contrast: random
legs - each leg consents with probability p (hashed, nested in p), i.e. the
same entity granularity without the spatial/temporal structure.

Geometry.  2D.  Horizontal extent is ~3,000 km while altitude spans 0-13 km;
with altitude at true scale the exact 3D MEB of all ~800k positions has the
same radius as the 2D MEB to within 0.1 m (results/consent_candidates.json,
"geometry"), so a third coordinate would only slow Welzl down; exaggerating
altitude would make the metric arbitrary.  Positions are projected with an
azimuthal-equidistant projection centred on the data's spherical centroid
(the extent, 32N-60N, is continental: an equirectangular projection at the
centroid latitude would scale east-west distances by +24% at 32N and -27% at
60N); its MEB radius is within 0.1% of the exact minimal spherical cap.  The algorithms receive coordinates in units of
1000 km because db_consents uses an absolute 1e-10 tolerance on squared
distances, which breaks decremental_distance_based at km scale.
"""

import argparse
import json
import os
import time

import flight_data as fd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=fd.DEFAULT_CSV, help="path to flight.csv (~9.3 GB)")
    ap.add_argument("--seeds", type=int, default=6, help="seeds per benchmark point (>= 5)")
    ap.add_argument("--rebuild-sample", action="store_true", help="re-stream the CSV")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="reuse results/*.json, only redo analysis + figures")
    ap.add_argument("--quick", action="store_true", help="small-n smoke test (overwrites results)")
    ap.add_argument("--zoom", type=int, default=5, help="basemap tile zoom level")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("== 1/4 whole-file sample")
    fs = fd.FlightSet(args.csv, rebuild=args.rebuild_sample)
    print(json.dumps(fs.describe(), indent=1))

    print("== 2/4 consent-column analysis")
    import consent_analysis
    consent_analysis.run(args.csv, verbose=True)

    if not args.skip_benchmark:
        print("== 3/4 benchmark")
        import flight_meb_benchmark
        flight_meb_benchmark.run(args.csv, seeds=args.seeds, quick=args.quick)
    else:
        print("== 3/4 benchmark skipped (using existing results/*.json)")

    print("== 4/4 figures")
    import plot_flight_figures
    plot_flight_figures.run(args.csv, zoom=args.zoom)
    print(f"Done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
