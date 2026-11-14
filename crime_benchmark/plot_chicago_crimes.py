"""
Plot Chicago Crime Incidents and Minimum Enclosing Ball (MEB) on OpenStreetMap.

This script loads Chicago crime records, determines point consent based on a user-specified
column (e.g., ARREST, DOMESTIC, PRIMARY=THEFT), computes the MEB of the consenting points,
and plots consenting, non-consenting, and MEB boundary points onto OpenStreetMap tiles.
"""

import argparse
import csv
import math
import os
import random
import sys
import urllib.request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import db_consents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_consents
from db_consents import Oracle, decremental_distance_based, incremental_distance_based


def lat_lng_to_tile(lat, lng, zoom):
    """Convert latitude/longitude to OSM tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lat_lng(xtile, ytile, zoom):
    """Convert OSM tile coordinates back to latitude/longitude."""
    n = 2.0 ** zoom
    lng = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat = math.degrees(lat_rad)
    return lat, lng


def fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=11):
    """Fetch OpenStreetMap tiles covering the bounding box and stitch them together."""
    xtile_min, ytile_min = lat_lng_to_tile(lat_max, lng_min, zoom)
    xtile_max, ytile_max = lat_lng_to_tile(lat_min, lng_max, zoom)

    print(f"Fetching map tiles for zoom {zoom}: x=[{xtile_min}..{xtile_max}], y=[{ytile_min}..{ytile_max}]...")

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile_cache")
    os.makedirs(cache_dir, exist_ok=True)

    tile_size = 256
    width = (xtile_max - xtile_min + 1) * tile_size
    height = (ytile_max - ytile_min + 1) * tile_size

    stitched_img = Image.new("RGBA", (width, height))

    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            tile_file = os.path.join(cache_dir, f"light_{zoom}_{x}_{y}.png")

            if not os.path.exists(tile_file):
                url = f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{zoom}/{x}/{y}.png"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChicagoCrimeMEBPlotter/1.0"}
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as response:
                        with open(tile_file, "wb") as f:
                            f.write(response.read())
                except Exception as e:
                    print(f"  Failed to fetch tile {zoom}/{x}/{y}: {e}")
                    continue

            try:
                tile_img = Image.open(tile_file).convert("RGBA")
                x_pos = (x - xtile_min) * tile_size
                y_pos = (y - ytile_min) * tile_size
                stitched_img.paste(tile_img, (x_pos, y_pos))
            except Exception as e:
                print(f"  Failed to process tile {tile_file}: {e}")

    lat_top, lng_left = tile_to_lat_lng(xtile_min, ytile_min, zoom)
    lat_bottom, lng_right = tile_to_lat_lng(xtile_max + 1, ytile_max + 1, zoom)

    extent = [lng_left, lng_right, lat_bottom, lat_top]
    return stitched_img, extent


def parse_consent_spec(column_spec, default_value="true"):
    """
    Parses column specification into (column_name, expected_value).
    Examples:
        'ARREST' -> ('ARREST', 'true')
        'PRIMARY=THEFT' -> ('PRIMARY', 'THEFT')
        'DOMESTIC' -> ('DOMESTIC', 'true')
    """
    if "=" in column_spec:
        col, val = column_spec.split("=", 1)
        return col.strip(), val.strip()
    return column_spec.strip(), default_value.strip()


def find_column_key(headers, target_col):
    """Finds the actual CSV header key matching target_col (case-insensitive, ignores type suffix)."""
    target_clean = target_col.split(":")[0].strip().upper()
    for h in headers:
        clean_h = h.split(":")[0].strip().upper()
        if clean_h == target_clean or h.upper() == target_clean:
            return h
    return None


def load_crimes_with_consent(csv_path, max_points, consent_col, consent_val):
    """
    Loads crime records and assigns boolean consent based on the specified column/value.
    """
    points_with_consent = []
    total_read = 0

    print(f"Reading crime data from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        headers = reader.fieldnames or []

        col_key = find_column_key(headers, consent_col)
        if not col_key:
            print(f"Error: Column '{consent_col}' not found in CSV. Available columns:")
            for h in headers:
                print(f"  - {h}")
            sys.exit(1)

        print(f"Using column '{col_key}' for consent criteria (target value: '{consent_val}')...")

        is_bool_mode = consent_val.lower() in ("true", "false", "1", "0", "yes", "no")
        target_bool = consent_val.lower() in ("true", "1", "yes")

        for row in reader:
            total_read += 1
            try:
                lat_str = row.get("LATITUDE:DOUBLE", "")
                lng_str = row.get("LONGITUDE:DOUBLE", "")
                if not lat_str or not lng_str:
                    continue

                lat = float(lat_str)
                lng = float(lng_str)

                # Valid Chicago area bounds check
                if not (41.60 <= lat <= 42.10 and -87.95 <= lng <= -87.50):
                    continue

                raw_val = row.get(col_key, "").strip()

                if is_bool_mode:
                    consent = (raw_val.lower() in ("true", "1", "yes")) == target_bool
                else:
                    consent = (raw_val.upper() == consent_val.upper())

                points_with_consent.append(([lat, lng], consent))

                if len(points_with_consent) >= max_points * 3:
                    break

            except (ValueError, KeyError):
                continue

    if len(points_with_consent) > max_points:
        random.seed(42)
        points_with_consent = random.sample(points_with_consent, max_points)

    consenting_count = sum(1 for _, c in points_with_consent if c)
    print(f"Sampled {len(points_with_consent)} points from {total_read} rows.")
    print(f"Consenting: {consenting_count} / {len(points_with_consent)} "
          f"({100 * consenting_count / max(len(points_with_consent), 1):.1f}%)")

    return points_with_consent, col_key


def main():
    parser = argparse.ArgumentParser(
        description="Plot Chicago Crime MEB and (Non-)Consenting Points on OpenStreetMap"
    )
    parser.add_argument(
        "--algo",
        choices=["decremental", "incremental"],
        default="decremental",
        help="MEB algorithm to run: 'decremental' or 'incremental' (default: decremental)"
    )
    parser.add_argument(
        "--column",
        "--consent-column",
        type=str,
        default="ARREST",
        help="Column to determine consent (e.g. 'ARREST', 'DOMESTIC', 'PRIMARY=THEFT'). Default: ARREST"
    )
    parser.add_argument(
        "--consent-value",
        type=str,
        default="true",
        help="Expected value in column to consider consenting (default: 'true')"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=6000,
        help="Number of points to sample and plot (default: 6000)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to Crimes_Ordered.csv (default: Crimes_Ordered.csv in script directory)"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=11,
        help="Zoom level for OpenStreetMap tiles (default: 11)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG file path (default: chicago_crimes_meb_map.png)"
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if args.csv else os.path.join(base_dir, "Crimes_Ordered.csv")
    output_path = args.output if args.output else os.path.join(base_dir, "chicago_crimes_meb_map.png")

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    consent_col, consent_val = parse_consent_spec(args.column, args.consent_value)

    points_with_consent, actual_col_name = load_crimes_with_consent(
        csv_path=csv_path,
        max_points=args.n_points,
        consent_col=consent_col,
        consent_val=consent_val
    )

    consenting_pts = [pt for pt, c in points_with_consent if c]
    has_consenting = len(consenting_pts) > 0

    if has_consenting:
        print(f"Computing Minimum Enclosing Ball of consenting points ({args.algo.upper()})...")
        db_consents.USE_CPP_BACKEND = False
        oracle = Oracle()
        if args.algo == "incremental":
            center, radius = incremental_distance_based(points_with_consent, oracle)
        else:
            center, radius = decremental_distance_based(points_with_consent, oracle)

        oracle_calls = oracle.get_call_count()
        lat_c, lng_c = center[0], center[1]
        print(f"Computed MEB Center: Lat={lat_c:.5f}, Lng={lng_c:.5f}")
        print(f"MEB Radius (degrees): {radius:.5f}")
        print(f"Consent Requests (Oracle Calls): {oracle_calls}")
    else:
        print("Warning: No consenting points found. MEB will not be drawn.")
        lat_c, lng_c, radius = None, None, 0.0
        oracle_calls = 0

    # Separate points and find boundary support points
    consenting_lats = []
    consenting_lngs = []
    non_consenting_lats = []
    non_consenting_lngs = []
    boundary_lats = []
    boundary_lngs = []

    tolerance = 1e-4
    for (pt, consent) in points_with_consent:
        if consent:
            if has_consenting:
                dist = math.sqrt((pt[0] - lat_c) ** 2 + (pt[1] - lng_c) ** 2)
                if abs(dist - radius) <= tolerance:
                    boundary_lats.append(pt[0])
                    boundary_lngs.append(pt[1])
                else:
                    consenting_lats.append(pt[0])
                    consenting_lngs.append(pt[1])
            else:
                consenting_lats.append(pt[0])
                consenting_lngs.append(pt[1])
        else:
            non_consenting_lats.append(pt[0])
            non_consenting_lngs.append(pt[1])

    print(f"Identified {len(boundary_lats)} boundary support points.")

    # Calculate map bounding box
    all_lats = [pt[0] for pt, _ in points_with_consent]
    all_lngs = [pt[1] for pt, _ in points_with_consent]

    if has_consenting:
        plot_lat_min = min(min(all_lats), lat_c - radius)
        plot_lat_max = max(max(all_lats), lat_c + radius)
        plot_lng_min = min(min(all_lngs), lng_c - radius)
        plot_lng_max = max(max(all_lngs), lng_c + radius)
    else:
        plot_lat_min, plot_lat_max = min(all_lats), max(all_lats)
        plot_lng_min, plot_lng_max = min(all_lngs), max(all_lngs)

    lat_span = plot_lat_max - plot_lat_min
    lng_span = plot_lng_max - plot_lng_min

    pad_lat = max(lat_span * 0.06, 0.01)
    pad_lng = max(lng_span * 0.06, 0.01)

    lat_min, lat_max = plot_lat_min - pad_lat, plot_lat_max + pad_lat
    lng_min, lng_max = plot_lng_min - pad_lng, plot_lng_max + pad_lng

    map_img, extent = fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=args.zoom)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(map_img, extent=extent, origin="upper")

    clean_col = actual_col_name.split(":")[0]
    criteria_str = f"{clean_col} = {consent_val}"
    consenting_total = len(consenting_lats) + len(boundary_lats)

    # Non-consenting points (grey)
    ax.scatter(
        non_consenting_lngs,
        non_consenting_lats,
        c="#8e8e93",
        s=2.5,
        alpha=0.35,
        zorder=2,
        label=f"Non-Consenting ({clean_col} \u2260 {consent_val}, n={len(non_consenting_lats)})"
    )

    # Consenting interior points (emerald green)
    ax.scatter(
        consenting_lngs,
        consenting_lats,
        c="#10b981",
        s=3.5,
        alpha=0.6,
        zorder=3,
        label=f"Consenting ({criteria_str}, n={len(consenting_lats)})"
    )

    # Boundary points (vivid blue)
    if boundary_lats:
        ax.scatter(
            boundary_lngs,
            boundary_lats,
            c="#007aff",
            edgecolors="#ffffff",
            linewidth=1.2,
            s=55,
            alpha=0.95,
            zorder=6,
            label=f"MEB Boundary Points (Support Set, n={len(boundary_lats)})"
        )

    # MEB footprint and center
    if has_consenting:
        circle = plt.Circle(
            (lng_c, lat_c),
            radius,
            color="#ff3b30",
            fill=True,
            alpha=0.08,
            linewidth=2,
            linestyle="--",
            label=f"MEB Footprint (Radius: {radius:.4f}\u00b0, Oracle Calls: {oracle_calls})"
        )
        ax.add_patch(circle)
        circle_border = plt.Circle(
            (lng_c, lat_c),
            radius,
            color="#ff3b30",
            fill=False,
            linewidth=2,
            linestyle="-"
        )
        ax.add_patch(circle_border)

        ax.plot(
            lng_c,
            lat_c,
            marker="P",
            color="#ff3b30",
            markersize=10,
            zorder=7,
            label=f"MEB Center ({lat_c:.4f}, {lng_c:.4f})"
        )

    title_line1 = f"Chicago Crimes MEB ({args.algo.upper()}) \u2014 Criteria: {criteria_str}"
    title_line2 = (
        f"Total Points: {len(points_with_consent)}  |  "
        f"Consenting: {consenting_total} ({100*consenting_total/max(len(points_with_consent),1):.1f}%)  |  "
        f"Consent Requests (Oracle Calls): {oracle_calls}"
    )

    ax.set_title(
        f"{title_line1}\n{title_line2}",
        fontsize=12,
        fontweight="bold",
        color="#1c1c1e",
        pad=15
    )
    ax.set_xlabel("Longitude", fontsize=9, color="#3a3a3c")
    ax.set_ylabel("Latitude", fontsize=9, color="#3a3a3c")
    ax.set_xlim(lng_min, lng_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e5ea", linestyle=":", alpha=0.6)

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#3a3a3c", labelsize=8)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#e5e5ea",
        fontsize=8.5
    )
    for text in legend.get_texts():
        text.set_color("black")

    plt.tight_layout()
    plt.savefig(output_path, facecolor="#ffffff", edgecolor="none", bbox_inches="tight")
    print(f"Map successfully rendered and saved to: {output_path}")


if __name__ == "__main__":
    main()
