"""
Plot Aircraft Trajectories and Minimum Enclosing Ball (MEB) on OpenStreetMap.

This script loads ADS-B flight trajectory telemetry from flight.csv,
determines point consent based on flags such as ALERT or SPI (or custom criteria),
computes the MEB of the consenting flight positions using decremental or incremental algorithms,
and plots consenting, non-consenting, and boundary support points onto OpenStreetMap tiles.
"""

import argparse
import csv
import math
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def _download_single_tile(args):
    """Worker function to download a single tile."""
    zoom, x, y, tile_file = args
    if not os.path.exists(tile_file):
        url = f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{zoom}/{x}/{y}.png"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FlightMEBPlotter/1.0"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                with open(tile_file, "wb") as f:
                    f.write(response.read())
        except Exception as e:
            pass


def fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=6):
    """Fetch OpenStreetMap tiles covering the bounding box in parallel and stitch them together."""
    # Constrain coordinates to valid Mercator range
    lat_min = max(min(lat_min, 85.0), -85.0)
    lat_max = max(min(lat_max, 85.0), -85.0)
    lng_min = max(min(lng_min, 179.9), -179.9)
    lng_max = max(min(lng_max, 179.9), -179.9)

    xtile_min, ytile_min = lat_lng_to_tile(lat_max, lng_min, zoom)
    xtile_max, ytile_max = lat_lng_to_tile(lat_min, lng_max, zoom)

    print(f"Fetching map tiles for zoom {zoom}: x=[{xtile_min}..{xtile_max}], y=[{ytile_min}..{ytile_max}]...", flush=True)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Download missing tiles concurrently
    download_tasks = []
    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            tile_file = os.path.join(cache_dir, f"light_{zoom}_{x}_{y}.png")
            download_tasks.append((zoom, x, y, tile_file))

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_download_single_tile, download_tasks))

    tile_size = 256
    width = (xtile_max - xtile_min + 1) * tile_size
    height = (ytile_max - ytile_min + 1) * tile_size

    stitched_img = Image.new("RGBA", (width, height), (240, 240, 240, 255))

    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            tile_file = os.path.join(cache_dir, f"light_{zoom}_{x}_{y}.png")
            if os.path.exists(tile_file):
                try:
                    tile_img = Image.open(tile_file).convert("RGBA")
                    x_pos = (x - xtile_min) * tile_size
                    y_pos = (y - ytile_min) * tile_size
                    stitched_img.paste(tile_img, (x_pos, y_pos))
                except Exception:
                    pass

    lat_top, lng_left = tile_to_lat_lng(xtile_min, ytile_min, zoom)
    lat_bottom, lng_right = tile_to_lat_lng(xtile_max + 1, ytile_max + 1, zoom)

    extent = [lng_left, lng_right, lat_bottom, lat_top]
    return stitched_img, extent


def load_flight_data(csv_path, max_points, consent_mode, scan_limit=300000):
    """
    Loads flight telemetry points and evaluates consent flag based on consent_mode.
    Supported consent_mode values:
      - 'ALERT' (ALERT == 1)
      - 'SPI' (SPI == 1)
      - 'ALERT_OR_SPI' (ALERT == 1 or SPI == 1)
      - Custom: 'COLUMN=VALUE' or 'COLUMN'
    """
    all_points = []
    total_read = 0
    mode_upper = consent_mode.strip().upper()

    print(f"Reading flight trajectory data from {csv_path} (scanning up to {scan_limit} rows)...", flush=True)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            total_read += 1
            try:
                lat_str = row.get("LAT", "")
                lon_str = row.get("LON", "")
                if not lat_str or not lon_str:
                    continue

                lat = float(lat_str)
                lon = float(lon_str)

                # Geographic bounds filter (valid range)
                if not (-85.0 <= lat <= 85.0 and -180.0 <= lon <= 180.0):
                    continue

                alert_val = row.get("ALERT", "0").strip()
                spi_val = row.get("SPI", "0").strip()

                if mode_upper == "ALERT":
                    consent = (alert_val == "1" or alert_val.lower() == "true")
                elif mode_upper == "SPI":
                    consent = (spi_val == "1" or spi_val.lower() == "true")
                elif mode_upper in ("ALERT_OR_SPI", "SPI_OR_ALERT"):
                    consent = (alert_val == "1" or spi_val == "1" or alert_val.lower() == "true" or spi_val.lower() == "true")
                elif "=" in consent_mode:
                    col, target_val = consent_mode.split("=", 1)
                    raw_val = row.get(col.strip().upper(), "").strip()
                    consent = (raw_val.upper() == target_val.strip().upper())
                else:
                    raw_val = row.get(consent_mode.strip().upper(), "0").strip()
                    consent = (raw_val in ("1", "true", "True", "YES", "yes"))

                all_points.append(([lat, lon], consent))

                if total_read >= scan_limit:
                    break

            except (ValueError, KeyError):
                continue

    if len(all_points) > max_points:
        random.seed(42)
        sampled_points = random.sample(all_points, max_points)
    else:
        sampled_points = all_points

    consenting_count = sum(1 for _, c in sampled_points if c)
    print(f"Sampled {len(sampled_points)} points from {total_read} rows.", flush=True)
    print(f"Consent criteria: {consent_mode}", flush=True)
    print(f"Consenting positions: {consenting_count} / {len(sampled_points)} "
          f"({100 * consenting_count / max(len(sampled_points), 1):.2f}%)", flush=True)

    return sampled_points


def main():
    parser = argparse.ArgumentParser(
        description="Plot Flight MEB and (Non-)Consenting Points on OpenStreetMap"
    )
    parser.add_argument(
        "--consent",
        "--column",
        type=str,
        default="ALERT",
        help="Consent criteria flag: 'ALERT', 'SPI', 'ALERT_OR_SPI', or 'COLUMN=VALUE' (default: ALERT)"
    )
    parser.add_argument(
        "--algo",
        choices=["decremental", "incremental"],
        default="decremental",
        help="MEB algorithm to run: 'decremental' or 'incremental' (default: decremental)"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=25000,
        help="Number of flight data points to sample and plot (default: 25000)"
    )
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=250000,
        help="Number of raw CSV rows to scan for sampling (default: 250000)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to flight.csv (default: flight.csv in script directory)"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=6,
        help="Zoom level for OpenStreetMap tiles (default: 6 for European airspace)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG file path (default: flight_meb_map_osm.png)"
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if args.csv else os.path.join(base_dir, "flight.csv")
    output_path = args.output if args.output else os.path.join(base_dir, "flight_meb_map_osm.png")

    if not os.path.exists(csv_path):
        print(f"Error: Flight dataset not found at {csv_path}", flush=True)
        sys.exit(1)

    points_with_consent = load_flight_data(
        csv_path=csv_path,
        max_points=args.n_points,
        consent_mode=args.consent,
        scan_limit=args.scan_rows
    )

    consenting_pts = [pt for pt, c in points_with_consent if c]
    has_consenting = len(consenting_pts) > 0

    if has_consenting:
        print(f"Computing Minimum Enclosing Ball of consenting flight points ({args.algo.upper()})...", flush=True)
        db_consents.USE_CPP_BACKEND = False
        oracle = Oracle()
        if args.algo == "incremental":
            center, radius = incremental_distance_based(points_with_consent, oracle)
        else:
            center, radius = decremental_distance_based(points_with_consent, oracle)

        oracle_calls = oracle.get_call_count()
        lat_c, lng_c = center[0], center[1]
        print(f"Computed MEB Center: Lat={lat_c:.5f}, Lon={lng_c:.5f}", flush=True)
        print(f"MEB Radius (degrees): {radius:.5f}", flush=True)
        print(f"Consent Requests (Oracle Calls): {oracle_calls}", flush=True)
    else:
        print("Warning: No consenting points found in the sample. MEB will not be drawn.", flush=True)
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

    print(f"Identified {len(boundary_lats)} MEB boundary support points.", flush=True)

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

    pad_lat = max(lat_span * 0.06, 0.05)
    pad_lng = max(lng_span * 0.06, 0.05)

    lat_min, lat_max = plot_lat_min - pad_lat, plot_lat_max + pad_lat
    lng_min, lng_max = plot_lng_min - pad_lng, plot_lng_max + pad_lng

    map_img, extent = fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=args.zoom)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(map_img, extent=extent, origin="upper")

    criteria_str = f"Consent: {args.consent}"
    consenting_total = len(consenting_lats) + len(boundary_lats)

    # Non-consenting points (grey)
    ax.scatter(
        non_consenting_lngs,
        non_consenting_lats,
        c="#8e8e93",
        s=2,
        alpha=0.3,
        zorder=2,
        label=f"Regular Flights (No {args.consent}, n={len(non_consenting_lats)})"
    )

    # Consenting interior points (emerald green)
    ax.scatter(
        consenting_lngs,
        consenting_lats,
        c="#10b981",
        s=5.0,
        alpha=0.75,
        zorder=3,
        label=f"Consenting ({criteria_str}, n={len(consenting_lats)})"
    )

    # Boundary support points (vivid blue)
    if boundary_lats:
        ax.scatter(
            boundary_lngs,
            boundary_lats,
            c="#007aff",
            edgecolors="#ffffff",
            linewidth=1.2,
            s=65,
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

    title_line1 = f"Flight Telemetry MEB ({args.algo.upper()}) \u2014 Criteria: {criteria_str}"
    title_line2 = (
        f"Total Points: {len(points_with_consent)}  |  "
        f"Consenting: {consenting_total} ({100*consenting_total/max(len(points_with_consent),1):.2f}%)  |  "
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
    print(f"Map successfully rendered and saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
