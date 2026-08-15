"""
Plot Amsterdam Airbnb Minimum Enclosing Ball (MEB) with OpenStreetMap Background.

This script loads the Amsterdam Airbnb dataset, simulates host consent,
computes the MEB of consenting hosts in 3D (latitude, longitude, scaled price),
and plots the result overlaid on OpenStreetMap background map tiles.
"""

import os
import sys
import csv
import math
import random
import argparse
import urllib.request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import db_consents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_consents
from db_consents import Oracle, decremental_distance_based

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

def fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=13):
    """Fetch OpenStreetMap tiles covering the bounding box and stitch them together."""
    xtile_min, ytile_min = lat_lng_to_tile(lat_max, lng_min, zoom)
    xtile_max, ytile_max = lat_lng_to_tile(lat_min, lng_max, zoom)
    
    print(f"Fetching tiles for x: [{xtile_min}..{xtile_max}], y: [{ytile_min}..{ytile_max}]")
    
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    tile_size = 256
    width = (xtile_max - xtile_min + 1) * tile_size
    height = (ytile_max - ytile_min + 1) * tile_size
    
    # Create background image
    stitched_img = Image.new("RGBA", (width, height))
    
    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            tile_file = os.path.join(cache_dir, f"light_{zoom}_{x}_{y}.png")
            
            # Download tile if not cached
            if not os.path.exists(tile_file):
                # Using CartoDB Positron tiles for a clean light look
                url = f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{zoom}/{x}/{y}.png"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AirbnbMEBPlotter/1.0"}
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as response:
                        with open(tile_file, "wb") as f:
                            f.write(response.read())
                except Exception as e:
                    print(f"Failed to fetch tile {zoom}/{x}/{y}: {e}")
                    # Fallback to pure light gray if tile download fails
                    continue
            
            try:
                tile_img = Image.open(tile_file).convert("RGBA")
                x_pos = (x - xtile_min) * tile_size
                y_pos = (y - ytile_min) * tile_size
                stitched_img.paste(tile_img, (x_pos, y_pos))
            except Exception as e:
                print(f"Failed to process tile {tile_file}: {e}")
                
    # Calculate bounds of stitched image
    lat_top, lng_left = tile_to_lat_lng(xtile_min, ytile_min, zoom)
    lat_bottom, lng_right = tile_to_lat_lng(xtile_max + 1, ytile_max + 1, zoom)
    
    extent = [lng_left, lng_right, lat_bottom, lat_top]
    return stitched_img, extent

def main():
    parser = argparse.ArgumentParser(
        description="Plot Amsterdam Airbnb Minimum Enclosing Ball (MEB) with OpenStreetMap Background"
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=10,
        help="Minimum number of reviews required for a host to be considered consenting (default: 10)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to Inside Airbnb listings CSV (default: listings_amsterdam.csv in script directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the generated PNG plot (default: amsterdam_meb_map_osm.png)"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=13,
        help="Zoom level for OpenStreetMap tiles (default: 13)"
    )
    parser.add_argument(
        "--price-scale",
        type=float,
        default=1.0,
        help="Price scaling factor relative to geographic range (default: 1.0)"
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if args.csv else os.path.join(base_dir, "listings_amsterdam.csv")
    output_img = args.output if args.output else os.path.join(base_dir, "amsterdam_meb_map_osm.png")
    min_reviews_threshold = args.min_reviews

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please provide a valid CSV path.")
        sys.exit(1)

    print(f"Loading Amsterdam listings from {csv_path}...")
    listings = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lng = float(row["longitude"])
                price = float(row["price"])
                n_reviews = int(row["number_of_reviews"])
                if price > 0:
                    listings.append((lat, lng, price, n_reviews))
            except (ValueError, KeyError):
                continue

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # 2. Scale coordinates and apply static consent criteria
    lats = [l[0] for l in listings]
    lngs = [l[1] for l in listings]
    prices = [l[2] for l in listings]

    lat_range = max(lats) - min(lats)
    lng_range = max(lngs) - min(lngs)
    geo_range = max(lat_range, lng_range)
    price_range = max(prices) - min(prices)

    scale = (geo_range / price_range) * args.price_scale

    # Static review count threshold for consent
    points_with_consent = []
    for lat, lng, price, n_reviews in listings:
        point = [lat, lng, price * scale]
        consent = (n_reviews >= min_reviews_threshold)
        points_with_consent.append((point, consent))

    consenting_count = sum(1 for _, c in points_with_consent if c)
    print(f"Consent criteria: number_of_reviews >= {min_reviews_threshold}")
    print(f"Consenting listings: {consenting_count} / {len(listings)} ({100*consenting_count/len(listings):.1f}%)")

    # 3. Compute MEB using pure Python backend (highly stable on Windows/WSL)
    print("Computing Minimum Enclosing Ball...")
    db_consents.USE_CPP_BACKEND = False
    oracle = Oracle()
    center, radius = decremental_distance_based(points_with_consent, oracle)

    lat_c, lng_c, scaled_price_c = center
    print(f"Computed MEB Center: Lat={lat_c:.5f}, Lng={lng_c:.5f}, Price={scaled_price_c/scale:.2f}")
    print(f"MEB Radius (degrees): {radius:.5f}")

    # 4. Fetch and stitch OpenStreetMap tiles
    circle_lat_min = lat_c - radius
    circle_lat_max = lat_c + radius
    circle_lng_min = lng_c - radius
    circle_lng_max = lng_c + radius

    plot_lat_min = min(min(lats), circle_lat_min)
    plot_lat_max = max(max(lats), circle_lat_max)
    plot_lng_min = min(min(lngs), circle_lng_min)
    plot_lng_max = max(max(lngs), circle_lng_max)

    lat_span = plot_lat_max - plot_lat_min
    lng_span = plot_lng_max - plot_lng_min

    # Add 5% padding around the bounding box
    pad_lat = max(lat_span * 0.05, 0.005)
    pad_lng = max(lng_span * 0.05, 0.005)

    lat_min, lat_max = plot_lat_min - pad_lat, plot_lat_max + pad_lat
    lng_min, lng_max = plot_lng_min - pad_lng, plot_lng_max + pad_lng

    print(f"Fetching map background from OpenStreetMap (zoom={args.zoom})...")
    map_img, extent = fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=args.zoom)

    # 5. Plot
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)

    # Draw OpenStreetMap background
    ax.imshow(map_img, extent=extent, origin="upper")

    # Filter consenting vs non-consenting, and identify MEB boundary points
    consenting_lats = []
    consenting_lngs = []
    non_consenting_lats = []
    non_consenting_lngs = []
    boundary_lats = []
    boundary_lngs = []

    tolerance = 1e-4
    for (pt, consent) in points_with_consent:
        if consent:
            dist = math.sqrt(sum((pt[i] - center[i]) ** 2 for i in range(3)))
            if abs(dist - radius) <= tolerance:
                boundary_lats.append(pt[0])
                boundary_lngs.append(pt[1])
            else:
                consenting_lats.append(pt[0])
                consenting_lngs.append(pt[1])
        else:
            non_consenting_lats.append(pt[0])
            non_consenting_lngs.append(pt[1])

    print(f"Identified {len(boundary_lats)} MEB boundary (support) points.")

    # Plot non-consenting listings (grey)
    ax.scatter(
        non_consenting_lngs,
        non_consenting_lats,
        c='#8e8e93',
        s=1.5,
        alpha=0.3,
        zorder=2,
        label=f'No Consent / Inactive (Reviews < {min_reviews_threshold})'
    )

    # Plot consenting interior listings (vibrant emerald green)
    ax.scatter(
        consenting_lngs,
        consenting_lats,
        c='#10b981',
        s=3,
        alpha=0.6,
        zorder=3,
        label=f'Consenting Host (Reviews \u2265 {min_reviews_threshold})'
    )

    # Plot boundary points (vivid blue with white outline for high visibility)
    ax.scatter(
        boundary_lngs,
        boundary_lats,
        c='#007aff',
        edgecolors='#ffffff',
        linewidth=1.2,
        s=55,
        alpha=0.95,
        zorder=6,
        label=f'Boundary Points (Support Set, n={len(boundary_lats)})'
    )

    # Draw MEB projected circle (using a red dashed outline with a transparent fill)
    circle = plt.Circle(
        (lng_c, lat_c),
        radius,
        color='#ff3b30',
        fill=True,
        alpha=0.08,
        linewidth=2,
        linestyle='--',
        label='MEB Footprint (Spatial Projection)'
    )
    ax.add_patch(circle)
    circle_border = plt.Circle(
        (lng_c, lat_c),
        radius,
        color='#ff3b30',
        fill=False,
        linewidth=2,
        linestyle='-'
    )
    ax.add_patch(circle_border)

    # Draw MEB center
    ax.plot(
        lng_c,
        lat_c,
        marker='P',
        color='#ff3b30',
        markersize=9,
        label=f'MEB Center (Avg Price: \u20ac{scaled_price_c/scale:.0f})'
    )

    # Layout, title, and subtitle
    ax.set_title(
        f"Amsterdam Airbnb MEB \u2014 Consent Criteria: Reviews \u2265 {min_reviews_threshold}",
        fontsize=13,
        fontweight='bold',
        color='#1c1c1e',
        pad=15
    )
    ax.set_xlabel("Longitude", fontsize=9, color='#3a3a3c')
    ax.set_ylabel("Latitude", fontsize=9, color='#3a3a3c')
    ax.set_xlim(lng_min, lng_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='#e5e5ea', linestyle=':', alpha=0.6)

    # Set face colors for clean aesthetic
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#3a3a3c', labelsize=8)

    # Legend with criteria clearly documented
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e5e5ea', fontsize=8.5)
    for text in legend.get_texts():
        text.set_color('black')

    plt.tight_layout()
    plt.savefig(output_img, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
    print(f"Map successfully generated and saved to {output_img}")

if __name__ == "__main__":
    main()
