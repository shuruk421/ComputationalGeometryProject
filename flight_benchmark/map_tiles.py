"""
Basemap tiles in Web-Mercator coordinates.

The earlier plot_flight_meb.py drew Mercator tiles with a linear latitude
extent, which misregisters points against the basemap by up to ~1 degree over
a 30-degree latitude span.  Here everything is drawn in Mercator space:
x = longitude (degrees) and y = merc_y(latitude) in "Mercator degrees", so
x and y share a scale and ``aspect='equal'`` is conformal.

Tiles: the CARTO endpoint used previously now returns an "API KEY REQUIRED"
placeholder, so tiles come from the standard OpenStreetMap tile server (a few
dozen tiles, fetched once and cached in flight_benchmark/tile_cache/, which is
gitignored) and are rendered desaturated and faded so the data stays on top.
Map data (c) OpenStreetMap contributors.  If tiles cannot be downloaded the
plots still render, on a plain background.
"""

import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "tile_cache")
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "MEB-benchmark-plotter/1.0 (research figure; tiles cached locally)"
ATTRIBUTION = "Basemap (c) OpenStreetMap contributors"


def merc_y(lat):
    """Latitude (deg) -> Web-Mercator y expressed in degrees."""
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.0511, 85.0511)
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def inv_merc_y(y):
    return np.degrees(2 * np.arctan(np.exp(np.radians(y))) - np.pi / 2)


def lat_lng_to_tile(lat, lng, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lng(x, y, zoom):
    n = 2.0 ** zoom
    lng = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lng


def _tile_file(z, x, y):
    return os.path.join(TILE_DIR, f"osm_{z}_{x}_{y}.png")


def _download(job):
    z, x, y = job
    path = _tile_file(z, x, y)
    if os.path.exists(path):
        return True
    try:
        req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y),
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(path, "wb") as fh:
            fh.write(data)
        return True
    except Exception:
        return False


def basemap(lat_min, lat_max, lng_min, lng_max, zoom=5, fade=0.42):
    """
    Stitch the tiles covering the box, desaturated and faded towards white
    (``fade`` = fraction of the original contrast kept).  Returns
    (RGB image, extent) with extent = (lon_left, lon_right, mercY_bottom,
    mercY_top) for imshow.
    """
    os.makedirs(TILE_DIR, exist_ok=True)
    x0, y0 = lat_lng_to_tile(lat_max, lng_min, zoom)
    x1, y1 = lat_lng_to_tile(lat_min, lng_max, zoom)
    jobs = [(zoom, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    with ThreadPoolExecutor(max_workers=2) as ex:  # be polite to the tile server
        list(ex.map(_download, jobs))
    ts = 256
    img = Image.new("RGB", ((x1 - x0 + 1) * ts, (y1 - y0 + 1) * ts), (245, 245, 243))
    for (z, x, y) in jobs:
        path = _tile_file(z, x, y)
        if os.path.exists(path):
            try:
                img.paste(Image.open(path).convert("RGB"), ((x - x0) * ts, (y - y0) * ts))
            except Exception:
                pass
    g = np.asarray(ImageOps.grayscale(img), dtype=np.float64)
    g = 255.0 - (255.0 - g) * fade
    img = Image.fromarray(np.dstack([g, g, g * 0.995]).clip(0, 255).astype(np.uint8))
    lat_top, lng_left = tile_to_lat_lng(x0, y0, zoom)
    lat_bot, lng_right = tile_to_lat_lng(x1 + 1, y1 + 1, zoom)
    return img, [lng_left, lng_right, float(merc_y(lat_bot)), float(merc_y(lat_top))]
