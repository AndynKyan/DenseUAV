"""Georeference of the ortho ref map (GeoTIFF tags stored in the PNG EXIF) and region reading.

Tags used (WGS84, north-up, no rotation):
  33922 ModelTiepoint = (i, j, k, lon0, lat0, z)  -> pixel (i, j) maps to (lon0, lat0)
  33550 ModelPixelScale = (sx, sy, sz)  degrees per pixel
Pixel convention: (x, y) = (column, row); the tie point refers to the pixel corner (PixelIsArea),
so the centre of pixel (c, r) is at x = c + 0.5. All functions here use continuous coordinates.

The PNG is ~410 Mpx and not tiled, so it is decoded ONCE into a uint8 RGBA memmap cache
(<out_root>/cache/ortho_rgba.npy) and regions are read from that.

Run `python -m data_prep.georef` for the round-trip self-test.
"""
import math
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

EARTH_R = 6371008.8  # mean Earth radius (m) for our haversine


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Works with scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


class Georef:
    def __init__(self, tie, scale, width, height):
        i, j, _, lon0, lat0, _ = tie
        self.sx, self.sy = scale[0], scale[1]
        # lon/lat of pixel coordinate (0, 0)
        self.lon0 = lon0 - i * self.sx
        self.lat0 = lat0 + j * self.sy
        self.width, self.height = width, height
        lat_c = self.lat0 - self.sy * height / 2
        # metres per pixel (local, at map centre)
        self.mpp_x = haversine_m(lat_c, self.lon0, lat_c, self.lon0 + self.sx)
        self.mpp_y = haversine_m(lat_c, self.lon0, lat_c - self.sy, self.lon0)
        self.gsd = (self.mpp_x + self.mpp_y) / 2

    @classmethod
    def from_png(cls, path):
        im = Image.open(path)
        ex = im.getexif()
        tie, scale = ex.get(33922), ex.get(33550)
        if tie is None or scale is None:
            raise ValueError("No GeoTIFF tags 33922/33550 in %s" % path)
        if ex.get(34737) is not None and "WGS 84" not in str(ex.get(34737)):
            raise ValueError("Unexpected CRS: %s" % ex.get(34737))
        return cls(tie, scale, im.size[0], im.size[1])

    def px_to_latlon(self, x, y):
        return self.lat0 - np.asarray(y) * self.sy, self.lon0 + np.asarray(x) * self.sx

    def latlon_to_px(self, lat, lon):
        return (np.asarray(lon) - self.lon0) / self.sx, (self.lat0 - np.asarray(lat)) / self.sy

    def m_to_px(self, metres):
        return metres / self.gsd


class Ortho:
    """Ortho map pixels + georef. read_region pads outside the map with alpha=0."""

    def __init__(self, png_path, cache_dir):
        self.geo = Georef.from_png(png_path)
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, "ortho_rgba.npy")
        stamp = os.path.join(cache_dir, "ortho_rgba.src")
        src_id = "%s|%d|%d" % (png_path, os.path.getsize(png_path), int(os.path.getmtime(png_path)))
        if not (os.path.exists(cache) and os.path.exists(stamp) and open(stamp).read() == src_id):
            print("[georef] decoding %s once into %s (slow, ~1.6 GB)..." % (png_path, cache))
            im = Image.open(png_path).convert("RGBA")
            arr = np.asarray(im)
            mm = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=arr.shape)
            mm[:] = arr
            mm.flush()
            del mm, arr, im
            open(stamp, "w").write(src_id)
        self.arr = np.load(cache, mmap_mode="r")  # (H, W, 4) uint8
        assert self.arr.shape[:2] == (self.geo.height, self.geo.width)

    def read_region(self, x0, y0, w, h):
        """Integer pixel box [x0, x0+w) x [y0, y0+h) as RGBA uint8, zero-padded outside."""
        out = np.zeros((h, w, 4), np.uint8)
        H, W = self.arr.shape[:2]
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        if xb > xa and yb > ya:
            out[ya - y0:yb - y0, xa - x0:xb - x0] = self.arr[ya:yb, xa:xb]
        return out

    def window(self, cx, cy, half_px, factor=1):
        """Square window of half-size half_px around (cx, cy), downsampled by an integer factor.
        Returns an OrthoWindow; crops taken from it use full-resolution pixel coordinates."""
        x0, y0 = int(math.floor(cx - half_px)), int(math.floor(cy - half_px))
        size = int(math.ceil(2 * half_px)) + 2
        size = int(math.ceil(size / factor)) * factor
        reg = Image.fromarray(self.read_region(x0, y0, size, size), "RGBA")
        if factor > 1:
            reg = reg.reduce(factor)
        return OrthoWindow(reg, x0, y0, factor)

    def crop(self, cx, cy, side_px, angle_deg=0.0, out_size=None):
        """North-up square of side `side_px` centred at continuous pixel (cx, cy).

        angle_deg != 0 rotates the CONTENT counter-clockwise (PIL convention) before the
        centre crop, i.e. simulates a camera rotated by -angle_deg.
        Returns (RGB PIL image, nodata_fraction).
        """
        half = side_px / 2.0 * (math.sqrt(2) if angle_deg else 1.0) + 2
        return self.window(cx, cy, half).crop(cx, cy, side_px, angle_deg, out_size)


class OrthoWindow:
    def __init__(self, img, x0, y0, factor):
        self.img, self.x0, self.y0, self.f = img, x0, y0, factor

    def crop(self, cx, cy, side_px, angle_deg=0.0, out_size=None):
        """Same contract as Ortho.crop; (cx, cy, side_px) in full-resolution ortho pixels."""
        lcx, lcy, r = (cx - self.x0) / self.f, (cy - self.y0) / self.f, side_px / 2.0 / self.f
        reg = self.img
        if angle_deg:
            reg = reg.rotate(angle_deg, resample=Image.BICUBIC, center=(lcx, lcy))
        box = (lcx - r, lcy - r, lcx + r, lcy + r)
        out = out_size or int(round(2 * r))
        tile = reg.transform((out, out), Image.EXTENT, box, resample=Image.BICUBIC)
        nodata = float((np.asarray(tile)[..., 3] < 128).mean())
        return tile.convert("RGB"), nodata

    def thumbnail(self, max_side=2000):
        f = max(1, int(math.ceil(max(self.arr.shape[:2]) / max_side)))
        th = np.asarray(self.arr[::f, ::f])
        return Image.fromarray(th, "RGBA"), f


def _selftest():
    from .common import load_cfg
    cfg = load_cfg()
    g = Georef.from_png(cfg["paths"]["ortho_png"])
    print("size %dx%d  gsd x=%.4f y=%.4f m/px" % (g.width, g.height, g.mpp_x, g.mpp_y))
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(0, g.width, 10000), rng.uniform(0, g.height, 10000)
    lat, lon = g.px_to_latlon(xs, ys)
    x2, y2 = g.latlon_to_px(lat, lon)
    err = np.hypot(x2 - xs, y2 - ys)
    print("round-trip px->latlon->px: max err %.2e px (must be < 0.5)" % err.max())
    assert err.max() < 0.5
    d = haversine_m(10.0, 106.0, 10.0, 106.001)
    print("haversine 0.001 deg lon at lat 10: %.2f m (expect ~109.5)" % d)
    print("OK")


if __name__ == "__main__":
    _selftest()
