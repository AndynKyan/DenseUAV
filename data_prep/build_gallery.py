"""Multi-scale gallery over the WHOLE ortho (M3).

For every F in gallery.footprint_set_m: north-up square tiles of side F, centres on an axis-aligned
pixel grid with stride stride_frac * F, resized to gallery.out_size. Tiles with > max_nodata_frac
no-data (alpha == 0 or outside the map) are dropped.

Outputs: <out_root>/gallery/<F>m/<tile_id>.jpg and <out_root>/gallery/meta.csv with
         tile_id, scale_idx, F_m, px_x, px_y, center_lat, center_lon, nodata
Run: python -m data_prep.build_gallery [--workers 8]
"""
import argparse
import os
import shutil
from multiprocessing import Pool

import numpy as np

from .common import load_cfg, out_path, write_csv
from .georef import Ortho

FIELDS = ["tile_id", "scale_idx", "F_m", "px_x", "px_y", "center_lat", "center_lon", "nodata", "path"]
THUMB_F = 8
_G = {}


def _init(cfg):
    _G["cfg"] = cfg
    _G["ortho"] = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))


def _make(job):
    cfg, ortho = _G["cfg"], _G["ortho"]
    size = cfg["gallery"]["out_size"]
    side = job["side_px"]
    factor = max(1, int(side // size))
    win = ortho.window(job["px_x"], job["px_y"], side / 2 + 2, factor)
    tile, nod = win.crop(job["px_x"], job["px_y"], side, 0.0, size)
    if nod > cfg["ortho"]["max_nodata_frac"]:
        return None
    os.makedirs(os.path.dirname(job["abs"]), exist_ok=True)
    tile.save(job["abs"], quality=95)
    return round(nod, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_cfg()
    gc = cfg["gallery"]
    ortho = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))
    geo = ortho.geo
    root = out_path(cfg, "gallery")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)

    a = np.asarray(ortho.arr[::THUMB_F, ::THUMB_F, 3]) >= 128
    ii = np.pad((~a).astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    Ht, Wt = a.shape
    lim = cfg["ortho"]["max_nodata_frac"]

    jobs = []
    tid = 0
    for si, F in enumerate(gc["footprint_set_m"]):
        side = F / geo.gsd
        stride = gc["stride_frac"] * side
        xs = np.arange(side / 2 - side, geo.width + side / 2, stride)
        ys = np.arange(side / 2 - side, geo.height + side / 2, stride)
        n_pre = 0
        for cy in ys:
            for cx in xs:
                r = side / 2 / THUMB_F
                x0, x1 = int(np.clip(cx / THUMB_F - r, 0, Wt)), int(np.clip(cx / THUMB_F + r, 0, Wt))
                y0, y1 = int(np.clip(cy / THUMB_F - r, 0, Ht)), int(np.clip(cy / THUMB_F + r, 0, Ht))
                area = (2 * r) ** 2
                missing = area - (x1 - x0) * (y1 - y0) + (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
                if missing / area > lim + 0.05:        # cheap pre-filter, exact check after the crop
                    continue
                n_pre += 1
                lat, lon = geo.px_to_latlon(cx, cy)
                rel = os.path.join("%gm" % F, "%06d.jpg" % tid)
                jobs.append({"tile_id": tid, "scale_idx": si, "F_m": F, "px_x": float(cx), "px_y": float(cy),
                             "center_lat": float(lat), "center_lon": float(lon), "side_px": side,
                             "path": rel, "abs": os.path.join(root, rel)})
                tid += 1
        print("F=%g m: side %.0f px, stride %.1f m, %d candidate tiles" % (F, side, gc["stride_frac"] * F, n_pre))

    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_make, jobs, chunksize=16)
    rows = []
    for j, nod in zip(jobs, res):
        if nod is None:
            continue
        rows.append({k: (round(j[k], 8) if k in ("center_lat", "center_lon") else
                         round(j[k], 1) if k in ("px_x", "px_y") else j[k]) for k in FIELDS if k != "nodata"}
                    | {"nodata": nod})
    write_csv(os.path.join(root, "meta.csv"), rows, FIELDS)
    for si, F in enumerate(gc["footprint_set_m"]):
        print("F=%g m: kept %d tiles" % (F, sum(1 for r in rows if r["scale_idx"] == si)))
    print("gallery total %d tiles -> %s" % (len(rows), os.path.join(root, "meta.csv")))


if __name__ == "__main__":
    main()
