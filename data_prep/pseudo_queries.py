"""Pseudo-queries cut from the ortho, one CLASS per centre of a regular grid (docs/demo_deviations.md D16).

Grid: spacing grid_spacing_m, aligned with the flight line (axes u = along-track, v = across-track from
labels/regions.json). A grid centre belongs to pseudo-train / pseudo-val by split_geo.Regions.label
(same regions and test exclusion as before). S = median calibrated side of the real UAV north-up crops.

  satellite/<id>/sat.jpg      north-up ortho tile, side sat_scale * S, at the grid centre
  drone/<id>/v00..v11.jpg     variants_per_class views: centre shift uniform in a disc of radius
                              shift_max_m, side U(scale_range) * S, rotation U(+-rot_deg), colour C1,
                              Gaussian blur U(blur_sigma)
A centre is dropped if its satellite tile has > max_nodata_frac no-data; a variant is resampled
(max_tries_per_variant) if its tile has too much no-data.

Real train/val images closer than merge_real_within_m to a kept grid centre of the same split are
MERGED into that class (export_dataset.py adds the UAV image to drone/<id>/), so no two classes
show the same place. Others stay as their own class.

Outputs: <out_root>/work/pseudo/{train,val}/{drone,satellite}/<id>/, labels/pseudo_{train,val}.csv
Run: python -m data_prep.pseudo_queries [--workers 8]
"""
import argparse
import json
import math
import os
import shutil
from multiprocessing import Pool

import numpy as np

from .common import load_cfg, out_path, read_csv, write_csv
from .color_aug import apply_c1
from .georef import Ortho
from .split_geo import load_regions

FIELDS = ["id", "split", "lat", "lon", "px_x", "px_y", "s_m", "F_sat_m", "nodata_sat", "n_variants",
          "merged_real", "variants"]

_G = {}


def _init(cfg):
    _G["cfg"] = cfg
    _G["ortho"] = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))


def _make(job):
    cfg, ortho = _G["cfg"], _G["ortho"]
    pc = cfg["pseudo"]
    size = cfg["north_up"]["out_size"]
    max_nod = cfg["ortho"]["max_nodata_frac"]
    gsd = job["gsd"]
    S_px = job["S_px"]
    rng = np.random.default_rng(job["seed"])
    cx, cy = job["px_x"], job["px_y"]
    shift_px = pc["shift_max_m"] / gsd
    factor = max(1, int(pc["scale_range"][0] * S_px // size))
    half = pc["scale_range"][1] * S_px / 2 * math.sqrt(2) + shift_px + 4
    win = ortho.window(cx, cy, half, factor)
    sat, nod = win.crop(cx, cy, pc["sat_scale"] * S_px, 0.0, size)
    if nod > max_nod:
        return None
    variants, imgs = [], []
    for k in range(pc["variants_per_class"]):
        for _ in range(pc["max_tries_per_variant"]):
            r = shift_px * math.sqrt(rng.uniform())
            t = rng.uniform(0, 2 * math.pi)
            dx, dy = r * math.cos(t), r * math.sin(t)
            sc = float(rng.uniform(*pc["scale_range"]))
            ang = float(rng.uniform(-pc["rot_deg"], pc["rot_deg"]))
            d, dn = win.crop(cx + dx, cy + dy, sc * S_px, ang, size)
            if dn <= max_nod:
                break
        else:
            continue
        sigma = float(rng.uniform(*pc["blur_sigma"]))
        d, rec = apply_c1(d, rng, cfg["color_C1"], sigma)
        rec.update(rot_deg=round(ang, 2), scale=round(sc, 3), side_m=round(sc * S_px * gsd, 2),
                   shift_m=[round(dx * gsd, 2), round(dy * gsd, 2)], nodata=round(dn, 4))
        variants.append(rec)
        imgs.append(d)
    if not imgs:
        return None
    root = out_path(cfg, "work", "pseudo", job["split"])
    sid = "%06d" % job["id"]
    os.makedirs(os.path.join(root, "satellite", sid), exist_ok=True)
    os.makedirs(os.path.join(root, "drone", sid), exist_ok=True)
    sat.save(os.path.join(root, "satellite", sid, "sat.jpg"), quality=95)
    for k, im in enumerate(imgs):
        im.save(os.path.join(root, "drone", sid, "v%02d.jpg" % k), quality=95)
    return {"nodata_sat": round(nod, 4), "n_variants": len(imgs), "variants": json.dumps(variants)}


def grid_centres(cfg, geo, regions):
    """All grid centres (px_x, px_y, s_m, label) over the ortho, label from Regions.label."""
    sp = cfg["pseudo"]["grid_spacing_m"]
    u = regions.u
    v = np.array([-u[1], u[0]])
    corners = np.array([[0, 0], [geo.width, 0], [0, geo.height], [geo.width, geo.height]], float)
    e, n = regions.enu(corners[:, 0], corners[:, 1])
    E = np.stack([e, n], 1)
    a_rng, b_rng = E @ u, E @ v
    ii = np.arange(math.floor(a_rng.min() / sp), math.ceil(a_rng.max() / sp) + 1)
    jj = np.arange(math.floor(b_rng.min() / sp), math.ceil(b_rng.max() / sp) + 1)
    I, J = np.meshgrid(ii, jj, indexing="ij")
    P = I.reshape(-1, 1) * sp * u + J.reshape(-1, 1) * sp * v           # (e, n) metres
    px = regions.ref[0] + P[:, 0] / geo.mpp_x
    py = regions.ref[1] - P[:, 1] / geo.mpp_y
    inside = (px >= 0) & (px < geo.width) & (py >= 0) & (py < geo.height)
    px, py = px[inside], py[inside]
    return px, py, regions.s_of(px, py), regions.label(px, py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_cfg()
    pc = cfg["pseudo"]
    ortho = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))
    geo = ortho.geo
    regions = load_regions(cfg, geo)
    pairs = read_csv(out_path(cfg, "labels", "pairs_real.csv"))
    S = float(np.median([float(p["side_m"]) for p in pairs]))
    print("UAV north-up side S = %.2f m; satellite tile %.1f m; drone variants %.1f..%.1f m; grid %.1f m" %
          (S, pc["sat_scale"] * S, pc["scale_range"][0] * S, pc["scale_range"][1] * S, pc["grid_spacing_m"]))

    px, py, s, lab = grid_centres(cfg, geo, regions)
    shutil.rmtree(out_path(cfg, "work", "pseudo"), ignore_errors=True)
    jobs = []
    for split, which, off in (("train", 1, cfg["ids"]["pseudo_train_offset"]), ("val", 2, cfg["ids"]["pseudo_val_offset"])):
        idx = np.nonzero(lab == which)[0]
        idx = idx[np.lexsort((px[idx], s[idx]))]          # ids ordered north -> south, then west -> east
        print("%s: %d grid centres inside the region" % (split, len(idx)))
        for k, i in enumerate(idx):
            lat, lon = geo.px_to_latlon(px[i], py[i])
            jobs.append({"split": split, "id": off + k, "px_x": float(px[i]), "px_y": float(py[i]), "s_m": float(s[i]),
                         "lat": float(lat), "lon": float(lon), "gsd": geo.gsd, "S_px": S / geo.gsd,
                         "seed": pc["seed"] * 1000003 + off + k})

    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_make, jobs, chunksize=4)

    split_rows = read_csv(out_path(cfg, "labels", "split.csv"))
    for split in ("train", "val"):
        rows = []
        for j, r in zip(jobs, res):
            if j["split"] == split and r is not None:
                rows.append({"id": "%06d" % j["id"], "split": split, "lat": round(j["lat"], 8), "lon": round(j["lon"], 8),
                             "px_x": round(j["px_x"], 1), "px_y": round(j["px_y"], 1), "s_m": round(j["s_m"], 2),
                             "F_sat_m": round(pc["sat_scale"] * S, 2), "merged_real": "", **r})
        # merge real images of the same split into the nearest grid class
        if rows:
            G = np.array([[float(r["px_x"]), float(r["px_y"])] for r in rows])
            for rr in split_rows:
                if rr["split"] != split:
                    continue
                d = np.hypot((G[:, 0] - float(rr["px_x"])) * geo.mpp_x, (G[:, 1] - float(rr["px_y"])) * geo.mpp_y)
                k = int(np.argmin(d))
                if d[k] < pc["merge_real_within_m"]:
                    rows[k]["merged_real"] = ";".join(x for x in (rows[k]["merged_real"], rr["file"]) if x)
                    print("  merge %s -> class %s (%.1f m)" % (rr["file"], rows[k]["id"], d[k]))
                else:
                    print("  keep  %s as its own class (nearest grid centre %.1f m)" % (rr["file"], d[k]))
        write_csv(out_path(cfg, "labels", "pseudo_%s.csv" % split), rows, FIELDS)
        nrej = sum(1 for j, r in zip(jobs, res) if j["split"] == split and r is None)
        nv = sum(int(r["n_variants"]) for r in rows)
        print("%s: kept %d classes (%d drone variants), dropped %d centres for no-data" % (split, len(rows), nv, nrej))


if __name__ == "__main__":
    main()
