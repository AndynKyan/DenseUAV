"""M2 acceptance figures in <out_root>/debug/:
  pairs_real.png   all real pairs (UAV north-up | positive ortho tile), border colour = split
  pairs_pseudo.png 8 grid classes, one per row: satellite | 4 drone variants | merged real image
  split_map.png    ortho thumbnail with pseudo regions, test exclusion, real centres and pseudo centres

Run: python -m data_prep.debug_vis
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import load_cfg, out_path, read_csv
from .georef import Ortho
from .split_geo import load_regions

COL = {"train": (30, 120, 255), "val": (255, 150, 0), "test": (230, 30, 30), "excluded": (128, 128, 128)}


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def pairs_real(cfg):
    root = cfg["paths"]["out_root"]
    rows = read_csv(out_path(cfg, "labels", "split.csv"))
    t, gap, lab = 180, 8, 22
    ncol = 4
    nrow = int(np.ceil(len(rows) / ncol))
    cw = 2 * t + 3 * gap
    W, H = ncol * (cw + gap) + gap, nrow * (t + lab + 2 * gap) + gap + 30
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    f = _font(15)
    d.text((gap, 6), "UAV north-up (trái) | ô ortho positive (phải). Viền: xanh=train, cam=val, đỏ=test, xám=bị loại",
           fill="black", font=f)
    for k, r in enumerate(rows):
        stem = r["file"][:-4]
        x = gap + (k % ncol) * (cw + gap)
        y = 30 + gap + (k // ncol) * (t + lab + 2 * gap)
        col = COL[r["split"]]
        d.rectangle([x - 3, y - 3, x + cw - gap + 3, y + lab + t + 3], outline=col, width=4)
        d.text((x + 2, y), "%s  %s  s=%sm nodata=%.2f" % (stem[4:], r["split"], r["s_m"], float(r["nodata"])), fill=col, font=f)
        for j, sub in enumerate(("uav", "pos")):
            im = Image.open(os.path.join(root, "work", "real", sub, stem + ".jpg")).resize((t, t))
            c.paste(im, (x + j * (t + gap), y + lab))
    p = out_path(cfg, "debug", "pairs_real.png")
    c.save(p)
    return p


def pairs_pseudo(cfg, n=8, seed=0):
    """One row per class: satellite tile + 4 drone variants (+ merged real UAV image if any)."""
    import json
    root = cfg["paths"]["out_root"]
    rows = read_csv(out_path(cfg, "labels", "pseudo_train.csv"))
    rng = np.random.default_rng(seed)
    with_real = [i for i, r in enumerate(rows) if r["merged_real"]]
    others = [i for i, r in enumerate(rows) if not r["merged_real"]]
    k_real = min(len(with_real), n // 2)
    pick = sorted(list(rng.choice(with_real, k_real, replace=False)) +
                  list(rng.choice(others, min(len(others), n - k_real), replace=False)))
    t, gap, lab = 190, 8, 22
    ncol = 6
    W = gap + ncol * (t + gap)
    c = Image.new("RGB", (W, 34 + len(pick) * (t + lab + gap) + gap), "white")
    d = ImageDraw.Draw(c)
    f = _font(15)
    d.text((gap, 8), "Mỗi hàng là một lớp: ô ortho north-up | 4 biến thể drone (lệch ≤3 m, scale 0.8–1.25, xoay ±15°, C1, blur)"
           " | ảnh UAV thật đã gộp (nếu có)", fill="black", font=f)
    for row, i in enumerate(pick):
        r = rows[i]
        y = 34 + row * (t + lab + gap)
        vs = json.loads(r["variants"])
        d.text((gap, y), "lớp %s  s=%s m  sat=%s m  %d biến thể%s" % (
            r["id"], r["s_m"], r["F_sat_m"], len(vs), ("  + " + r["merged_real"]) if r["merged_real"] else ""),
            fill="black", font=f)
        tiles = [("sat", Image.open(os.path.join(root, "train", "satellite", r["id"], "sat.jpg")))]
        for k in range(min(4, len(vs))):
            v = vs[k]
            tiles.append(("%.0f° x%.2f %.1fm" % (v["rot_deg"], v["scale"], math.hypot(*v["shift_m"])),
                          Image.open(os.path.join(root, "train", "drone", r["id"], "v%02d.jpg" % k))))
        for fn in [x for x in r["merged_real"].split(";") if x]:
            tiles.append((fn[:-4] + " (thật)", Image.open(os.path.join(root, "train", "drone", r["id"], fn[:-4] + ".jpg"))))
        for col, (name, im) in enumerate(tiles[:ncol]):
            x = gap + col * (t + gap)
            c.paste(im.resize((t, t)), (x, y + lab))
            d.rectangle([x, y + lab + t - 20, x + t, y + lab + t], fill="white")
            d.text((x + 3, y + lab + t - 19), name, fill="black", font=_font(13))
    p = out_path(cfg, "debug", "pairs_pseudo.png")
    c.save(p)
    return p


def split_map(cfg, f=16):
    ortho = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))
    geo = ortho.geo
    reg = load_regions(cfg, geo)
    th = np.asarray(ortho.arr[::f, ::f]).copy()
    h, w = th.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    lab = reg.label((xx + 0.5) * f, (yy + 0.5) * f)
    valid = th[..., 3] >= 128
    rgb = th[..., :3].astype(np.float32)
    rgb[~valid] = 255
    for v, col in ((1, COL["train"]), (2, COL["val"])):
        m = (lab == v) & valid
        rgb[m] = rgb[m] * 0.55 + np.array(col) * 0.45
    excl = (reg.dist_test((xx + 0.5) * f, (yy + 0.5) * f) <= reg.d["test_exclusion_m"]) & valid
    rgb[excl] = rgb[excl] * 0.7 + np.array([255, 200, 200]) * 0.3
    pad_r = 420
    canvas = Image.new("RGB", (w + pad_r, h), "white")
    canvas.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 0))
    d = ImageDraw.Draw(canvas)
    fn = _font(16)
    for s, col in (("train", COL["train"]), ("val", COL["val"])):
        for r in read_csv(out_path(cfg, "labels", "pseudo_%s.csv" % s)):
            x, y = float(r["px_x"]) / f, float(r["px_y"]) / f
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 40, 140) if s == "train" else (90, 0, 90))
    # test exclusion circles
    for (tx, ty) in reg.test_px:
        rr = reg.d["test_exclusion_m"] / geo.gsd / f
        d.ellipse([tx / f - rr, ty / f - rr, tx / f + rr, ty / f + rr], outline=(230, 30, 30), width=1)
    for r in read_csv(out_path(cfg, "labels", "split.csv")):
        x, y = float(r["px_x"]) / f, float(r["px_y"]) / f
        col = COL[r["split"]]
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline="black")
        d.text((x + 10, y - 9), r["file"][4:8], fill="black", font=fn)
    lx = w + 15
    items = [("Tâm ảnh UAV: train", COL["train"]), ("Tâm ảnh UAV: val", COL["val"]),
             ("Tâm ảnh UAV: test", COL["test"]), ("Tâm ảnh UAV: bị loại", COL["excluded"]),
             ("Nền xanh: vùng pseudo-train", COL["train"]), ("Nền cam: vùng pseudo-val", COL["val"]),
             ("Nền hồng + vòng đỏ: vùng cấm quanh test", (230, 30, 30)),
             ("Chấm xanh đậm: tâm lớp lưới train (14 m)", (0, 40, 140)), ("Chấm tím: tâm lớp lưới val", (90, 0, 90))]
    for i, (t, col) in enumerate(items):
        d.rectangle([lx, 20 + i * 28, lx + 18, 38 + i * 28], fill=col)
        d.text((lx + 26, 20 + i * 28), t, fill="black", font=fn)
    d.text((lx, 20 + len(items) * 28 + 10),
           "buffer_m = %.1f m\nbán kính cấm test = %.1f m\nval: s từ %.0f đến %.0f m\nthu nhỏ 1/%d (%.2f m/px)" %
           (reg.d["buffer_m"], reg.d["test_exclusion_m"], reg.d["s_val_lo"], reg.d["s_val_hi"], f, geo.gsd * f),
           fill="black", font=fn)
    p = out_path(cfg, "debug", "split_map.png")
    canvas.save(p)
    return p


def main():
    cfg = load_cfg()
    for fn in (pairs_real, pairs_pseudo, split_map):
        print(fn(cfg))


if __name__ == "__main__":
    main()
