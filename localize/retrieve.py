"""Retrieval-based localization: query -> top-K gallery tiles -> predicted lat/lon = centre of top-1.

Models (--model):
  z0 | z0cls       timm ViT-S ImageNet, mean-pooled patch tokens / CLS, no head (docs/demo_deviations.md)
  denseuav         a train.py checkpoint: --ckpt_dir checkpoints/<name> --checkpoint net_XXX.pth
  oracle           sanity check: random gallery descriptors; each query gets EXACTLY the descriptor of the
                   gallery tile whose centre is closest to its GT (any scale). Must give R@1 = 100% and an error
                   equal to the gallery quantisation floor (distance GT -> nearest tile centre).
                   eval.metrics self-test (GT exactly on a tile centre) checks the error is exactly 0.
  random           sanity check: random query and gallery descriptors (chance level)
Query sets (--queries, comma-separated):
  test        14 north-up UAV test images (labels/test_gt.csv, flag=edge kept as a column)
  pseudo_val  val/drone/*: 12 pseudo variants per grid class (GT = grid centre + variant shift) + merged
              real val images (GT = their GPS)
Gallery descriptors are cached in <out_root>/cache/desc/<sha1(model key, footprint_set, meta.csv)>.npy.
Search: exact inner product with torch matmul on L2-normalised descriptors (D19, no FAISS).

Outputs: results/<exp>__<queries>.csv (one row per query, top-3 included) and results/<exp>__<queries>.png
Run: python -m localize.retrieve --exp Z0 --model z0 --queries test,pseudo_val
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from data_prep.common import REPO_ROOT, load_cfg, out_path, read_csv, write_csv
from data_prep.georef import Georef
from eval.metrics import per_query

TOPK = 20
FIELDS = ["query", "kind", "flag", "gt_lat", "gt_lon", "gt_px_x", "gt_px_y", "F_gt_m", "pred_lat", "pred_lon",
          "pred_px_x", "pred_px_y", "pred_F_m", "pred_tile", "score1", "err_m", "rank_first_pos", "n_pos",
          "sdm@1", "sdm@3", "sdm@5", "sdm@10", "top3"]


def load_gallery(cfg):
    root = out_path(cfg, "gallery")
    meta = read_csv(os.path.join(root, "meta.csv"))
    paths = [os.path.join(root, m["path"]) for m in meta]
    return meta, paths, hashlib.sha1(open(os.path.join(root, "meta.csv"), "rb").read()).hexdigest()


def load_queries(cfg, name, geo):
    root = cfg["paths"]["out_root"]
    Q = []
    if name == "test":
        for r in read_csv(out_path(cfg, "labels", "test_gt.csv")):
            stem = r["file"][:-4]
            sid = "%06d" % (cfg["ids"]["real_offset"] + int(stem.split("_")[-1]))
            Q.append({"query": stem, "kind": "real", "flag": r.get("flag", ""),
                      "path": os.path.join(root, "test", "query_drone", sid, stem + ".jpg"),
                      "px": float(r["px_x"]), "py": float(r["px_y"]), "F": float(r["side_m"])})
    elif name == "pseudo_val":
        split = {r["file"]: r for r in read_csv(out_path(cfg, "labels", "split.csv"))}
        for r in read_csv(out_path(cfg, "labels", "pseudo_val.csv")):
            for k, v in enumerate(json.loads(r["variants"])):
                Q.append({"query": "%s/v%02d" % (r["id"], k), "kind": "pseudo", "flag": "",
                          "path": os.path.join(root, "val", "drone", r["id"], "v%02d.jpg" % k),
                          "px": float(r["px_x"]) + v["shift_m"][0] / geo.gsd,
                          "py": float(r["px_y"]) + v["shift_m"][1] / geo.gsd, "F": float(v["side_m"])})
            for f in [x for x in r["merged_real"].split(";") if x]:
                s = split[f]
                Q.append({"query": "%s/%s" % (r["id"], f[:-4]), "kind": "real", "flag": "",
                          "path": os.path.join(root, "val", "drone", r["id"], f[:-4] + ".jpg"),
                          "px": float(s["px_x"]), "py": float(s["px_y"]), "F": float(s["side_m"])})
    else:
        raise ValueError(name)
    for q in Q:
        q["lat"], q["lon"] = (float(v) for v in geo.px_to_latlon(q["px"], q["py"]))
    return Q


def oracle_index(q, g_lat, g_lon, g_F):
    from data_prep.georef import haversine_m
    d = haversine_m(q["lat"], q["lon"], g_lat, g_lon)
    return int(np.argmin(d))


def plot(cfg, geo, rows, title, path, f=16):
    from PIL import Image, ImageDraw
    from data_prep.georef import Ortho
    ortho = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))
    th = np.asarray(ortho.arr[::f, ::f]).copy()
    rgb = th[..., :3].copy()
    rgb[th[..., 3] < 128] = 255
    im = Image.fromarray(rgb)
    pad = 420
    c = Image.new("RGB", (im.size[0] + pad, im.size[1]), "white")
    c.paste(im, (0, 0))
    d = ImageDraw.Draw(c)
    for r in rows:
        gx, gy = float(r["gt_px_x"]) / f, float(r["gt_px_y"]) / f
        px, py = float(r["pred_px_x"]) / f, float(r["pred_px_y"]) / f
        ok = int(r["rank_first_pos"]) == 1
        d.line([gx, gy, px, py], fill=(0, 160, 0) if ok else (220, 0, 0), width=2)
        d.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(255, 255, 255), outline="black")
        d.rectangle([px - 3, py - 3, px + 3, py + 3], fill=(0, 160, 0) if ok else (220, 0, 0))
    x = im.size[0] + 12
    from data_prep.debug_vis import _font
    font = _font(16)
    for i, t in enumerate([title, "", "o trắng = GT", "vuông = tâm ô top-1",
                           "xanh = top-1 là positive", "đỏ = top-1 sai", "",
                           "median err %.1f m" % np.median([float(r["err_m"]) for r in rows]),
                           "n = %d" % len(rows)]):
        d.text((x, 16 + 20 * i), t, fill="black", font=font)
    c.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, choices=["z0", "z0cls", "denseuav", "oracle", "random"])
    ap.add_argument("--queries", default="test,pseudo_val")
    ap.add_argument("--ckpt_dir", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results", help="relative to the repo root")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    geo = Georef.from_png(cfg["paths"]["ortho_png"])
    meta, gpaths, meta_hash = load_gallery(cfg)
    g_lat = np.array([float(m["center_lat"]) for m in meta])
    g_lon = np.array([float(m["center_lon"]) for m in meta])
    g_F = np.array([float(m["F_m"]) for m in meta])
    res_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(res_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ext = None
    if args.model in ("z0", "z0cls"):
        from localize.features import z0_extractor
        ext = z0_extractor("mean" if args.model == "z0" else "cls")
    elif args.model == "denseuav":
        from localize.features import denseuav_extractor
        ext = denseuav_extractor(args.ckpt_dir, args.checkpoint)

    if ext is not None:
        key = hashlib.sha1(("%s|%s|%s" % (ext.key, cfg["gallery"]["footprint_set_m"], meta_hash)).encode()).hexdigest()
        cache = out_path(cfg, "cache", "desc", key + ".npy")
        if os.path.exists(cache):
            G = np.load(cache)
            print("gallery descriptors from cache", cache)
        else:
            G = ext(gpaths, args.batch_size)
            np.save(cache, G)
            with open(cache[:-4] + ".txt", "w") as fp:
                fp.write(ext.key + "\n%s\n%s\n" % (cfg["gallery"]["footprint_set_m"], meta_hash))
            print("gallery descriptors computed:", G.shape, "->", cache)
    else:
        G = rng.standard_normal((len(meta), 128)).astype(np.float32)
        G /= np.linalg.norm(G, axis=1, keepdims=True)

    Gt = torch.from_numpy(G)
    for qs in args.queries.split(","):
        Q = load_queries(cfg, qs, geo)
        if ext is not None:
            QF = ext([q["path"] for q in Q], args.batch_size)
        elif args.model == "oracle":
            QF = np.stack([G[oracle_index(q, g_lat, g_lon, g_F)] for q in Q])
        else:
            QF = rng.standard_normal((len(Q), G.shape[1])).astype(np.float32)
            QF /= np.linalg.norm(QF, axis=1, keepdims=True)
        S = torch.from_numpy(QF) @ Gt.T
        scores, order = torch.sort(S, dim=1, descending=True)
        rows = []
        for i, q in enumerate(Q):
            o = order[i].numpy()
            m = per_query(o, q["lat"], q["lon"], q["F"], g_lat, g_lon)
            t = meta[o[0]]
            rows.append({"query": q["query"], "kind": q["kind"], "flag": q["flag"],
                         "gt_lat": round(q["lat"], 8), "gt_lon": round(q["lon"], 8),
                         "gt_px_x": round(q["px"], 1), "gt_px_y": round(q["py"], 1), "F_gt_m": round(q["F"], 2),
                         "pred_lat": t["center_lat"], "pred_lon": t["center_lon"], "pred_px_x": t["px_x"],
                         "pred_px_y": t["px_y"], "pred_F_m": t["F_m"], "pred_tile": t["tile_id"],
                         "score1": round(float(scores[i, 0]), 5), "err_m": round(m["err_m"], 2),
                         "rank_first_pos": m["rank_first_pos"], "n_pos": m["n_pos"],
                         **{k: round(m[k], 5) for k in ("sdm@1", "sdm@3", "sdm@5", "sdm@10")},
                         "top3": json.dumps([{"tile": meta[j]["tile_id"], "lat": float(meta[j]["center_lat"]),
                                              "lon": float(meta[j]["center_lon"]), "F": float(meta[j]["F_m"]),
                                              "score": round(float(S[i, j]), 5)} for j in o[:3]])})
        name = "%s__%s" % (args.exp, qs)
        write_csv(os.path.join(res_dir, name + ".csv"), rows, FIELDS)
        if not args.no_plot:
            plot(cfg, geo, rows, name, os.path.join(res_dir, name + ".png"))
        err = np.array([r["err_m"] for r in rows])
        r1 = np.mean([r["rank_first_pos"] == 1 for r in rows])
        print("%-28s n=%3d  median err %.1f m  mean %.1f m  R@1 %.0f%%  (no positive in gallery: %d)" %
              (name, len(rows), np.median(err), err.mean(), 100 * r1, sum(r["n_pos"] == 0 for r in rows)))


if __name__ == "__main__":
    main()
