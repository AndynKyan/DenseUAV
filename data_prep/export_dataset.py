"""Assemble the DenseUAV-format folders from work/ and print the count table.

<out_root>/train/{drone,satellite}/<id6>/   grid classes (pseudo) + real train images: merged into a grid class
                                            (drone view only) or their own class (id = DJI number)
<out_root>/val/{drone,satellite}/<id6>/     real val pairs + pseudo-val
<out_root>/test/query_drone/<id6>/          north-up UAV test queries (GT in labels/test_gt.csv)
train/, val/ and test/query_drone/ are rebuilt from scratch; hidden files (.DS_Store, ._*) are removed.
test/gallery_satellite/ is left alone (the M3 gallery is built separately by build_gallery.py).

Run: python -m data_prep.export_dataset
"""
import os
import shutil

import numpy as np

from .common import load_cfg, out_path, read_csv, remove_hidden
from .georef import Georef


def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _nn_stats(pts, geo):
    """Nearest-neighbour distance (m) between class centres: (min, median)."""
    if len(pts) < 2:
        return float("nan"), float("nan")
    p = np.asarray(pts, float)
    d = np.hypot((p[:, None, 0] - p[None, :, 0]) * geo.mpp_x, (p[:, None, 1] - p[None, :, 1]) * geo.mpp_y)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    return float(nn.min()), float(np.median(nn))


def main():
    cfg = load_cfg()
    geo = Georef.from_png(cfg["paths"]["ortho_png"])
    root = cfg["paths"]["out_root"]
    for sub in ("train", "val", os.path.join("test", "query_drone")):
        shutil.rmtree(os.path.join(root, sub), ignore_errors=True)
    off = cfg["ids"]["real_offset"]
    split = read_csv(out_path(cfg, "labels", "split.csv"))
    real = os.path.join(root, "work", "real")
    pseudo = {s: read_csv(out_path(cfg, "labels", "pseudo_%s.csv" % s)) for s in ("train", "val")}
    merged = {f: (s, r["id"]) for s in pseudo for r in pseudo[s] for f in r["merged_real"].split(";") if f}

    # class centres (pixel) per split, for the NN statistics
    centres = {s: [[float(r["px_x"]), float(r["px_y"])] for r in pseudo[s]] for s in pseudo}
    n_real = {"train": 0, "val": 0, "test": 0}
    n_merged = {"train": 0, "val": 0}
    for r in split:
        stem = os.path.splitext(r["file"])[0]
        sid = "%06d" % (off + int(stem.split("_")[-1]))
        if r["split"] in ("train", "val"):
            if r["file"] in merged:
                ms, mid = merged[r["file"]]
                assert ms == r["split"]
                _link(os.path.join(real, "uav", stem + ".jpg"), os.path.join(root, ms, "drone", mid, stem + ".jpg"))
                n_merged[ms] += 1
            else:
                _link(os.path.join(real, "uav", stem + ".jpg"), os.path.join(root, r["split"], "drone", sid, stem + ".jpg"))
                _link(os.path.join(real, "pos", stem + ".jpg"), os.path.join(root, r["split"], "satellite", sid, stem + ".jpg"))
                centres[r["split"]].append([float(r["px_x"]), float(r["px_y"])])
                n_real[r["split"]] += 1
        elif r["split"] == "test":
            _link(os.path.join(real, "uav", stem + ".jpg"), os.path.join(root, "test", "query_drone", sid, stem + ".jpg"))
            n_real["test"] += 1

    for s in ("train", "val"):
        src = os.path.join(root, "work", "pseudo", s)
        for r in pseudo[s]:
            for view in ("satellite", "drone"):
                d = os.path.join(src, view, r["id"])
                for fn in sorted(os.listdir(d)):
                    if not fn.startswith("."):
                        _link(os.path.join(d, fn), os.path.join(root, s, view, r["id"], fn))

    removed = sum(remove_hidden(os.path.join(root, s)) for s in ("train", "val", "test"))

    # sanity: every class has both views, folder names are 6-digit integers
    for s in ("train", "val"):
        sat = set(os.listdir(os.path.join(root, s, "satellite")))
        dr = set(os.listdir(os.path.join(root, s, "drone")))
        assert sat == dr, "%s: satellite/drone class sets differ" % s
        assert all(c.isdigit() and len(c) == 6 for c in sat), "%s: non 6-digit class folder" % s

    def count(s, view):
        d = os.path.join(root, s, view)
        cls = os.listdir(d)
        return len(cls), sum(len(os.listdir(os.path.join(d, c))) for c in cls)

    excl = [r for r in split if r["split"] == "excluded"]
    lines = ["| Tập | Nguồn | Số lớp | Ảnh drone | Ảnh satellite |", "|---|---|---|---|---|"]
    for s in ("train", "val"):
        nc, nd = count(s, "drone")
        _, ns = count(s, "satellite")
        nv = sum(int(r["n_variants"]) for r in pseudo[s])
        lines.append("| %s | lớp lưới (pseudo) | %d | %d biến thể + %d ảnh UAV thật đã gộp | %d |" %
                     (s, len(pseudo[s]), nv, n_merged[s], len(pseudo[s])))
        lines.append("| %s | UAV thật, lớp riêng | %d | %d | %d |" % (s, n_real[s], n_real[s], n_real[s]))
        lines.append("| %s | **tổng** | **%d** | **%d** | **%d** |" % (s, nc, nd, ns))
    lines.append("| test | UAV thật (query) | %d | %d | – |" % (n_real["test"], n_real["test"]))
    lines.append("| bị loại | UAV thật | %d (buffer %d, nodata %d) | | |" % (
        len(excl), sum(r["reason"].startswith("buffer") for r in excl), sum(r["reason"].startswith("nodata") for r in excl)))
    lines.append("")
    lines.append("Khoảng cách tới tâm lớp gần nhất (m):")
    lines.append("")
    lines.append("| Tập | Tâm tính | min | trung vị |")
    lines.append("|---|---|---|---|")
    for s in ("train", "val"):
        grid = [[float(r["px_x"]), float(r["px_y"])] for r in pseudo[s]]
        a = _nn_stats(grid, geo)
        b = _nn_stats(centres[s], geo)
        lines.append("| %s | chỉ lớp lưới | %.2f | %.2f |" % (s, *a))
        lines.append("| %s | lưới + lớp UAV thật riêng | %.2f | %.2f |" % (s, *b))
    lines.append("")
    lines.append("file ẩn đã xoá: %d" % removed)
    table = "\n".join(lines)
    with open(out_path(cfg, "labels", "counts.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
