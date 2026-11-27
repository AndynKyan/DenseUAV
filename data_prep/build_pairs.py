"""North-up UAV crops + positive ortho tiles for every real image.

UAV:  centred square (min(w,h) px) -> PIL rotate(rotate_sign * gimbal_yaw, expand=True)
      -> largest inscribed axis-aligned square -> resize out_size.
Positive: ortho square centred at the image GPS, side = side of the UAV north-up crop in metres
      times north_up.footprint_scale (calibrated, see configs/data.yaml),
      north-up, resize out_size.
Outputs: <out_root>/work/real/{uav,pos}/<stem>.jpg and labels/pairs_real.csv

Run: python -m data_prep.build_pairs [--only DJI_0282] [--sign -1]
"""
import argparse
import os
from PIL import Image

from .common import load_cfg, out_path, load_cameras, read_csv, write_csv
from .footprint import gsd_uav, north_up_side
from .georef import Ortho

FIELDS = ["file", "lat", "lon", "px_x", "px_y", "side_m", "side_m_nominal", "side_px_ortho", "gsd_uav", "nodata"]


def uav_north_up(path, yaw, sign, out_size):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    sq = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    rot = sq.rotate(sign * yaw, resample=Image.BICUBIC, expand=True)
    from .footprint import inscribed_square_factor
    side = s * inscribed_square_factor(yaw)
    cx, cy = rot.size[0] / 2, rot.size[1] / 2
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return rot.transform((out_size, out_size), Image.EXTENT, box, resample=Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated file stems, for quick checks")
    ap.add_argument("--sign", type=int, default=None, help="override north_up.rotate_sign")
    ap.add_argument("--tag", default="", help="suffix for output folders (orientation checks)")
    args = ap.parse_args()

    cfg = load_cfg()
    cams = load_cameras(cfg)
    sign = cfg["north_up"]["rotate_sign"] if args.sign is None else args.sign
    size = cfg["north_up"]["out_size"]
    ortho = Ortho(cfg["paths"]["ortho_png"], out_path(cfg, "cache"))
    g = ortho.geo
    rows = read_csv(out_path(cfg, "labels", "metadata.csv"))
    only = set(s for s in args.only.split(",") if s)
    out = []
    for r in rows:
        stem = os.path.splitext(r["file"])[0]
        if only and stem not in only:
            continue
        cam = cams[r["model"]]
        lat, lon = float(r["lat"]), float(r["lon"])
        alt, focal, yaw = float(r["rel_alt"]), float(r["focal"]), float(r["gimbal_yaw"])
        w, h = int(r["w"]), int(r["h"])
        _, side_nom = north_up_side(alt, cam["sensor_width_mm"], focal, w, h, yaw)
        side_m = side_nom * cfg["north_up"].get("footprint_scale", 1.0)
        uav = uav_north_up(os.path.join(cfg["paths"]["uav_dir"], r["file"]), yaw, sign, size)
        px, py = g.latlon_to_px(lat, lon)
        side_px = g.m_to_px(side_m)
        pos, nod = ortho.crop(float(px), float(py), side_px, out_size=size)
        uav.save(out_path(cfg, "work", "real" + args.tag, "uav", stem + ".jpg"), quality=95)
        pos.save(out_path(cfg, "work", "real" + args.tag, "pos", stem + ".jpg"), quality=95)
        out.append({"file": r["file"], "lat": lat, "lon": lon, "px_x": round(float(px), 2), "px_y": round(float(py), 2),
                    "side_m": round(side_m, 3), "side_m_nominal": round(side_nom, 3), "side_px_ortho": round(side_px, 1),
                    "gsd_uav": round(gsd_uav(alt, cam["sensor_width_mm"], focal, w), 5), "nodata": round(nod, 4)})
    if not only:
        write_csv(out_path(cfg, "labels", "pairs_real.csv"), out, FIELDS)
    for o in out[:3] + out[-2:]:
        print(o)
    print("pairs: %d, rotate_sign=%d" % (len(out), sign))


if __name__ == "__main__":
    main()
