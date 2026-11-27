"""EXIF/XMP of the raw UAV images -> <out_root>/labels/metadata.csv.

Fields: file, lat, lon, rel_alt, gimbal_yaw, gimbal_pitch, focal, w, h (+ model, flight_yaw, abs_alt).
lat/lon come from DJI XMP (GpsLatitude/GpsLongitude, 7 decimals); EXIF GPS is the fallback.
Images with |gimbal_pitch + 90| > max_pitch_dev_deg are dropped and logged.
GPS is a LABEL only; it is never given to the model.

Run: python -m data_prep.read_metadata
"""
import glob
import os
import re
from PIL import Image
from PIL.ExifTags import TAGS

from .common import load_cfg, out_path, write_csv

FIELDS = ["file", "lat", "lon", "rel_alt", "gimbal_yaw", "gimbal_pitch", "focal", "w", "h",
          "model", "flight_yaw", "abs_alt"]


def _dms(v, ref):
    d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -d if ref in ("S", "W") else d


def read_one(path):
    im = Image.open(path)
    ex = im.getexif()
    tags = {TAGS.get(k, k): v for k, v in ex.items()}
    sub = ex.get_ifd(0x8769)
    gps = ex.get_ifd(0x8825)
    raw = open(path, "rb").read()
    a, b = raw.find(b"<x:xmpmeta"), raw.find(b"</x:xmpmeta>")
    xmp = dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', raw[a:b].decode("utf8", "ignore"))) if a >= 0 else {}

    if "GpsLatitude" in xmp:
        lat, lon = float(xmp["GpsLatitude"]), float(xmp["GpsLongitude"])
    else:
        lat, lon = _dms(gps[2], gps[1]), _dms(gps[4], gps[3])
    return {
        "file": os.path.basename(path),
        "lat": lat, "lon": lon,
        "rel_alt": float(xmp["RelativeAltitude"]),
        "abs_alt": float(xmp.get("AbsoluteAltitude", "nan")),
        "gimbal_yaw": float(xmp["GimbalYawDegree"]),
        "gimbal_pitch": float(xmp["GimbalPitchDegree"]),
        "flight_yaw": float(xmp.get("FlightYawDegree", "nan")),
        "focal": float(sub[0x920A]),
        "w": im.size[0], "h": im.size[1],
        "model": str(tags.get("Model", "")).strip(),
    }


def main():
    cfg = load_cfg()
    files = sorted(glob.glob(os.path.join(cfg["paths"]["uav_dir"], "*.JPG")) +
                   glob.glob(os.path.join(cfg["paths"]["uav_dir"], "*.jpg")))
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    rows, dropped = [], []
    tol = cfg["metadata"]["max_pitch_dev_deg"]
    for f in files:
        r = read_one(f)
        if abs(r["gimbal_pitch"] + 90) > tol:
            dropped.append(r)
        else:
            rows.append(r)
    path = out_path(cfg, "labels", "metadata.csv")
    write_csv(path, rows, FIELDS)
    log = out_path(cfg, "labels", "metadata_log.txt")
    with open(log, "w") as fp:
        fp.write("read %d images, kept %d, dropped %d (|pitch+90| > %.1f)\n" % (len(files), len(rows), len(dropped), tol))
        for r in dropped:
            fp.write("dropped %s pitch=%.2f\n" % (r["file"], r["gimbal_pitch"]))
    print(open(log).read().strip())
    import statistics as st
    for k in ("rel_alt", "gimbal_yaw", "gimbal_pitch", "focal"):
        v = [r[k] for r in rows]
        print("  %-12s min %.3f  median %.3f  max %.3f" % (k, min(v), st.median(v), max(v)))
    print("  models:", sorted({r["model"] for r in rows}))
    print("->", path)


if __name__ == "__main__":
    main()
