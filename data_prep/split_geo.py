"""Geographic split along the flight line (no random split).

Along-track coordinate s (m): projection of the local ENU offset onto the principal axis of the
image centres, oriented so that s grows southward (s = 0 near the northernmost image).

  test  = the n_test southernmost images (GT GPS -> labels/test_gt.csv, used only for evaluation)
  buffer_m = max(buffer_min_m, buffer_factor * footprint_max), footprint_max = longest nominal
             ground side of any image (from RelativeAltitude, no calibration -> conservative)
  candidates = non-test images whose centre is >= buffer_m from every test centre
  val   = candidates with |s - s_vc| <= val_block_m / 2, s_vc = median s of the candidates
  train = candidates >= buffer_m from every val centre
  everything else -> excluded (reason logged)
  Real images whose positive tile has > max_nodata_frac no-data are excluded from train/val
  (they can stay in test: the query is the UAV image).

Pseudo-query regions (centres of tiles cut from the ortho):
  allowed only if distance to every test centre > buffer_m + footprint_max / 2  (= test exclusion)
  pseudo-val   : s in [s_v_lo, s_v_hi]                    (s_v_lo/hi = val block limits)
  pseudo-train : s < s_v_lo - buffer_m  or  s > s_v_hi + buffer_m

Outputs: labels/split.csv, labels/test_gt.csv (flag=edge if the positive tile has > edge_flag_nodata
no-data; the query itself is a full UAV image), labels/regions.json
Run: python -m data_prep.split_geo
"""
import json
import numpy as np

from .common import load_cfg, out_path, load_cameras, read_csv, write_csv
from .footprint import footprint_m
from .georef import Georef


class Regions:
    """Region test for ortho pixel coordinates; built from labels/regions.json."""

    def __init__(self, d, geo):
        self.d, self.geo = d, geo
        self.u = np.array(d["axis_u"])                    # unit vector (east, north), southward
        self.ref = np.array(d["axis_ref_px"])              # pixel where s = 0
        self.test_px = np.array(d["test_centres_px"])      # (n, 2)

    def enu(self, px, py):
        e = (np.asarray(px, float) - self.ref[0]) * self.geo.mpp_x
        n = -(np.asarray(py, float) - self.ref[1]) * self.geo.mpp_y
        return e, n

    def s_of(self, px, py):
        e, n = self.enu(px, py)
        return e * self.u[0] + n * self.u[1]

    def dist_test(self, px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        dx = (px[..., None] - self.test_px[:, 0]) * self.geo.mpp_x
        dy = (py[..., None] - self.test_px[:, 1]) * self.geo.mpp_y
        return np.sqrt(dx ** 2 + dy ** 2).min(-1)

    def label(self, px, py):
        """Array of 0 = none, 1 = pseudo-train, 2 = pseudo-val."""
        d = self.d
        s = self.s_of(px, py)
        ok = self.dist_test(px, py) > d["test_exclusion_m"]
        val = ok & (s >= d["s_val_lo"]) & (s <= d["s_val_hi"])
        tr = ok & ((s < d["s_val_lo"] - d["buffer_m"]) | (s > d["s_val_hi"] + d["buffer_m"]))
        return np.where(val, 2, np.where(tr, 1, 0))


def load_regions(cfg, geo):
    with open(out_path(cfg, "labels", "regions.json")) as f:
        return Regions(json.load(f), geo)


def main():
    cfg = load_cfg()
    cams = load_cameras(cfg)
    geo = Georef.from_png(cfg["paths"]["ortho_png"])
    meta = {r["file"]: r for r in read_csv(out_path(cfg, "labels", "metadata.csv"))}
    pairs = read_csv(out_path(cfg, "labels", "pairs_real.csv"))
    sc = cfg["split"]
    max_nod = cfg["ortho"]["max_nodata_frac"]

    px = np.array([float(p["px_x"]) for p in pairs])
    py = np.array([float(p["px_y"]) for p in pairs])
    fps = []
    for p in pairs:
        m = meta[p["file"]]
        fps.append(max(footprint_m(float(m["rel_alt"]), cams[m["model"]]["sensor_width_mm"], float(m["focal"]),
                                   int(m["w"]), int(m["h"]))))
    footprint_max = max(fps)
    buffer_m = max(sc["buffer_min_m"], sc["buffer_factor"] * footprint_max)

    # principal axis of the centres, oriented southward
    e = (px - px.mean()) * geo.mpp_x
    n = -(py - py.mean()) * geo.mpp_y
    X = np.stack([e, n], 1)
    u = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    if u[1] > 0:
        u = -u
    s_raw = X @ u
    i0 = int(np.argmin(s_raw))
    ref = [float(px[i0]), float(py[i0])]
    s = s_raw - s_raw[i0]

    order = np.argsort(s)
    test_idx = set(order[-sc["n_test"]:].tolist())
    test_px = np.stack([px[list(sorted(test_idx))], py[list(sorted(test_idx))]], 1)

    def dmin(i, idxs):
        if not idxs:
            return np.inf
        j = np.array(sorted(idxs))
        return float(np.min(np.hypot((px[j] - px[i]) * geo.mpp_x, (py[j] - py[i]) * geo.mpp_y)))

    status, reason = {}, {}
    cand = []
    for i in range(len(pairs)):
        if i in test_idx:
            status[i] = "test"
        elif dmin(i, test_idx) < buffer_m:
            status[i], reason[i] = "excluded", "buffer_test (%.0f m < %.0f m)" % (dmin(i, test_idx), buffer_m)
        else:
            cand.append(i)
    s_vc = float(np.median(s[cand]))
    s_v_lo, s_v_hi = s_vc - sc["val_block_m"] / 2, s_vc + sc["val_block_m"] / 2
    val_idx = {i for i in cand if s_v_lo <= s[i] <= s_v_hi}
    for i in cand:
        if i in val_idx:
            status[i] = "val"
        elif dmin(i, val_idx) < buffer_m:
            status[i], reason[i] = "excluded", "buffer_val (%.0f m < %.0f m)" % (dmin(i, val_idx), buffer_m)
        else:
            status[i] = "train"
    lost_buffer = sum(1 for i in reason if reason[i].startswith("buffer"))
    n_train_pool = len(pairs) - len(test_idx) - len(val_idx)
    for i in range(len(pairs)):
        if status[i] in ("train", "val") and float(pairs[i]["nodata"]) > max_nod:
            status[i], reason[i] = "excluded", "nodata %.2f > %.2f" % (float(pairs[i]["nodata"]), max_nod)

    rows = []
    for i, p in enumerate(pairs):
        rows.append({"file": p["file"], "split": status[i], "reason": reason.get(i, ""), "s_m": round(float(s[i]), 2),
                     "px_x": p["px_x"], "px_y": p["px_y"], "side_m": p["side_m"], "nodata": p["nodata"]})
    write_csv(out_path(cfg, "labels", "split.csv"), rows, ["file", "split", "reason", "s_m", "px_x", "px_y", "side_m", "nodata"])
    edge = cfg["ortho"].get("edge_flag_nodata", 0.05)
    gt = [{"file": p["file"], "lat": p["lat"], "lon": p["lon"], "px_x": p["px_x"], "px_y": p["px_y"],
           "side_m": p["side_m"], "nodata": p["nodata"], "flag": "edge" if float(p["nodata"]) > edge else ""}
          for i, p in enumerate(pairs) if status[i] == "test"]
    write_csv(out_path(cfg, "labels", "test_gt.csv"), gt, ["file", "lat", "lon", "px_x", "px_y", "side_m", "nodata", "flag"])

    regions = {
        "axis_u": u.tolist(), "axis_ref_px": ref, "footprint_max_m": footprint_max, "buffer_m": buffer_m,
        "test_exclusion_m": buffer_m + footprint_max / 2, "s_val_lo": s_v_lo, "s_val_hi": s_v_hi,
        "test_centres_px": test_px.tolist(), "s_max_m": float(s.max()),
    }
    with open(out_path(cfg, "labels", "regions.json"), "w") as f:
        json.dump(regions, f, indent=2)

    print("footprint_max %.1f m -> buffer_m %.1f m, test exclusion radius %.1f m" %
          (footprint_max, buffer_m, regions["test_exclusion_m"]))
    print("track length %.0f m; val block s in [%.0f, %.0f] m" % (s.max(), s_v_lo, s_v_hi))
    for k in ("train", "val", "test", "excluded"):
        fs = [r["file"][4:8] for r in rows if r["split"] == k]
        print("%-9s %2d  %s" % (k, len(fs), " ".join(fs)))
    for r in rows:
        if r["reason"]:
            print("   %s  %s" % (r["file"], r["reason"]))
    frac = lost_buffer / max(1, n_train_pool)
    print("lost to buffer: %d of %d non-test/non-val images (%.0f%%)" % (lost_buffer, n_train_pool, 100 * frac))
    if frac > 0.70:
        raise SystemExit("STOP: buffer removes > 70%% of the train images (spec rule 2.4). Ask the user.")


if __name__ == "__main__":
    main()
