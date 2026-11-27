"""Checkpoint selection on pseudo-val ONLY (test is never touched here).

For every <ckpt_dir>/net_XXX.pth: localize.retrieve --model denseuav --queries pseudo_val
(results in results/ckpt_select/<exp>/), then pick the checkpoint with the smallest median error;
ties -> higher share within 25 m. Writes results/<exp>_ckpt_select.md and prints the chosen file.

Run: python -m eval.select_ckpt --exp E0a --ckpt_dir checkpoints/forest_E0a
"""
import argparse
import glob
import os
import subprocess
import sys

from data_prep.common import REPO_ROOT, read_csv
from eval.metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    args = ap.parse_args()
    ckpts = sorted(glob.glob(os.path.join(REPO_ROOT, args.ckpt_dir, "net_*.pth")))
    out_rel = os.path.join("results", "ckpt_select", args.exp)
    rows = []
    for c in ckpts:
        tag = os.path.splitext(os.path.basename(c))[0]
        exp = "%s_%s" % (args.exp, tag)
        csv = os.path.join(REPO_ROOT, out_rel, exp + "__pseudo_val.csv")
        if not os.path.exists(csv):
            subprocess.run([sys.executable, "-m", "localize.retrieve", "--exp", exp, "--model", "denseuav",
                            "--ckpt_dir", args.ckpt_dir, "--checkpoint", os.path.basename(c),
                            "--queries", "pseudo_val", "--out_dir", out_rel, "--no_plot"], cwd=REPO_ROOT, check=True)
        s = summarize(read_csv(csv))
        rows.append((os.path.basename(c), s))
    best = min(rows, key=lambda t: (round(t[1]["median"], 6), -t[1]["<25m"][0]))
    md = ["# Chọn checkpoint %s theo pseudo-val" % args.exp, "",
          "Tiêu chí: median sai số nhỏ nhất trên pseudo-val (n = %d); hoà thì lấy tỉ lệ <25 m cao hơn. "
          "Tập test không được dùng ở bước này." % rows[0][1]["n"], "",
          "| checkpoint | median (m) [95% CI] | mean | <25m | <50m | R@1 | R@5 | SDM@1 | chọn |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        md.append("| %s | %.1f [%.1f–%.1f] | %.1f | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.3f | %s |" % (
            name, s["median"], s["median_lo"], s["median_hi"], s["mean"], 100 * s["<25m"][0], 100 * s["<50m"][0],
            100 * s["R@1"][0], 100 * s["R@5"][0], s["SDM@1"], "**✓**" if name == best[0] else ""))
    path = os.path.join(REPO_ROOT, "results", "%s_ckpt_select.md" % args.exp)
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("SELECTED", best[0])


if __name__ == "__main__":
    main()
