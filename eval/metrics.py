"""Localization + retrieval metrics (spec section 10) and the results summary.

Per query (computed in localize/retrieve.py with `per_query`):
  err_m            haversine distance between the centre of the top-1 gallery tile and the GT
  rank_first_pos   1-based rank of the first POSITIVE tile (centre < 0.5 * F_gt from GT), -1 if none in top-N
  sdm@K            SDM@K with the exact formula of evaluateDistance.py (euclideanDistance + evaluateSingle,
                   on (lon, lat) in degrees) -- imported, not rewritten
Aggregate (`summarize`): mean / median (bootstrap 95% CI) / P90 error, share within 10/25/50/100 m and
R@1/5/10 with Wilson 95% CI, mean SDM@1/3/5/10.

Run: python -m eval.metrics              -> rebuilds results/summary.md from results/*.csv
     python -m eval.metrics --selftest   -> synthetic check that the metrics are exact
"""
import glob
import os

import numpy as np

from data_prep.common import REPO_ROOT, read_csv
from data_prep.georef import haversine_m
from eval.stats import wilson, bootstrap_ci, bootstrap_diff_median, mcnemar

SDM_K = (1, 3, 5, 10)
THRESH_M = (10, 25, 50, 100)
RECALL_K = (1, 5, 10)


def _sdm(q_lonlat, g_lonlat_topk, K):
    import evaluateDistance as ed            # repo formula, imported on purpose
    return float(ed.evaluateSingle(ed.euclideanDistance(q_lonlat, g_lonlat_topk[:K]), K))


def per_query(order, gt_lat, gt_lon, F_gt, g_lat, g_lon):
    """order: gallery indices sorted by score (desc). Returns dict of per-query metrics."""
    d_all = haversine_m(gt_lat, gt_lon, g_lat[order], g_lon[order])
    pos = np.nonzero(d_all < 0.5 * F_gt)[0]
    out = {"err_m": float(d_all[0]), "rank_first_pos": int(pos[0]) + 1 if len(pos) else -1,
           "n_pos": int(np.sum(haversine_m(gt_lat, gt_lon, g_lat, g_lon) < 0.5 * F_gt))}
    gl = np.stack([g_lon[order[:max(SDM_K)]], g_lat[order[:max(SDM_K)]]], 1)
    for K in SDM_K:
        out["sdm@%d" % K] = _sdm([gt_lon, gt_lat], gl, K)
    return out


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    err = np.array([float(r["err_m"]) for r in rows])
    rk = np.array([int(r["rank_first_pos"]) for r in rows])
    s = {"n": n, "mean": float(err.mean()), "p90": float(np.quantile(err, 0.9))}
    s["median"], s["median_lo"], s["median_hi"] = bootstrap_ci(err)
    for t in THRESH_M:
        s["<%dm" % t] = wilson(int(np.sum(err < t)), n)
    for K in RECALL_K:
        s["R@%d" % K] = wilson(int(np.sum((rk > 0) & (rk <= K))), n)
    for K in SDM_K:
        s["SDM@%d" % K] = float(np.mean([float(r["sdm@%d" % K]) for r in rows]))
    return s


def _fmt_p(t):
    return "%.0f%% [%.0f–%.0f]" % (100 * t[0], 100 * t[1], 100 * t[2])


def table(named_summaries):
    head = ["Thí nghiệm / tập", "n", "median (m) [95% CI]", "mean", "P90"] + ["<%dm" % t for t in THRESH_M] + \
           ["R@%d" % k for k in RECALL_K] + ["SDM@%d" % k for k in SDM_K]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in named_summaries:
        if s["n"] == 0:
            continue
        cells = [name, str(s["n"]), "%.1f [%.1f–%.1f]" % (s["median"], s["median_lo"], s["median_hi"]),
                 "%.1f" % s["mean"], "%.1f" % s["p90"]] + [_fmt_p(s["<%dm" % t]) for t in THRESH_M] + \
                [_fmt_p(s["R@%d" % k]) for k in RECALL_K] + ["%.3f" % s["SDM@%d" % k] for k in SDM_K]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def subsets(name, rows):
    """Report subsets: test -> all + without flag=edge; pseudo_val -> all + pseudo only + real only."""
    out = [(name + " (all)", rows)]
    if any(r.get("flag") for r in rows):
        out.append((name + " (không edge)", [r for r in rows if r.get("flag") != "edge"]))
    kinds = sorted({r.get("kind", "") for r in rows})
    if len(kinds) > 1:
        for k in kinds:
            out.append((name + " (%s)" % k, [r for r in rows if r.get("kind") == k]))
    return out


def main():
    res_dir = os.path.join(REPO_ROOT, "results")
    files = sorted(glob.glob(os.path.join(res_dir, "*.csv")))
    sections = {}
    data = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        rows = read_csv(f)
        data[name] = rows
        qs = name.split("__")[-1]
        sections.setdefault(qs, []).extend((nm, summarize(r)) for nm, r in subsets(name, rows))
    md = ["# Kết quả định vị (demo rừng)", "",
          "Sai số = haversine giữa tâm ô top-1 và GT. Positive = ô có tâm cách GT < 0.5 × F_gt. "
          "SDM@K theo đúng công thức `evaluateDistance.py`. Ngoặc vuông = khoảng tin cậy 95% "
          "(Wilson cho tỉ lệ, bootstrap 10 000 lần cho median).",
          "Lưu ý D11: nhãn GPS lệch ortho tới khoảng 10 m, nên sai số dưới khoảng 10 m không có ý nghĩa.", ""]
    for qs, items in sections.items():
        md += ["## Tập query: %s" % qs, "", table(items), ""]
    # Paired comparisons between real experiments that share a query set:
    # McNemar for R@1, <25 m, <50 m and a paired bootstrap CI of median(err_A) - median(err_B).
    names = [n for n in data if not n.startswith(("ORACLE", "RANDOM"))]
    comp = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.split("__")[-1] != b.split("__")[-1]:
                continue
            for subset_name, keep in (("all", lambda r: True), ("không edge", lambda r: r.get("flag") != "edge")):
                ra = {r["query"]: r for r in data[a] if keep(r)}
                rb = {r["query"]: r for r in data[b] if keep(r)}
                common = sorted(set(ra) & set(rb))
                if not common or (subset_name != "all" and len(common) == len(set(r["query"] for r in data[a]))):
                    continue
                cells = []
                for crit, f in (("R@1", lambda r: int(r["rank_first_pos"]) == 1),
                                ("<25m", lambda r: float(r["err_m"]) < 25), ("<50m", lambda r: float(r["err_m"]) < 50)):
                    bb, cc, p = mcnemar([f(ra[q]) for q in common], [f(rb[q]) for q in common])
                    cells.append("%d/%d, p=%.3g" % (bb, cc, p))
                d, lo, hi = bootstrap_diff_median([float(ra[q]["err_m"]) for q in common],
                                                  [float(rb[q]["err_m"]) for q in common])
                comp.append("| %s | %s | %s | %d | %s | %+.1f [%+.1f, %+.1f] |" %
                            (a, b, subset_name, len(common), " | ".join(cells), d, lo, hi))
    if comp:
        md += ["## So sánh ghép cặp (cùng query)", "",
               "McNemar chính xác: ô ghi `b/c, p`, trong đó b = A đúng B sai, c = A sai B đúng. "
               "Δmedian = median(A) − median(B) (m); âm nghĩa là A tốt hơn. Ngoặc là CI 95% bootstrap ghép cặp, 10 000 lần.", "",
               "| A | B | tập con | n | R@1 | <25m | <50m | Δmedian [95% CI] |",
               "|---|---|---|---|---|---|---|---|"] + comp + [""]
    # Multi-seed: experiments named <base>, <base>_s1, <base>_s2 ... on the same query set
    import re
    groups = {}
    for n in names:
        base, qs = n.split("__")[0], n.split("__")[-1]
        groups.setdefault((re.sub(r"_s\d+$", "", base), qs), []).append(n)
    seed_lines = []
    for (base, qs), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for subset_name, keep in (("all", lambda r: True), ("không edge", lambda r: r.get("flag") != "edge")):
            sm = [summarize([r for r in data[m] if keep(r)]) for m in sorted(members)]
            if sm[0]["n"] == 0 or (subset_name != "all" and sm[0]["n"] == len(data[members[0]])):
                continue
            med = np.array([x["median"] for x in sm])
            p25 = np.array([x["<25m"][0] for x in sm])
            p50 = np.array([x["<50m"][0] for x in sm])
            r1 = np.array([x["R@1"][0] for x in sm])
            seed_lines.append("| %s | %s | %s | %d | %d | %.1f ± %.1f (%s) | %.0f ± %.0f%% | %.0f ± %.0f%% | %.0f ± %.0f%% |" % (
                base, qs, subset_name, len(sm), sm[0]["n"], med.mean(), med.std(ddof=1),
                ", ".join("%.1f" % v for v in med), 100 * p25.mean(), 100 * p25.std(ddof=1),
                100 * p50.mean(), 100 * p50.std(ddof=1), 100 * r1.mean(), 100 * r1.std(ddof=1)))
    if seed_lines:
        md += ["## Nhiều seed (mean ± std, std mẫu)", "",
               "| thí nghiệm | tập | tập con | số seed | n | median err (m) (từng seed) | <25m | <50m | R@1 |",
               "|---|---|---|---|---|---|---|---|---|"] + seed_lines + [""]
    path = os.path.join(res_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print("->", path)


def _selftest():
    """Synthetic check: GT exactly on a tile centre ranked first -> err 0, rank 1, SDM@1 = 1."""
    rng = np.random.default_rng(0)
    g_lat = 10.46 + rng.uniform(0, 0.007, 300)
    g_lon = 106.878 + rng.uniform(0, 0.002, 300)
    for t in range(20):
        i = int(rng.integers(300))
        order = np.r_[i, rng.permutation(np.delete(np.arange(300), i))]
        m = per_query(order, g_lat[i], g_lon[i], 55.0, g_lat, g_lon)
        assert m["err_m"] == 0 and m["rank_first_pos"] == 1 and abs(m["sdm@1"] - 1) < 1e-9, m
    # a far tile first, the exact tile second -> rank 2 and error > 0
    i, j = 0, int(np.argmax(haversine_m(g_lat[0], g_lon[0], g_lat, g_lon)))
    order = np.r_[j, i, np.setdiff1d(np.arange(300), [i, j])]
    m = per_query(order, g_lat[i], g_lon[i], 55.0, g_lat, g_lon)
    assert m["rank_first_pos"] == 2 and m["err_m"] > 100, m
    print("eval.metrics self-test OK (err 0 / rank 1 / SDM@1 = 1 on exact tiles; rank 2 when shifted)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
