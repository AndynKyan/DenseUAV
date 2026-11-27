"""Statistics for small test sets (spec section 10): Wilson CI, bootstrap CI of the median, McNemar."""
import math

import numpy as np
from scipy.stats import binomtest


def wilson(k, n, z=1.959964):
    """Wilson score interval for a proportion k/n. Returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def bootstrap_ci(values, stat=np.median, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of `stat`. Returns (stat, lo, hi)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = stat(v[idx], axis=1)
    return float(stat(v)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on paired booleans (same queries). Returns (b, c, p_value),
    b = A right & B wrong, c = A wrong & B right."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def bootstrap_diff_median(err_a, err_b, n_boot=10000, alpha=0.05, seed=0):
    """Paired bootstrap (resample queries) of median(err_a) - median(err_b). Returns (diff, lo, hi)."""
    a, b = np.asarray(err_a, float), np.asarray(err_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.median(a) - np.median(b)), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


if __name__ == "__main__":
    print("wilson(7, 14)", wilson(7, 14))
    print("bootstrap median", bootstrap_ci([1, 2, 3, 4, 100]))
    print("mcnemar", mcnemar([1, 1, 1, 0, 1], [0, 0, 1, 0, 1]))
