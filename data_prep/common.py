"""Shared helpers for data_prep: config loading, output paths, hidden-file cleanup."""
import os
import csv
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path="configs/data.yaml"):
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = os.path.expanduser(v)
        if not os.path.isabs(p):
            p = os.path.join(REPO_ROOT, p)
        cfg["paths"][k] = p
    return cfg


def out_path(cfg, *parts):
    p = os.path.join(cfg["paths"]["out_root"], *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def load_cameras(cfg):
    with open(cfg["paths"]["cameras"]) as f:
        return yaml.safe_load(f)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def remove_hidden(root):
    """Delete hidden files/dirs (.DS_Store, ._*, ...) under root. Returns number removed."""
    import shutil
    n = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for d in [d for d in dirnames if d.startswith(".")]:
            shutil.rmtree(os.path.join(dirpath, d))
            dirnames.remove(d)
            n += 1
        for fn in filenames:
            if fn.startswith("."):
                os.remove(os.path.join(dirpath, fn))
                n += 1
    return n
