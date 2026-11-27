"""Colour preset C1 (spec section 6) + Gaussian blur, for generating pseudo drone views.

C1: brightness +-0.2, contrast +-0.2, gamma [0.8, 1.2], saturation +-0.1, hue +-0.02,
    white balance (per-channel gain [0.92, 1.08]), random shadow (1-3 polygons, darken 20-50%).
Blur sigma is passed separately (demo uses [0, 1.5], spec C1 says [0, 1.0]).
All parameters actually drawn are returned so they can be logged.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def apply_c1(img, rng, p, blur_sigma):
    x = np.asarray(img.convert("RGB"), np.float32) / 255.0
    rec = {}
    wb = rng.uniform(*p["white_balance"], size=3)
    x = x * wb
    rec["wb"] = np.round(wb, 3).tolist()
    b = 1 + rng.uniform(-p["brightness"], p["brightness"])
    x = x * b
    c = 1 + rng.uniform(-p["contrast"], p["contrast"])
    x = (x - x.mean()) * c + x.mean()
    s = 1 + rng.uniform(-p["saturation"], p["saturation"])
    gray = (x * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1, keepdims=True)
    x = gray + (x - gray) * s
    x = np.clip(x, 0, 1)
    rec.update(brightness=round(b, 3), contrast=round(c, 3), saturation=round(s, 3))

    h = rng.uniform(-p["hue"], p["hue"])
    hsv = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(round(h * 255))) % 256
    x = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"), np.float32) / 255.0
    g = rng.uniform(*p["gamma"])
    x = np.power(np.clip(x, 0, 1), g)
    rec.update(hue=round(h, 4), gamma=round(g, 3))

    H, W = x.shape[:2]
    n_sh = int(rng.integers(p["shadow_polygons"][0], p["shadow_polygons"][1] + 1))
    mask = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(mask)
    for _ in range(n_sh):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        rad = rng.uniform(0.1, 0.35) * W
        k = int(rng.integers(3, 7))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rr = rad * rng.uniform(0.5, 1.0, k)
        dr.polygon([(float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a, r in zip(ang, rr)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(W * 0.02))
    dark = rng.uniform(*p["shadow_darken"])
    m = np.asarray(mask, np.float32)[..., None] / 255.0
    x = x * (1 - dark * m)
    rec.update(shadows=n_sh, shadow_darken=round(dark, 3))

    out = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    if blur_sigma > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur_sigma))
    rec["blur_sigma"] = round(float(blur_sigma), 3)
    return out, rec
