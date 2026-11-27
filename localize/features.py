"""Descriptor extraction, same recipe as test.py:122 extract_feature (which cannot be imported:
test.py runs argparse and loads a model at module level).

  for each batch: f = branch(img) + branch(fliplr(img)); f = f / ||f||_2
  - DenseUAV models: branch(x) = model(x, None)[0][1]  (outputs[1] = ClassBlock bottleneck feature)
    3-D outputs (LPN etc.) are normalised per part and divided by sqrt(opt.block) like test.py.
  - Z0: branch(x) = mean of the patch tokens of timm vit_small_patch16_224 (ImageNet), no head.
  A batch of size 1 is padded with a copy (SingleBranch calls .squeeze(), which would drop the batch dim).
Preprocessing = test.py:74-85: Resize((h, w), bicubic), ToTensor, Normalize(ImageNet).
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from tool.utils import device

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_transform(h=224, w=224):
    return transforms.Compose([transforms.Resize((h, w), interpolation=3), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class _Paths(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def fliplr(img):
    return img.flip(3)


class Extractor:
    """Wraps a model as `branch(x) -> (B, D) or (B, D, P)` plus a cache key."""

    def __init__(self, branch, key, h=224, w=224, block=1):
        self.branch, self.key, self.h, self.w, self.block = branch, key, h, w, block

    @torch.no_grad()
    def __call__(self, paths, batch_size=64, num_workers=4):
        dl = torch.utils.data.DataLoader(_Paths(paths, make_transform(self.h, self.w)), batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
        feats = []
        for img in dl:
            n = img.shape[0]
            if n == 1:
                img = torch.cat([img, img], 0)
            ff = None
            for i in range(2):
                x = (fliplr(img) if i == 1 else img).to(device)
                out = self.branch(x).float()
                ff = out if ff is None else ff + out
            if ff.dim() == 3:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(self.block)
                ff = (ff / fnorm).reshape(ff.shape[0], -1)
            else:
                ff = ff / torch.norm(ff, p=2, dim=1, keepdim=True)
            feats.append(ff[:n].cpu())
        return torch.cat(feats, 0).numpy().astype(np.float32)


def z0_extractor(pool="mean"):
    import timm
    m = timm.create_model("vit_small_patch16_224", pretrained=True, img_size=(224, 224)).eval().to(device)
    tag = m.pretrained_cfg.get("tag", "") if hasattr(m, "pretrained_cfg") else ""

    def branch(x):
        t = m.forward_features(x)
        return t[:, 1:].mean(1) if pool == "mean" else t[:, 0]
    return Extractor(branch, "z0|vit_small_patch16_224.%s|%s" % (tag, pool))


def denseuav_extractor(ckpt_dir, checkpoint):
    """Model trained with train.py: reads <ckpt_dir>/opts.yaml, loads <ckpt_dir>/<checkpoint>."""
    from models.taskflow import make_model
    with open(os.path.join(ckpt_dir, "opts.yaml")) as f:
        opt = SimpleNamespace(**yaml.safe_load(f))
    opt.load_from = ""                     # do not re-load any pre-training checkpoint
    model = make_model(opt)
    path = os.path.join(ckpt_dir, checkpoint)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model = model.eval().to(device)
    h = hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]

    def branch(x):
        return model(x, None)[0][1]
    return Extractor(branch, "denseuav|%s|%s" % (os.path.abspath(path), h), opt.h, opt.w, getattr(opt, "block", 1))
