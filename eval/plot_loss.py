"""Plot per-epoch losses (total, CE, triplet, KL) and train accuracies from a train.py log.

Run: python -m eval.plot_loss --log checkpoints/forest_E0a/train.log --out results/E0a_loss.png
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"Loss: ([\d.]+) Cls_Loss:([\d.]+) KL_Loss:([\d.]+) Triplet_Loss ([\d.]+) "
                 r"Satellite_Acc: ([\d.]+)\s+Drone_Acc: ([\d.]+) lr_backbone:([\d.]+) lr_other ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    vals = [tuple(map(float, m.groups())) for m in PAT.finditer(open(args.log).read())]
    if not vals:
        raise SystemExit("no epoch lines found in %s" % args.log)
    ep = list(range(len(vals)))
    col = list(zip(*vals))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, name in ((0, "total"), (1, "CE (sat + drone)"), (3, "WeightedSoftTriplet"), (2, "KL")):
        ax[0].plot(ep, col[i], label=name)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss / sample"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Loss")
    ax[1].plot(ep, col[4], label="satellite acc"); ax[1].plot(ep, col[5], label="drone acc")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.05); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("Train classification accuracy")
    ax[2].plot(ep, col[6], label="lr backbone"); ax[2].plot(ep, col[7], label="lr head")
    ax[2].set_yscale("log"); ax[2].set_xlabel("epoch"); ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_title("LR")
    fig.suptitle(args.title or args.log)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("%d epochs -> %s" % (len(vals), args.out))


if __name__ == "__main__":
    main()
