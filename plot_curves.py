"""
Download training curves from wandb and draw the report figures.

    python plot_curves.py --loss_run <run_id> --fid_run <run_id>

Writes plots/loss_1k.png (Q6.1) and plots/fid_1k.png (Q6.2).
The raw per-step loss is noisy because every step uses random timesteps,
so a moving average is drawn on top of it.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY_PROJECT = "fahad00888-itu-lahore/DDPM_AFHQ"


def history(run_id, key):
    run = wandb.Api().run(f"{ENTITY_PROJECT}/{run_id}")
    rows = [r for r in run.scan_history(keys=["_step", key]) if r.get(key) is not None]
    steps = np.array([r["_step"] for r in rows]) + 1          # 1-based training step
    values = np.array([r[key] for r in rows], dtype=float)
    return steps, values


def moving_average(v, k):
    k = max(1, min(k, len(v)))
    return np.convolve(v, np.ones(k) / k, mode="valid")


def plot_loss(run_id, out, window=25):
    steps, loss = history(run_id, "loss")
    plt.figure(figsize=(6.4, 3.8))
    plt.plot(steps, loss, color="0.75", linewidth=0.8, label="per-step L1 loss")
    ma = moving_average(loss, window)
    plt.plot(steps[window - 1:], ma, color="C0", linewidth=2, label=f"moving average ({window} steps)")
    plt.xlabel("training step")
    plt.ylabel("L1 noise-prediction loss")
    plt.title("DDPM training loss (AFHQ cat, 1,000 steps)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"saved {out}  (first {loss[0]:.3f}, last {loss[-1]:.3f}, final avg {ma[-1]:.3f})")


def plot_fid(run_id, out):
    steps, fid = history(run_id, "fid")
    plt.figure(figsize=(6.4, 3.8))
    plt.plot(steps, fid, marker="o", color="C3")
    for s, f in zip(steps, fid):
        plt.annotate(f"{f:.1f}", (s, f), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    plt.xlabel("training step")
    plt.ylabel("FID (lower is better)")
    plt.title("FID of 512 samples vs. real cat images")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"saved {out}")
    for s, f in zip(steps, fid):
        print(f"  step {s:5d}: FID {f:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_run", required=True, help="wandb run id of the 1k-step run (Q6.1)")
    ap.add_argument("--fid_run", help="wandb run id of the 1k-step FID run (Q6.2)")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_loss(args.loss_run, os.path.join(args.out_dir, "loss_1k.png"))
    if args.fid_run:
        plot_fid(args.fid_run, os.path.join(args.out_dir, "fid_1k.png"))
