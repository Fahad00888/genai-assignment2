# Assignment 2 – Generative Models of Images (DDPM on AFHQ cats)

Generative AI & LLM, Fall 2026, ITU · Fahad Khalid

Implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on the cat class of AFHQ at 32×32, plus the written answers in `Report.pdf`.

- **Report:** [`Report.pdf`](Report.pdf) (LaTeX source: `report.tex`)
- **Trained 10k-step model:** [Google Drive – model.pt](https://drive.google.com/file/d/1W4vOGj6Rkms7JzoKhYAFHsiS_bTbB12t/view?usp=sharing)
- **Experiment tracking (wandb):** https://forge.coreweave.com/wandb/fahad00888-itu-lahore/DDPM_AFHQ
- **Run log:** [`experiments.md`](experiments.md)

## Results

| Question | Result | File |
|---|---|---|
| 6.1 Training loss (1k steps) | 0.85 → 0.18 | `plots/loss_1k.png` |
| 6.2 FID every 100 steps (1k steps) | 279.3 → **152.6** | `plots/fid_1k.png` |
| 6.3 Samples after 10k steps | final loss 0.159 | `plots/samples_10k.png` |
| 6.4 Forward diffusion (0/25/50/75/99 %) | | `plots/forward_diffusion.png` |
| 6.5 Backward diffusion (0/25/50/75/99 %) | | `plots/backward_diffusion.png` |

All runs were trained locally on a MacBook Pro (Apple M5, PyTorch MPS backend). The 10k-step run took 31.5 minutes.

## Repository layout

```
data/                AFHQ subset (train/{cat,dog,wild}, val/{cat,dog,wild}); only cat is used
diffusion.py         DDPM: cosine schedule, q_sample, L1 p_losses, forward, p_sample, p_sample_loop, sample
unet.py              U-Net noise predictor (forward pass implemented)
trainer.py           training loop, sampling, FID, forward/backward visualisation
utils.py             setup / train / visualise entry points (wandb)
main.py              command-line entry point
test_diffusion.py    unit tests (need data.pt from the course staff)
plot_curves.py       downloads loss/FID curves from wandb and draws the report plots
plots/               figures used in the report
experiments.md       log of every run (settings, time, loss, wandb link, observations)
report.tex           report source
requirements.txt     Python dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib        # only for plot_curves.py
wandb login
```

`scipy<1.16` is pinned because clean-fid calls `scipy.linalg.sqrtm(..., disp=False)`, which was removed in newer scipy.

## Reproducing the experiments

On macOS use `--dataloader_workers 0` (the dataset uses a lambda transform that cannot be pickled for worker processes on macOS). On Linux/Colab the default works.

```bash
# Q6.1 – training loss
python main.py --train_steps 1000 --save_and_sample_every 100 --save_folder ./results/run_1k --dataloader_workers 0

# Q6.2 – FID every 100 steps
python main.py --train_steps 1000 --save_and_sample_every 100 --fid --save_folder ./results/run_1k_fid --dataloader_workers 0

# Q6.3 – full 10k-step training
python main.py --train_steps 10000 --save_and_sample_every 1000 --save_folder ./results/run_10k --dataloader_workers 0

# Q6.4 / Q6.5 – forward and backward visualisation (uses the 10k model)
python main.py --visualize --load_path ./results/run_10k/model.pt --save_folder ./results/run_10k_vis \
    --train_steps 10000 --save_and_sample_every 1000 --dataloader_workers 0

# Report plots (loss and FID) from the wandb runs
python plot_curves.py --loss_run 7s0eri6w --fid_run p1sjl5gm
```

To use the provided checkpoint instead of training, download `model.pt` from the Drive link above into `results/run_10k/` and run only the visualisation command.

Unit tests (requires `data.pt` in this folder):
```bash
python test_diffusion.py
```

## Changes to the provided starter code

Besides implementing the TODOs in `diffusion.py` and `unet.py`, the following fixes were made (each is its own commit):

1. **`main.py`**: `--visualize` passed the argument dict positionally; it is now unpacked with `**`.
2. **`trainer.py`, FID on CPU**: clean-fid defaults to CUDA; FID now runs on CPU so it works on Apple Silicon.
3. **`trainer.py`, cat-only FID reference**: the reference folder was `data/val`, which mixes cat, dog and wild images; it now uses `data/val/<data_class>` (500 real cats).
4. **`trainer.py`, backward visualisation start**: the reverse process started from the forward image after it had been clamped and rescaled to [0, 1] for display; it now starts from the raw noisy sample in model space [-1, 1].
5. **`trainer.py`, black images on MPS**: images logged to wandb were completely black when tensors were on the Apple GPU; they are now moved to CPU first, and the forward/backward figures are also saved to `results/<run>/`.
6. **`requirements.txt`**: added (it was listed in the handout but missing from the starter zip), with `scipy<1.16` pinned.
