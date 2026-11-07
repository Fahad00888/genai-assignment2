# Experiment log

All runs: AFHQ cat (5,153 images), 32x32, T=50 cosine schedule, U-Net dim 16 mults [1,2,4,8],
batch 32 x 2 grad-accumulation, Adam lr 1e-3. Hardware: MacBook Pro M5 (Apple GPU / MPS).

| Run | Date | Settings | Time | Final loss | wandb |
|---|---|---|---|---|---|
| smoke | 2026-11-07 | 20 steps, offline (pipeline check) | 39 s | 0.492 | offline |
| run_1k | 2026-11-07 | 1000 steps, sample every 100, fid=False (Q6.1) | 5m23s | 0.182 | https://forge.coreweave.com/wandb/fahad00888-itu-lahore/DDPM_AFHQ/runs/7s0eri6w |
| run_10k | 2026-11-07 | 10000 steps, sample every 1000, fid=False (Q6.3) | 31m30s | 0.159 | https://forge.coreweave.com/wandb/fahad00888-itu-lahore/DDPM_AFHQ/runs/ppg6l1ru |

## Notes
- run_1k: loss fell quickly from ~0.5 to ~0.2 in the first few hundred steps, then flattened with noise.
- run_1k samples at step 1000: Recognisable but blurry cat faces, centred in each 32x32 frame. Ear shapes and eye positions are visible in most samples, but fine detail (whiskers, fur texture, sharp eye outlines) is smeared. Colours lean towards grey, white and black with a slight blue tint, and orange/ginger cats are rare. Fur patterns still vary (dark, white, tabby, two-tone), so no sign of mode collapse yet.
- run_10k: loss dropped only modestly after 1k steps (0.182 -> 0.159) and stays noisy, because each step sees a random timestep and the L1 noise error differs a lot across t.
- run_10k samples at step 10000: clearly better than at 1k. Faces are sharper, with distinct eyes, noses and pointed ears, and backgrounds separate from the cat. The blue tint is gone and colours are natural: orange and grey tabbies, black-and-white tuxedo cats, white cats and Siamese-like faces all appear, so diversity is good. Remaining flaws: soft fur texture and occasional asymmetric or smudged eyes.
