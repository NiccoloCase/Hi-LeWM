# Decoder Probe Architecture

This note explains the architecture and training setup of the frozen HOPE2 decoder probe that was added to the repo.

Relevant implementation files:

- [hi_decoder_probe.py](/gpfs/home2/scur0200/main/hi_decoder_probe.py:1)
- [hi_train_decoder_probe.py](/gpfs/home2/scur0200/main/hi_train_decoder_probe.py:1)
- [hi_decoder_probe_eval.py](/gpfs/home2/scur0200/main/hi_decoder_probe_eval.py:1)
- [config/train/hi_decoder_probe.yaml](/gpfs/home2/scur0200/main/config/train/hi_decoder_probe.yaml:1)

## Goal

The probe is not part of world-model training. Its purpose is to answer:

1. Is the frozen HOPE2 waypoint latent visually decodable at all?
2. Do predicted waypoint latents stay on the same decodable manifold as true waypoint latents?
3. When high-level predictions fail, do they fail in a visually obvious way?

The probe therefore trains a separate latent-to-pixel decoder while keeping the full HOPE2 checkpoint frozen.

## High-Level Structure

The probe system has 2 components:

1. A frozen HOPE2 model, used only to produce waypoint latents and predicted waypoint latents.
2. A trainable decoder, used to reconstruct image-space waypoint observations from those latents.

The HOPE2 side is loaded from an object checkpoint and is never updated during probe training.

## Frozen HOPE2 Path

The frozen model provides:

- the image encoder and projector, which turn waypoint frames into latent waypoint embeddings
- the latent-action encoder, which compresses primitive action chunks between waypoints into macro-actions
- the high-level predictor, which predicts the next waypoint latent from context waypoint latents and macro-actions

For the current default probe config:

- `history_size = 3`
- `waypoints.num = 4`
- `waypoints.strategy = fixed_stride`
- `waypoints.stride = 5`

So each training sample uses 4 waypoint frames and 3 waypoint transitions.

## Decoder Architecture

The decoder is intentionally lightweight. It takes a single projected waypoint latent and reconstructs a `224 x 224` RGB image.

Current default architecture:

- input latent: `D` from HOPE2 projected waypoint space
- image size: `224`
- patch size: `16`
- patch grid: `14 x 14 = 196` patch queries
- hidden dimension: `256`
- decoder depth: `4`
- attention heads: `4`
- MLP ratio: `4.0`

Implementation-wise:

1. The latent is projected with a linear layer into the decoder hidden dimension.
2. A learned bank of `196` query tokens is used, one per image patch.
3. Learned positional embeddings are added to these query tokens.
4. Each decoder block applies:
   - self-attention over the patch queries
   - cross-attention from the patch queries to the single latent memory token
   - an MLP block
5. The final patch embeddings are projected to `16 x 16 x 3` pixel patches.
6. The patches are folded back into a full image.

This is a query-based latent decoder, not a spatial feature decoder. That is important because HOPE2 exposes a single global latent per waypoint, not a spatial token map.

## Why The Decoder Is Small

The first version of the plan used a heavier decoder. That was simplified because this is a probe, not a production image model.

The smaller default was chosen because:

- the target question is qualitative sanity-checking, not photorealistic reconstruction
- the conditioning signal is a single global latent
- training and debugging are easier with a smaller decoder
- Phase A already converged to low validation MSE and good PSNR with this smaller setting

If later probe quality is clearly bottlenecked by decoder capacity, the simplest scaling path is:

- raise `hidden_dim`
- then raise `depth`
- only then consider larger attention width

## Training Phases

The decoder is trained in 2 phases.

### Phase A: `true_only`

In Phase A, the decoder only sees true HOPE2 waypoint latents:

- sample waypoint frames from the dataset
- encode those frames with frozen HOPE2
- decode the resulting waypoint latents back to pixels
- optimize pixel reconstruction MSE

This isolates whether the latent itself is decodable.

### Phase B: `pred_exposed`

In Phase B, the decoder is initialized from Phase A and additionally sees predicted waypoint latents:

- build macro-actions from the primitive action chunks between waypoints
- run frozen HOPE2 high-level prediction
- decode both true waypoint latents and predicted waypoint latents
- train on a combined loss

Default weighting:

- true-latent reconstruction weight: `1.0`
- predicted-latent reconstruction weight: `0.5`

This phase is not meant to improve HOPE2. It is meant to make the decoder robust to the distribution shift between encoded true latents and model-predicted latents.

## Data Path And Efficiency

The probe reuses the same frozen-P2 data optimization used by the hierarchical training code:

- only sampled waypoint frames are pixel-preprocessed
- full trajectories are not image-preprocessed when only waypoints are needed
- dataset and checkpoint are copied to node-local storage on Snellius scratch-node jobs

This matters because the probe is mostly I/O-bound during image loading and should not waste work on unused frames.

## Losses And Metrics

Training loss:

- normalized-image pixel MSE

Logged diagnostics:

- `train/loss`, `val/loss`
- `train/pixel_mse`, `val/pixel_mse`
- `val/psnr`
- latent norm statistics
- predicted-vs-true latent gap
- comparison panels with context frame, target frame, decoded true latent, and decoded predicted latent

Important caveat:

- the current SSIM logging is not reliable and should not be used to judge model quality yet

## Artifacts

Each epoch writes:

- a probe bundle `..._epoch_<N>_probe.pt`
- a Lightning checkpoint `..._epoch_epoch=<N>.ckpt`
- the latest rolling probe bundle `..._probe.pt`
- the latest rolling training checkpoint `last.ckpt`
- validation image panels under `visuals/`

This makes it easy to stop Phase A early, inspect results, and start Phase B from a specific epoch.

## Interpretation

The probe should be interpreted as a diagnostic layer over HOPE2, not as part of the HOPE2 model itself.

Good Phase A reconstructions mean:

- the frozen waypoint latent still contains substantial scene information

Good Phase B reconstructions mean:

- predicted waypoint latents are still close enough to the true-latent manifold that a single decoder can visualize them cleanly

If Phase A is good but Phase B looks bad, that points more toward high-level predictor drift than decoder weakness.
