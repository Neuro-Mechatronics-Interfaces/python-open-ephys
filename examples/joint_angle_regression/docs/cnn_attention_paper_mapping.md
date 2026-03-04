# CNN-Attention paper mapping for joint-angle regression

Source paper:
- `A_CNN-Attention_Network_for_Continuous_Estimation_of_Finger_Kinematics_from_Surface_Electromyography.pdf`

Text extraction snapshot:
- `_cnn_attention_paper_text.txt`

## What the paper does (key details)

## Dataset and preprocessing
- 12-channel sEMG, 2 kHz sampling.
- Finger kinematics from glove sampled at 20 Hz, resampled to 2 kHz for alignment.
- EMG preprocessing: bandpass + notch filtering.
- Sliding window segmentation: 100 ms window, 0.5 ms step.
- Feature used for model input: RMS feature sequence.
- Targets: 10 finger joint angles, using the last angle value in each window.

## CNN-Attention architecture
- Two multi-scale convolution modules.
- Each module has 3 parallel Conv paths with kernel sizes: 3, 5, 7.
- Uses padding to keep sequence length.
- Average pooling after first multi-scale module.
- Attention stack:
  - 3 stacked multi-head self-attention blocks.
  - 3 heads per block.
  - Attention dims reported: dq=64, dk=64, dv=128.
  - Positional encoding added before attention.
- Regression output: linear layer to 10 joint angles.

## Reported outcomes
- Metrics: CC, RMSE (deg), R2.
- CNN-Attention average (reported): CC=0.87, RMSE=9.65°, R2=0.73.
- Better than LSTM and SPGP in their setup.
- Lower training cost than LSTM:
  - ~100 epochs to converge vs ~162 for LSTM.
  - ~43 min vs ~58 min average total training time.
- Paper discusses parameter count: CNN-Attention ~224,010 vs LSTM ~602,378.

## Current repo model vs paper (gap analysis)

File: `scripts/train_cnn_attention_regressor.py`

Current model (repo):
- Input channels: 8 EMG channels (+ optional IMU branch).
- Conv front-end: sequential Conv1D(64, k=5) then Conv1D(128, k=3).
- Single MHA block: heads=4, key_dim=32.
- No explicit sinusoidal positional encoding layer.
- Feed-forward + global average pooling + dense regression head.

Main differences from paper:
1. Multi-scale conv branch (3/5/7 kernels) is not implemented.
2. Two multi-scale conv modules are not implemented.
3. Paper-style stacked attention depth (3 blocks) is not implemented.
4. Positional encoding before attention is not implemented.
5. Input feature design differs (paper uses long-exposure RMS framing).
6. Channel count differs (paper 12 vs current 8).

## Practical implementation plan in this repo

1) Add a paper-faithful model option (`--arch paper_msattn`)
- Keep existing model as default for backward compatibility.
- New branch in `_build_model`:
  - Block A: multi-scale Conv1D paths (k=3/5/7) + concat + BN + activation.
  - AvgPool after block A.
  - Block B: second multi-scale Conv1D paths (k=3/5/7).
  - Positional encoding layer.
  - 3 stacked transformer-style self-attention blocks (3 heads each).
  - Regression head.

2) Add RMS long-exposure input mode (`--input_mode rms_longexp`)
- Build subframe RMS sequence similar to paper windows.
- Keep current raw/log1p mode as alternate baseline.

3) Add strict benchmark protocol
- Evaluate both architectures under same split and smoothing policy.
- Log in experiment registry with unique fields:
  - `arch`, `input_mode`, `msconv_kernels`, `attn_blocks`, `attn_heads`, `posenc`.

4) Add targeted sweeps for R2
- Sweep: attention depth {1,2,3}, heads {2,3,4}, FF dim {128,256}, dropout {0.0,0.1,0.2}.
- Sweep smoothing around proven best region (20–35 samples).

## Recommended first run to target higher R2
- Start with:
  - `arch=paper_msattn`
  - `input_mode=rms_longexp`
  - `attn_blocks=3`, `attn_heads=3`
  - `msconv_kernels=3,5,7`
  - moderate dropout (`0.1`)
- Compare against current tuned CNN baseline (`eval_cnn_attention_001to005_tuned`) using the same held-out session-6 protocol.

## Notes
- Absolute metric parity with the paper is unlikely due to different dataset composition and channel setup.
- The strongest immediate lever in this repo so far remains temporal smoothing; combine architecture changes with smoothing and registry logging.
