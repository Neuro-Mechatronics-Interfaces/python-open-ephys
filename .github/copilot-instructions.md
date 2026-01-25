````instructions
# Python Open-Ephys AI Coding Instructions

Short, actionable guidance for an AI coding agent working on this repo.

1) Big picture
- Purpose: tools to stream/process EMG via the Open-Ephys GUI and run realtime gesture decoding.
- Core components: `OpenEphysClient` (ZMQ client), data preprocessing utilities in `src/old_utils` and `old_utils`, and realtime runner `realtime_decode.py`.

2) How data flows
- Open-Ephys GUI -> ZMQ plugin -> `OpenEphysClient.get_latest_sample()` (samples often shape (channels,) or (channels,samples)).
- `realtime_decode.py` collects samples into a rolling RMS/window buffer, builds feature tensors and calls a PyTorch `EMGCNN` model for inference.

3) Key files to inspect
- [realtime_decode.py](realtime_decode.py) — main realtime pipeline and model-loading logic (uses `torch.load` and expects checkpoint keys such as `conv1.0.weight` to infer input channels).
- [README.md](README.md) — install/run notes, Python 3.10 target, Open-Ephys/ZMQ setup.
- `src/old_utils` & `old_utils` — signal-processing and ML utility functions (preprocessing, `EMGCNN` definition).
- `config.txt` (repo root) — default config path used by example scripts.

4) Developer workflows (concrete commands)
- Create and activate virtualenv (Windows example):
  - `python -m venv .ephys`
  - `call .ephys/Scripts/activate`
- Install deps: `pip install -r requirements.txt`
- Run realtime demo: `python realtime_decode.py --config_path=config.txt`

5) Project-specific conventions & pitfalls
- Channel/order: code assumes channels-first conventions when building tensors for the CNN; check shapes before converting (model expects `(batch, channels, seq_len)` after minor reshaping in `realtime_decode.py`).
- COM ports on Windows use `COMn` (e.g., `COM5`). Serial integration is optional (`--use_serial`).
- Model loading: the script reads a PyTorch checkpoint directly (`torch.load`) and constructs a model with an inferred `input_channels` from checkpoint weights — avoid changing checkpoint key names.
- Performance: realtime loops use `asyncio` + small sleeps; be conservative when changing timing to avoid dropping samples.

6) Integration points & external deps
- Open-Ephys GUI with ZMQ plugin (data source). See Open-Ephys docs and local `README.md` instructions.
- PyTorch for model inference. Check `requirements.txt` for pinned versions.
- Optional serial/Pico integration (PicoMessager) used to output predicted gestures.

7) Tests & examples
- Examples live in repository root and `examples/` — run `realtime_decode.py` to validate end-to-end behavior.
- There is no heavy test harness; validate changes by running the realtime script against a live or simulated Open-Ephys stream.

8) When editing code
- Preserve existing data-shape and checkpoint key assumptions. When changing model I/O, update `realtime_decode.py` where `n_channels` is inferred from the checkpoint key `conv1.0.weight`.
- Update `config.txt` examples if you change expected config keys.

9) If uncertain, inspect these lines
- Model-channel inference: [realtime_decode.py](realtime_decode.py) near `_load_model()` where `checkpoint['conv1.0.weight'].shape[1]` is used.
- Buffer/window logic: [realtime_decode.py](realtime_decode.py) where `deque(maxlen=self.window_size)` and `self.data_buffer` are used.

Please review and tell me if you want this merged into a different path or expanded with more examples (unit tests, CI commands, or exact dependency pins).
````
