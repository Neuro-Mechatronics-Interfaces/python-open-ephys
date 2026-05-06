![Logo](https://raw.githubusercontent.com/Neuro-Mechatronics-Interfaces/python-open-ephys/main/docs/figs/logo.jpg)

# python-open-ephys

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://neuro-mechatronics-interfaces.github.io/python-open-ephys/)
[![Tests](https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/actions/workflows/test.yml/badge.svg)](https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/python-oephys.svg)](https://pypi.org/project/python-oephys/)
[![Python](https://img.shields.io/pypi/pyversions/python-oephys.svg)](https://pypi.org/project/python-oephys/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

python-open-ephys is a Python toolkit for loading, streaming, processing, and visualizing Open Ephys electrophysiology data. The project combines file I/O, real-time interfaces, signal-processing utilities, and higher-level application examples in one package.

## Highlights

- Load Open Ephys Binary recordings and normalized NPZ exports.
- Stream live data from the Open Ephys GUI over ZMQ and LSL.
- Process EMG and electrophysiology signals with filtering, synchronization, and QC utilities.
- Run GUI tools for viewing recordings and inspecting real-time streams.
- Prototype machine-learning workflows on top of the same session format.

## Installation

Install the latest published release from PyPI:

```bash
pip install python-oephys
```

The published package currently includes the runtime GUI and ML dependencies in the base install. Install the docs toolchain separately if you need to build the documentation locally:

```bash
pip install "python-oephys[docs]"
```

Install from source for development:

```bash
git clone https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys.git
cd python-open-ephys
pip install -e .
```

Install a pre-release build from TestPyPI when you want to validate an upcoming release candidate:

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys
```

## Quick Start

```python
from pyoephys.io import load_open_ephys_session
from pyoephys.processing import bandpass_filter

session = load_open_ephys_session("path/to/recording.oebin")
amplifier_data = session["amplifier_data"]
sample_rate = session["sample_rate"]

filtered = bandpass_filter(
    amplifier_data,
    lowcut=10,
    highcut=450,
    fs=sample_rate,
)
```

Launch the real-time viewer against an Open Ephys ZMQ stream:

```bash
python -m pyoephys.applications._realtime_viewer --host 127.0.0.1 --channels 0:8
```

## Examples

The repository includes runnable examples for common workflows:

- `examples/read_files/` for loading `.oebin` recordings and exporting session data.
- `examples/interface/` for ZMQ, LSL, IMU, and hardware integration workflows.
- `examples/applications/` for standalone viewers and analysis tools.
- `examples/analysis/`, `examples/benchmarks/`, and `examples/visualization/` for downstream analysis and diagnostics.

## Documentation

Hosted documentation: https://neuro-mechatronics-interfaces.github.io/python-open-ephys/

Build the docs locally with:

```bash
python -m sphinx -b html docs/source docs/build/html
```

## Development

Run the test suite:

```bash
pytest tests/
```

Build source and wheel distributions:

```bash
python -m build
```

## Release Workflow

- Pushes to `main` publish the Sphinx site to GitHub Pages.
- The `Test Publish to TestPyPI` workflow is available for manual pre-release validation.
- Publishing a GitHub Release triggers the PyPI publish workflow.

## Contributing

Issues and pull requests are welcome. If you are proposing a new feature, include the target workflow, example data format, and any GUI or hardware assumptions so the change can be tested cleanly.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
