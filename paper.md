---
title: 'python-oephys: A Unified Python Toolkit for Real-Time Open Ephys Data Processing and Machine Learning'
tags:
  - Python
  - electrophysiology
  - Open Ephys
  - EMG
  - real-time
  - machine learning
authors:
  - name: Jonathan Shulgach
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Neuromechatronics Lab, Carnegie Mellon University, Pittsburgh, PA, USA
    index: 1
date: 26 January 2026
bibliography: paper.bib
---

# Summary

`python-oephys` is an open-source Python library designed to facilitate the acquisition, processing, and analysis of high-density electrophysiology data, specifically targeting the Open Ephys ecosystem. It provides a unified interface for both offline analysis of binary recordings and real-time streaming via ZeroMQ (ZMQ). Key capabilities include modular signal processing pipelines, automated channel quality assessment, and integrated machine learning models (CNN-LSTM) optimized for low-latency gesture classification.

# Statement of need

The Open Ephys GUI is a widely used platform for neural data acquisition, but researchers often face significant friction when transitioning from raw data acquisition to real-time closed-loop control or advanced offline analysis. Existing tools often handle either file I/O or real-time streaming, but rarely both in a unified, high-performance package. 

`python-oephys` satisfies this need by providing:
1. **Unified I/O**: A consistent API for both `.oebin` and `.npz` formats.
2. **Real-time Integration**: Low-latency ZMQ clients that allow Python scripts and GUI applications to react to live neural streams.
3. **ML-Ready Pipelines**: Pre-integrated deep learning architectures tailored for spatio-temporal neural signals, reducing the time required to build predictive models for BCIs or myoelectric control.

# State of the field

Similar tools like `LibEMG` focus on general myoelectric control, while the official `open-ephys-python-tools` provide basic file reading. `python-oephys` bridges these domains by specializing in the high-density spatial configurations typical of Open Ephys hardware while providing the real-time application layer (viewers and decoders) missing from low-level I/O libraries.

# Key Features

- **Real-time ZMQ Client**: Handles multi-channel streaming with minimal overhead.
- **Channel QC**: Implements automated noise floor and saturation detection.
- **EMGClassifierCNNLSTM**: A hybrid neural network for spatio-temporal feature extraction.
- **Applications**: Includes desktop GUIs for data visualization and manual trial segmentation.

# Acknowledgements

This work was supported by the Neuromechatronics Lab at Carnegie Mellon University.

# References
