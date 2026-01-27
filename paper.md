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
    orcid: 0009-0004-0449-9918
    affiliation: 1
affiliations:
  - name: Department of Mechanical Engineering, Carnegie Mellon University, Pittsburgh, PA, USA
    index: 1
date: 26 January 2026
bibliography: paper.bib
---

# Summary

`python-oephys` is an open-source Python library designed to facilitate the acquisition, processing, and analysis of high-density electrophysiology data, specifically targeting the Open Ephys ecosystem. It provides a unified interface for both offline analysis of binary recordings and real-time streaming via ZeroMQ [@pyzmq]. Key capabilities include modular signal processing pipelines, automated channel quality assessment (QC), and integrated machine learning models (CNN-LSTM) optimized for low-latency gesture classification.

# Statement of need

The Open Ephys GUI [@siegle2017open] is a widely used platform for neural data acquisition, but researchers often face significant friction when transitioning from raw data acquisition to real-time closed-loop control or advanced offline analysis. Existing tools often handle either file I/O or real-time streaming, but rarely both in a unified, high-performance package. 

`python-oephys` satisfies this need by providing:
1. **Unified I/O**: A consistent API for both `.oebin` and `.npz` formats.
2. **Real-time Integration**: Low-latency ZeroMQ [@pyzmq] clients that allow Python scripts and GUI applications to react to live neural streams.
3. **ML-Ready Pipelines**: Pre-integrated deep learning architectures tailored for spatio-temporal neural signals, reducing the time required to build predictive models for BCIs or myoelectric control.

# State of the field

The field of neural data analysis is supported by several specialized tools. The official `open-ephys-python-tools` [@OpenEphysPythonTools] provide basic file loading capabilities but lack high-level processing or real-time application layers. In the domain of myoelectric control, libraries like `LibEMG` [@Campbell2022] offer comprehensive pipelines but are often decoupled from the specific streaming protocols used by hardware like Open Ephys.

`python-oephys` bridges these domains by specializing in the high-density spatial configurations typical of Open Ephys hardware while providing the real-time application layer (viewers and decoders) missing from low-level I/O libraries. It leverages the scientific Python stack, including NumPy [@harris2020array] and SciPy [@virtanen2020scipy], and relies on high-performance visualization frameworks like PyQt5 [@PyQt5] and pyqtgraph [@pyqtgraph] to provide the responsive interfaces necessary for real-time neural data monitoring.

# Software Design

`python-oephys` is designed with a modular architecture that separates data acquisition, processing, and visualization (see Figure 1).

![EMG Processing Pipeline. A) Raw signals from five representative channels (20–30s). B) Signals after CAR, bandpass (20-500Hz), and 60Hz notch filtering. C) Automated channel quality indicators evaluated on the first 5s of data. D) Mean RMS features extracted from the processed segment.](docs/figs/pipeline.png)

- **Interface Layer**: Implements ZeroMQ [@pyzmq] and LSL [@LSL] clients for low-latency data streaming. The `ZMQClient` is designed to run asynchronously, ensuring that data acquisition does not block processing or UI updates.
- **Processing Layer**: Provides a suite of filters and feature extraction tools. This includes the `EMGPreprocessor` for standardized filtering and `ChannelQC` for real-time signal quality monitoring.
- **ML Layer**: Integrated with PyTorch [@paszke2019pytorch] and scikit-learn [@pedregosa2011scikit], this layer provides pre-configured models like `EMGClassifierCNNLSTM`. These models are designed to handle the variable channel counts and sampling rates common in high-density recordings.
- **Visualization Layer**: Built on PyQt5 [@PyQt5] and pyqtgraph [@pyqtgraph], providing high-frame-rate real-time plots and interactive offline analysis tools.

# Research Impact

`python-oephys` is a foundational component of the research infrastructure at the Neuromechatronics Lab at Carnegie Mellon University. Its deployment has significantly advanced several core research thrusts:

1. **High-Density Myoelectric Control**: By providing a high-performance pipeline capable of processing 64+ channels of HD-EMG data in real-time, the toolkit enables the development of sophisticated human-machine interfaces. It supports the transition from laboratory-based offline analysis to live, closed-loop control of robotic prostheses and assistive devices.
2. **Standardization and Reproducibility**: The modular design of `pyoephys` also ensures that signal processing standards (e.g., CAR, specific filtering bands, and QC metrics) are consistent across various research projects. This reduces "re-invention" time and lowers the barrier for new researchers entering the field of neuroscience and physiological signal processing.
3. **Low-Latency Inference**: The tight integration of PyTorch and ZeroMQ allows for sub-10ms feature extraction and classification latencies. This responsiveness is critical for minimizing the user-perceived delay in myoelectric control, which is a primary determinant of user acceptance and system efficacy.
4. **Democratizing High-Density EMG**: Commercial HD-EMG acquisition systems are often prohibitively expensive for many research groups. By leveraging the open-source Open Ephys platform and providing a high-quality, free software toolkit, `python-oephys` makes high-density myoelectric research significantly more affordable and accessible to the broader scientific community.

# Acknowledgements

This work was supported by the Neuromechatronics Lab at Carnegie Mellon University. We would like to express our gratitude to the Open Ephys community and the developers of the Open Ephys GUI for their foundational work and support of open-source electrophysiology tools.

# References
