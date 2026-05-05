Overview
========

python-open-ephys is a toolkit for working with Open Ephys electrophysiology data in Python. It covers offline file loading, real-time streaming interfaces, signal-processing utilities, GUI tools, and example application code.

Core capabilities
-----------------

- Load Open Ephys Binary recordings and normalized NPZ exports.
- Stream live data from the Open Ephys GUI over ZMQ and LSL.
- Filter, synchronize, and quality-check EMG and electrophysiology signals.
- Launch viewer applications for online and offline inspection.
- Reuse the same data model across analysis and machine-learning workflows.

Installation
------------

Install the latest published package:

.. code-block:: bash

   pip install python-oephys

Install optional extras when you need them:

.. code-block:: bash

   pip install "python-oephys[gui]"
   pip install "python-oephys[ml]"
   pip install "python-oephys[docs]"

Quick start
-----------

.. code-block:: python

   from pyoephys.io import load_open_ephys_session
   from pyoephys.processing import filter_emg

   session = load_open_ephys_session("path/to/recording.oebin")
   filtered = filter_emg(
       session["amplifier_data"],
       filter_type="bandpass",
       lowcut=10,
       highcut=500,
       fs=session["sample_rate"],
   )

Examples
--------

- ``examples/read_files/`` contains file-loading and export examples.
- ``examples/interface/`` contains ZMQ, LSL, IMU, and hardware-interface examples.
- ``examples/applications/`` contains viewer-style application entry points.