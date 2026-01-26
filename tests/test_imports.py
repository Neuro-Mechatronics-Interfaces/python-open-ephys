import pytest
import sys
import os

def test_package_import():
    import pyoephys
    assert pyoephys is not None

def test_submodules_import():
    from pyoephys import processing, ml, applications, io, interface
    assert processing is not None
    assert ml is not None
    assert applications is not None
    assert io is not None
    assert interface is not None

def test_ml_imports():
    from pyoephys.ml import EMGClassifierCNNLSTM, EMGClassifier, EMGRegressor
    assert EMGClassifierCNNLSTM is not None

def test_app_imports():
    from pyoephys.applications import RealTimeEMGViewer, EMGViewer
    assert RealTimeEMGViewer is not None
