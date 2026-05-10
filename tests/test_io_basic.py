import json
from pathlib import Path

import numpy as np
import pytest
from pyoephys.io import load_open_ephys_session
from pyoephys.io._file_utils import discover_and_group_files, find_oebin_files


@pytest.fixture
def mock_fs(tmp_path):
    # Setup: root/session_1/structure.oebin
    s1 = tmp_path / "session_1"
    s1.mkdir()
    (s1 / "structure.oebin").touch()

    # Setup: root/session_2/foo/bar/structure.oebin
    s2 = tmp_path / "session_2" / "foo" / "bar"
    s2.mkdir(parents=True)
    (s2 / "structure.oebin").touch()

    # Setup: root/ignore_me/README.txt
    s3 = tmp_path / "ignore_me"
    s3.mkdir()
    (s3 / "README.txt").touch()

    return tmp_path


def test_find_oebin_files(mock_fs):
    oebins = find_oebin_files(mock_fs)
    assert len(oebins) == 2
    names = sorted([f.name for f in oebins])
    assert names == ["structure.oebin", "structure.oebin"]
    parent_names = sorted([f.parent.name for f in oebins])
    assert "session_1" in parent_names or "bar" in parent_names
    # Logic depends on where oebin sits (bar, session_1)


def test_discover_files(mock_fs):
    # Discovery usually for .rhd, let's test mocking .rhd
    (mock_fs / "data1.rhd").touch()
    (mock_fs / "data2_210101.rhd").touch()

    groups = discover_and_group_files(str(mock_fs), file_type="rhd")
    assert "data1" in groups
    assert "data2" in groups  # timestamp stripped


def _write_oebin_session(root: Path, meta: dict) -> Path:
    oebin = root / "structure.oebin"
    oebin.write_text(json.dumps(meta), encoding="utf-8")
    return oebin


def _write_continuous_block(
    folder: Path, samples: np.ndarray, timestamps: np.ndarray
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.asarray(samples, dtype="<i2").tofile(folder / "continuous.dat")
    np.save(folder / "timestamps.npy", np.asarray(timestamps, dtype=np.float64))


def test_load_session_converts_sample_counter_timestamps(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    meta = {
        "continuous": [
            {
                "sample_rate": 2000.0,
                "num_channels": 2,
                "channels": [
                    {"channel_name": "CH1", "bit_volts": 0.195, "units": "uV"},
                    {"channel_name": "CH2", "bit_volts": 0.195, "units": "uV"},
                ],
            }
        ]
    }
    oebin = _write_oebin_session(session_dir, meta)
    samples = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    _write_continuous_block(session_dir, samples, np.array([100, 101, 102]))

    session = load_open_ephys_session(oebin)

    np.testing.assert_allclose(session["t_amplifier"], np.array([0.05, 0.0505, 0.051]))
    assert session["amplifier_data"].shape == (2, 3)
    assert session["channel_names"] == ["CH1", "CH2"]


def test_load_session_prefers_stream_local_files(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    meta = {
        "continuous": [
            {
                "sample_rate": 1000.0,
                "num_channels": 1,
                "folder_name": "stream_b",
                "channels": [
                    {"channel_name": "CH1", "bit_volts": 1.0, "units": "uV"},
                ],
            }
        ]
    }
    oebin = _write_oebin_session(session_dir, meta)

    _write_continuous_block(
        session_dir, np.array([[11], [22]], dtype=np.int16), np.array([0.0, 0.001])
    )
    _write_continuous_block(
        session_dir / "stream_b",
        np.array([[101], [202]], dtype=np.int16),
        np.array([10, 11]),
    )

    session = load_open_ephys_session(oebin)

    np.testing.assert_allclose(
        session["amplifier_data"], np.array([[101.0, 202.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(session["t_amplifier"], np.array([0.01, 0.011]))
