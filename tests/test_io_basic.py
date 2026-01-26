import pytest
import shutil
from pathlib import Path
from pyoephys.io._file_utils import find_oebin_files, discover_and_group_files

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
    assert "data2" in groups # timestamp stripped
