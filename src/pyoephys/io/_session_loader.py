import os
from open_ephys.analysis.session import Session


def load_session(path: str) -> Session:
    """
    Load an Open Ephys session from the specified directory.
    Raises informative errors if loading fails.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path '{path}' is not a valid directory.")

    try:
        session = Session(path)
        print(f"[Loaded] Session from {path}")
        print(f"[Info] {len(session.recordnodes)} record node(s) found.")
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load session from {path}: {e}")
