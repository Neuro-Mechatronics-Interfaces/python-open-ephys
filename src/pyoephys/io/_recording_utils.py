

def list_recordings(session):
    """
    Return a flat list of all recordings in the session across record nodes.
    """
    all_recordings = []
    for rn in session.recordnodes:
        all_recordings.extend(rn.recordings)
    return all_recordings


def get_recording_by_name(session, name: str):
    """
    Retrieve a recording object by name.
    """
    for recording in list_recordings(session):
        if recording.name == name:
            return recording
    raise ValueError(f"No recording found with name: {name}")

