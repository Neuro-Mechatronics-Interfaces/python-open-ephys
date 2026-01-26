"""
Input and Output (IO) utilities for Open Ephys and NPZ data.
"""
from ._session_loader import load_open_ephys_session
from ._file_utils import (
    find_oebin_files,
    load_yaml_file,
    load_json_file,
    load_config_file,
    labels_from_events,
    discover_and_group_files,
    find_event_for_file,
    stem_without_timestamp,
    parse_event_file
)
from ._grid_utils import infer_grid_dimensions, apply_grid_permutation
from ._dataset_utils import process_recording, save_dataset, load_open_ephys_data, select_channels
from ._config_utils import load_simple_config, prompt_directory, prompt_file, get_or_prompt_value
from ._utilities import (
    parse_numeric_args,
    normalize_name,
    load_metadata_json,
    align_channels_by_name,
    select_training_channels_by_name,
    convert_events_to_list,
    lock_params_to_meta,
    build_indices_from_mapping
)

# Aliases for backward compatibility
load_oebin_file = load_open_ephys_session
load_npz_file = load_open_ephys_session
