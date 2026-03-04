"""
Input and Output (IO) utilities for Open Ephys and NPZ data.
"""

from ._config_utils import (
    get_or_prompt_value,
    load_simple_config,
    prompt_directory,
    prompt_file,
    prompt_options,
    prompt_save_file,
)
from ._conversion import convert_session_api, convert_session_ui
from ._dataset_utils import (
    assess_channel_quality,
    load_dataset,
    load_open_ephys_data,
    process_recording,
    process_recordings,
    save_dataset,
    save_session_to_mat,
    save_session_to_npz,
    select_channels,
)
from ._file_utils import (
    discover_and_group_files,
    find_event_for_file,
    find_oebin_files,
    labels_from_events,
    load_config_file,
    load_json_file,
    load_yaml_file,
    parse_event_file,
    stem_without_timestamp,
)
from ._grid_utils import apply_grid_permutation, infer_grid_dimensions
from ._session_loader import load_open_ephys_session, load_xdf_session
from ._utilities import (
    align_channels_by_name,
    build_indices_from_mapping,
    convert_events_to_list,
    load_metadata_json,
    lock_params_to_meta,
    normalize_name,
    parse_numeric_args,
    select_training_channels_by_name,
)

# Aliases for backward compatibility
load_oebin_file = load_open_ephys_session
load_npz_file = load_open_ephys_session
