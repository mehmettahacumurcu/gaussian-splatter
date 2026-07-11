import json

import pytest
from pydantic import ValidationError

from backend.notebooks.models import StaticNotebookRunSpec, parse_run_spec_json


def test_runspec_defaults_are_smart_balanced_l4_and_fixed_four() -> None:
    spec = StaticNotebookRunSpec(input_folder="MyDrive/captures/room")
    assert spec.input_folder == "captures/room"
    assert spec.frame_selection.mode.value == "smart"
    assert spec.frame_selection.fixed_fps == 4
    assert spec.quality.profile.value == "balanced_l4"
    assert spec.quality.n_iters is None
    assert spec.publish.replace_owned_result is True


def test_runspec_rejects_unknown_advanced_keys_and_out_of_bounds_overrides() -> None:
    with pytest.raises(ValidationError):
        StaticNotebookRunSpec.model_validate({
            "schema_version": 1,
            "input_folder": "captures/room",
            "quality": {"n_iters": 999, "advanced": {"colmap_axis": "guess"}},
        })


def test_parser_round_trips_only_json_data() -> None:
    payload = json.dumps({"schema_version": 1, "input_folder": "captures/room;print('x')"})
    spec = parse_run_spec_json(payload)
    assert spec.input_folder == "captures/room;print('x')"
