import pytest

from backend.notebooks.drive_paths import derive_result_folder, normalize_input_folder


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("captures/myroom", "captures/myroom"), ("MyDrive/captures/myroom", "captures/myroom")],
)
def test_normalizes_my_drive_relative_paths(raw: str, canonical: str) -> None:
    assert normalize_input_folder(raw) == canonical
    assert derive_result_folder(canonical) == "captures/myroom_result"


@pytest.mark.parametrize(
    "raw",
    ["", "MyDrive", "MyDrive/", "/content/drive/MyDrive/x", "C:/x", "a\\b", "a/../b", "a/./b", "a//b", "/a/b", "a/b/", "a/result_result", "a/\x00b"],
)
def test_rejects_unsafe_or_ambiguous_paths(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_input_folder(raw)
