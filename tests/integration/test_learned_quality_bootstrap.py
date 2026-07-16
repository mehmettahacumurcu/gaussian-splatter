from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "colab" / "static_notebook_bootstrap.sh"


def test_learned_bootstrap_installs_python_312_venv() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "apt-get install -y python3.12-venv" in source
