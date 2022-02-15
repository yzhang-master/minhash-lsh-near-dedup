from pathlib import Path

import tomllib

import minhash_lsh


def test_scaffold_is_importable_and_matches_project_metadata() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert minhash_lsh.__version__ == project["version"]
    assert project["name"] == "minhash-lsh-near-dedup"
    assert (root / "LICENSE").is_file()
    assert (root / ".github/workflows/ci.yml").is_file()
