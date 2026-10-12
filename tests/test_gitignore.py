import subprocess
from pathlib import Path


def test_private_indexes_ignored_and_synthetic_corpus_included() -> None:
    paths = [
        "data/chroma/chroma.sqlite3",
        "nested/data/chroma/header.bin",
        "custom-index/chroma.sqlite3",
        "custom-index/segment/index_metadata.pickle",
        "eval/data/private.sqlite3",
        "eval/data/atlas_policy.txt",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(paths[:-1])
