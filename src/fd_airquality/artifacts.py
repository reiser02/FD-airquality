"""Local artifact integrity and preparation source fingerprints."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

PREPARATION_FILES = (
    "artifacts.py",
    "data.py",
    "loaders.py",
    "preprocessing.py",
    "splitting.py",
    "metrics.py",
)


def source_fingerprints() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {name: file_digest(package / name) for name in PREPARATION_FILES}


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(base_dir: Path, name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        candidate = base_dir / (name if suffix == 0 else f"{name}_{suffix}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
