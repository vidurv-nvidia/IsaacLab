# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a clean customer overlay archive for this demonstrator."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_DIR.parents[4]
_ARCHIVE_ROOT = "isaaclab-heterogeneous-motion-demo"
_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "logs"}
_EXCLUDED_SUFFIXES = {".log", ".pt", ".tmp"}


def _iter_bundle_files() -> list[tuple[Path, Path]]:
    """Return source and archive-relative paths included in the customer overlay."""
    package_root = Path("source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion")
    entries = [
        (_REPOSITORY_ROOT / "LICENSE", Path("LICENSE")),
        (_REPOSITORY_ROOT / "VERSION", Path("VERSION")),
        (
            _REPOSITORY_ROOT / "heterogeneous_environments_motion_references.md",
            Path("heterogeneous_environments_motion_references.md"),
        ),
        (
            _REPOSITORY_ROOT / "source/isaaclab_tasks/pyproject.toml",
            Path("source/isaaclab_tasks/pyproject.toml"),
        ),
        (
            _REPOSITORY_ROOT / "source/isaaclab_tasks/test/contrib/test_heterogeneous_motion.py",
            Path("source/isaaclab_tasks/test/contrib/test_heterogeneous_motion.py"),
        ),
        (
            _REPOSITORY_ROOT / "source/isaaclab_tasks/test/contrib/test_heterogeneous_motion_layout_routing.py",
            Path("source/isaaclab_tasks/test/contrib/test_heterogeneous_motion_layout_routing.py"),
        ),
        (
            _REPOSITORY_ROOT / "source/isaaclab_tasks/test/contrib/test_heterogeneous_motion_registration.py",
            Path("source/isaaclab_tasks/test/contrib/test_heterogeneous_motion_registration.py"),
        ),
    ]
    for source_path in _PACKAGE_DIR.rglob("*"):
        relative_path = source_path.relative_to(_PACKAGE_DIR)
        if (
            not source_path.is_file()
            or any(part in _EXCLUDED_PARTS for part in relative_path.parts)
            or source_path.suffix in _EXCLUDED_SUFFIXES
        ):
            continue
        entries.append((source_path, package_root / relative_path))
    return entries


def create_bundle(output_path: str | Path) -> Path:
    """Create a gzip-compressed archive with only demonstrator inputs.

    Args:
        output_path: Destination ``.tar.gz`` path.

    Returns:
        The created archive path.
    """
    output_path = Path(output_path).resolve()
    if output_path.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Output path must end in '.tar.gz'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source_path, archive_path in _iter_bundle_files():
            archive.add(source_path, arcname=str(Path(_ARCHIVE_ROOT) / archive_path), recursive=False)
    return output_path


def main() -> None:
    """Create the overlay archive and print its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination .tar.gz archive.")
    args = parser.parse_args()
    output_path = create_bundle(args.output)
    print(output_path)
    for _, archive_path in _iter_bundle_files():
        print(archive_path)


if __name__ == "__main__":
    main()
