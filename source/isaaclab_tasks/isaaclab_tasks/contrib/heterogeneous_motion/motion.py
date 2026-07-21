# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import torch

MOTION_DIR = Path(__file__).with_name("motions")
UPPER_MOTION_FILE = MOTION_DIR / "maze_upper.csv"
LOWER_MOTION_FILE = MOTION_DIR / "maze_lower.csv"

UPPER_WAYPOINTS = ((-2.3, 0.0), (-0.65, 1.15), (0.65, 1.15), (2.3, 0.0))
LOWER_WAYPOINTS = ((-2.3, 0.0), (-0.65, -1.15), (0.65, -1.15), (2.3, 0.0))


def generate_motion_file(
    file_path: str | Path,
    waypoints: Sequence[tuple[float, float]],
    spacing: float = 0.1,
) -> Path:
    """Generate a planar motion file by uniformly sampling line segments.

    Args:
        file_path: Output CSV file.
        waypoints: Planar waypoint positions [m].
        spacing: Maximum distance between adjacent samples [m].

    Returns:
        Path to the generated CSV file.
    """
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required.")
    if spacing <= 0.0:
        raise ValueError("Spacing must be positive.")

    samples = [waypoints[0]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        count = max(1, math.ceil(math.hypot(dx, dy) / spacing))
        samples.extend((start[0] + dx * i / count, start[1] + dy * i / count) for i in range(1, count + 1))

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(("x", "y"))
        writer.writerows((f"{x:.6f}", f"{y:.6f}") for x, y in samples)
    return output_path


def generate_demo_motion_files(output_dir: str | Path = MOTION_DIR) -> tuple[Path, Path]:
    """Generate both motion files used by the heterogeneous maze demo."""
    output_dir = Path(output_dir)
    return (
        generate_motion_file(output_dir / UPPER_MOTION_FILE.name, UPPER_WAYPOINTS),
        generate_motion_file(output_dir / LOWER_MOTION_FILE.name, LOWER_WAYPOINTS),
    )


def load_motion_file(file_path: str | Path, device: str) -> torch.Tensor:
    """Load planar positions [m] from a generated motion CSV file."""
    with Path(file_path).open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["x", "y"]:
            raise ValueError(f"Expected CSV columns ['x', 'y'], got {reader.fieldnames}.")
        points = [(float(row["x"]), float(row["y"])) for row in reader]
    if not points:
        raise ValueError(f"Motion file '{file_path}' contains no samples.")
    return torch.tensor(points, dtype=torch.float32, device=device)


if __name__ == "__main__":
    for generated_file in generate_demo_motion_files():
        print(generated_file)
