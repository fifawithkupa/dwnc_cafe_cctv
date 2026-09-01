"""Save one clean and one annotated still per judged tick.

The harness that grades detection needs two pictures of the same instant.
The clean one is what a grader counts people in; the marked one is what
explains a disagreement afterwards.  They must never be swapped: showing a
grader the boxes first anchors the count to whatever SeatNow already drew,
which is how a scoring harness quietly starts grading itself.

Stills rather than an MP4 is also what keeps this usable under the
deployment rule that forbids annotated video on disk (CLAUDE.md).

    frames/angle1/clean/t0015.0s.jpg    # counted from
    frames/angle1/marked/t0015.0s.jpg   # diagnosed from
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


CLEAN_DIR = "clean"
MARKED_DIR = "marked"


def frame_stem(timestamp: float) -> str:
    """Media timestamp as a filename stem that sorts in time order.

    Zero-padded because plain ``t105.0s`` would sort before ``t15.0s`` and
    the folder is meant to be paged through in order by a human.
    """
    if timestamp < 0:
        raise ValueError(f"timestamp cannot be negative: {timestamp}")
    return f"t{timestamp:06.1f}s"


def frame_paths(frame_dir: Path, timestamp: float) -> Tuple[Path, Path]:
    """Return (clean, marked) paths for one tick.  Clean is always first."""
    stem = frame_stem(timestamp)
    root = Path(frame_dir)
    return root / CLEAN_DIR / f"{stem}.jpg", root / MARKED_DIR / f"{stem}.jpg"


def save_frame_pair(
    frame_dir: Path,
    timestamp: float,
    clean: np.ndarray,
    marked: np.ndarray,
) -> Tuple[Path, Path]:
    """Write both stills, raising rather than skipping on failure."""
    clean_path, marked_path = frame_paths(frame_dir, timestamp)
    for path, image in ((clean_path, clean), (marked_path, marked)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write frame: {path}")
    return clean_path, marked_path
