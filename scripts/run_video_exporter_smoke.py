#!/usr/bin/env env python3
"""Smoke-run video_exporter_new for a short clip (used by agent verification)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "src/eye_tracking_system_tools/figures/reproduction/video_creation/video_exporter_new.ipynb"

PATH_TO_BLOCK = Path("/Volumes/Data/Nimrod/experiments/PV_228/2026_05_31/block_002")
VIDEO_START_MS = 33786.2
VIDEO_END_MS = VIDEO_START_MS + 20_000.0  # 20 s smoke clip
HALF_WINDOW_MS = 5000.0
INTERPOLATE = True
ASPECT_RATIO = "16:9"


def _load_cell_sources() -> dict[int, str]:
    nb = json.loads(NB.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def main() -> Path:
    sys.path.insert(0, str(REPO / "src"))
    cells = _load_cell_sources()
    g: dict = {"__name__": "__main__"}
    for idx in (1, 4, 6):
        code = cells[idx].replace("%matplotlib inline\n", "")
        exec(compile(code, f"video_exporter_new_cell_{idx}", "exec"), g)

    prepare = g["prepare_block_for_export"]
    export = g["export_block_synchronized_montage_video_moving_window"]

    block = prepare(PATH_TO_BLOCK)
    out_path = (
        block.analysis_path
        / f"montage_moving_window_smoke_{int(VIDEO_START_MS)}_{int(VIDEO_END_MS)}.mp4"
    )
    print(f"Exporting {VIDEO_START_MS}..{VIDEO_END_MS} ms -> {out_path}")
    export(
        block,
        out_path=out_path,
        start_ms=VIDEO_START_MS,
        end_ms=VIDEO_END_MS,
        half_window_ms=HALF_WINDOW_MS,
        fps=60.0,
        eye_video_mode="raw",
        arena_video=0,
        arena_frame_shift=0,
        require_all_three=True,
        interpolate=INTERPOLATE,
        aspect_ratio=ASPECT_RATIO,
        banner_title="Synchronized Video",
        show_debug_prints=True,
    )
    return out_path


if __name__ == "__main__":
    out = main()
    print("Saved:", out)
