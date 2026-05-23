#!/usr/bin/env python
"""Headless check: load a block, read frames, build QPixmap (no window)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    load_block_session,
)
from eye_tracking_system_tools.annotation.block_annotator.config_io import (
    ensure_config_template,
    load_config,
)
from eye_tracking_system_tools.annotation.block_annotator.models import (
    first_timeline_index_with_frame,
)
from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
    VideoReader,
    numpy_rgb_to_qpixmap,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ensure_config_template(args.output)
    cfg = load_config(None, args.output)
    session = load_block_session(args.block, args.output, cfg)

    start = first_timeline_index_with_frame(session.final_sync_df)
    arena_f, le_f, re_f = session.frame_ids_at(start)
    print(f"rows={session.n} start_index={start}")
    print(f"frames at start: arena={arena_f} L={le_f} R={re_f}")
    print(f"arena_videos={len(session.arena_videos)} le={len(session.le_videos)} re={len(session.re_videos)}")
    if session.arena_videos:
        print(f"  arena path: {session.arena_videos[0]}")
    if session.le_videos:
        print(f"  le path: {session.le_videos[0]}")

    app = QApplication(sys.argv)
    ok = True
    for label, paths, fid in (
        ("arena", session.arena_videos, arena_f),
        ("left", session.le_videos, le_f),
        ("right", session.re_videos, re_f),
    ):
        if not paths or fid is None:
            print(f"[skip] {label}: no video or frame id")
            continue
        reader = VideoReader(paths[0])
        frame = reader.read_frame(fid)
        if frame is None:
            print(f"[FAIL] {label}: could not read frame {fid} from {paths[0]}")
            ok = False
            continue
        pix = numpy_rgb_to_qpixmap(frame)
        if pix.isNull():
            print(f"[FAIL] {label}: QPixmap is null for frame {fid}")
            ok = False
        else:
            print(f"[OK] {label}: frame {fid} shape={frame.shape} pixmap={pix.width()}x{pix.height()}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
