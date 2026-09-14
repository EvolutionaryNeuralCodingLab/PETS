"""Run a frozen supplementary ``replot.py`` with a chosen KIND.

The generated replot scripts encode every drawing branch; only the KIND
constant selects which one runs. These helpers never read recording blocks.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def run_replot(
    folder: Path,
    kind: str,
    pickle_name: str | None = None,
    *,
    overwrite: bool = True,
) -> None:
    folder = Path(folder).resolve()
    src = folder / "replot.py"
    if not src.is_file():
        raise FileNotFoundError(src)
    text = src.read_text(encoding="utf-8")
    patched, n = re.subn(r'^KIND = ".*"', f'KIND = "{kind}"', text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"could not patch KIND in {src}")
    tmp = folder / f".replot_{kind}.py"
    try:
        tmp.write_text(patched, encoding="utf-8")
        cmd = [sys.executable, str(tmp)]
        if overwrite:
            cmd.append("--overwrite")
        if pickle_name:
            cmd.extend(["--pickle", pickle_name])
        subprocess.check_call(cmd, cwd=folder)
    finally:
        if tmp.exists():
            tmp.unlink()
