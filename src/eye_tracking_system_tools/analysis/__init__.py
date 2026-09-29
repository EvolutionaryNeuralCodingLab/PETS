"""Saccade detection used by the Preprocessing GUI, plus the saccade viewer.

Paper figure PDFs are not built from this package. Redraw them with
``figures/plot_s1_main.py`` and ``S1_Data.xlsx``.

The preprocessing GUI Saccades and Calibration tabs import::

    saccade_export, param_tune, eye_trace_io, pixel_calibration

Detection defaults are the ``saccade`` and ``binocular`` sections of
``configs/analysis_params.yaml``.

Review detected events after a block is finalized::

    python -m eye_tracking_system_tools.analysis.saccade_viewer --registry PATH.yaml
"""

from eye_tracking_system_tools.analysis.block_registry import (
    BlockSpec,
    load_registry,
)

__all__ = [
    "BlockSpec",
    "load_registry",
]
