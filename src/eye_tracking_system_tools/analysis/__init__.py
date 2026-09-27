"""Post-preprocessing analysis: saccade I/O, jitter, span, DLC QC, and viewers.

Paper figure PDFs are not built from this package. Redraw them with
``figures/plot_s1_main.py`` and ``S1_Data.xlsx``.

The preprocessing GUI Saccades and Calibration tabs import::

    saccade_export, param_tune, eye_trace_io, pixel_calibration, eye_movement_span

Optional tools after a block is finalized::

    python -m eye_tracking_system_tools.analysis.saccade_viewer --registry PATH.yaml
    python -m eye_tracking_system_tools.analysis.jitter_gui
    python -m eye_tracking_system_tools.analysis.span_gui
    python -m eye_tracking_system_tools.analysis.data_yield_gui
    python -m eye_tracking_system_tools.analysis.param_tune_gui --registry PATH.yaml --params configs/analysis_params.yaml
"""

from eye_tracking_system_tools.analysis.block_registry import (
    BlockSpec,
    load_registry,
)

__all__ = [
    "BlockSpec",
    "load_registry",
]
