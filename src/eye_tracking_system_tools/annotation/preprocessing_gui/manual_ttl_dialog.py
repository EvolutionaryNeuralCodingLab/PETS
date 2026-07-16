"""Qt dialog for manual Open Ephys TTL line mapping and arena window selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

REQUIRED_TTL_ROLES = frozenset({"Arena_TTL", "LED_driver", "L_eye_TTL", "R_eye_TTL"})
MappingSources = list[tuple[str, dict[str, int]]]


def _events_csv_path_for_block(blocksync: BlockSync) -> Path:
    return blocksync.block_path / "oe_files" / blocksync.oe_dirname / "events.csv"


def default_ttl_line_for_role(blocksync: BlockSync, role: str, fallback: int = 0) -> int:
    """Resolve a role's Open Ephys line from ``channeldict`` or a saved sidecar."""
    channeldict = getattr(blocksync, "channeldict", None) or {}
    for line, name in channeldict.items():
        if str(name) == role:
            return int(line)

    oe_dirname = getattr(blocksync, "oe_dirname", None)
    if oe_dirname:
        events_csv = blocksync.block_path / "oe_files" / oe_dirname / "events.csv"
        sidecar = events_csv.parent / "ttl_manual_mapping.json"
        if sidecar.is_file():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                mapped = payload.get("manual_line_map", {})
                if role in mapped:
                    return int(mapped[role])
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    return int(fallback)


def _arena_rising_samples(events_csv_path: Path, arena_line: int) -> np.ndarray:
    df = pd.read_csv(events_csv_path)
    df_on = df[(df["state"] == 1) & (df["line"] == int(arena_line))].copy()
    samples = df_on["sample_number"].to_numpy(dtype=np.int64)
    samples.sort()
    return samples


def auto_assign_eye_ttl_lines(
    blocksync: BlockSync,
    events_csv_path: Path,
    line_a: int,
    line_b: int,
) -> dict[str, int]:
    """Mirror ``BlockSync._manual_ttl_map_and_window`` L/R assignment by frame counts."""
    if line_a == line_b:
        raise ValueError("The two eye line numbers must be different.")

    df_events = pd.read_csv(events_csv_path)
    df_on = df_events[df_events["state"] == 1]
    count_a = int(len(df_on[df_on["line"] == line_a]))
    count_b = int(len(df_on[df_on["line"] == line_b]))

    left_frame_count = getattr(blocksync, "le_frame_count", None)
    right_frame_count = getattr(blocksync, "re_frame_count", None)
    if (
        left_frame_count is None
        or right_frame_count is None
        or left_frame_count <= 0
        or right_frame_count <= 0
    ):
        if blocksync.le_videos and blocksync.re_videos:
            import cv2

            cap_l = cv2.VideoCapture(str(blocksync.le_videos[0]))
            cap_r = cv2.VideoCapture(str(blocksync.re_videos[0]))
            left_frame_count = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
            right_frame_count = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_l.release()
            cap_r.release()

    if (
        left_frame_count is not None
        and right_frame_count is not None
        and left_frame_count > 0
        and right_frame_count > 0
    ):
        diff_a_as_left = abs(count_a - left_frame_count) + abs(count_b - right_frame_count)
        diff_a_as_right = abs(count_a - right_frame_count) + abs(count_b - left_frame_count)
        if diff_a_as_left <= diff_a_as_right:
            return {"L_eye_TTL": int(line_a), "R_eye_TTL": int(line_b)}
        return {"L_eye_TTL": int(line_b), "R_eye_TTL": int(line_a)}

    raise RuntimeError(
        "Could not read eye video frame counts; set L_eye_TTL and R_eye_TTL explicitly."
    )


def compute_arena_window(
    blocksync: BlockSync,
    events_csv_path: Path,
    arena_line: int,
    *,
    mode: str,
    start_index: int | str = 0,
    end_index: int | str = -1,
    start_sample: int | None = None,
    end_sample: int | None = None,
    gap_threshold_ms: float = 1000.0,
) -> dict[str, int]:
    """Build ``arena_window`` dict using the same rules as the notebook fallback."""
    arena_samples = _arena_rising_samples(events_csv_path, arena_line)
    if len(arena_samples) < 2:
        raise ValueError(f"Arena line {arena_line} has too few rising edges.")

    mode = (mode or "i").strip().lower()
    if mode == "a":
        diff_arr_ms = np.diff(arena_samples) / (blocksync.sample_rate / 1000.0)
        arena_start_stop = np.where(diff_arr_ms > gap_threshold_ms)[0]
        option_count = len(arena_start_stop)
        if option_count == 0:
            raise ValueError(
                f"No gaps > {gap_threshold_ms} ms found. "
                "Try a lower threshold or use manual mode (index/sample)."
            )
        if option_count == 1:
            arena_start_timestamp = int(arena_samples[0])
            arena_end_timestamp = int(arena_samples[-1])
            arena_start_index = 0
        elif option_count == 2:
            arena_start_timestamp = int(arena_samples[arena_start_stop[0] + 1])
            arena_end_timestamp = int(arena_samples[arena_start_stop[1]])
            arena_start_index = int(arena_start_stop[0]) + 1
        else:
            ind_max_diff = int(np.argmax(np.diff(arena_start_stop)))
            start_ind = int(arena_start_stop[ind_max_diff])
            end_ind = int(arena_start_stop[ind_max_diff + 1])
            arena_start_timestamp = int(arena_samples[start_ind + 1])
            arena_end_timestamp = int(arena_samples[end_ind])
            arena_start_index = start_ind + 1
    elif mode == "i":
        start_i = BlockSync._parse_edge_index(
            blocksync, str(start_index), n_edges=len(arena_samples)
        )
        end_i = BlockSync._parse_edge_index(
            blocksync, str(end_index), n_edges=len(arena_samples)
        )
        if end_i < start_i:
            raise ValueError(f"End index ({end_i}) is before start index ({start_i}).")
        arena_start_timestamp = int(arena_samples[start_i])
        arena_end_timestamp = int(arena_samples[end_i])
        arena_start_index = int(start_i)
    elif mode == "s":
        if start_sample is None or end_sample is None:
            raise ValueError("Sample mode requires start_sample and end_sample.")
        arena_start_timestamp = int(start_sample)
        arena_end_timestamp = int(end_sample)
        arena_start_index = int(
            np.searchsorted(arena_samples, arena_start_timestamp, side="left")
        )
    else:
        raise ValueError(f"Unknown arena window mode {mode!r}; use 'i', 's', or 'a'.")

    return {
        "arena_start_timestamp": arena_start_timestamp,
        "arena_end_timestamp": arena_end_timestamp,
        "arena_start_index": arena_start_index,
    }


def build_manual_ttl_payload(
    blocksync: BlockSync,
    events_csv_path: Path,
    *,
    arena_line: int,
    l_eye_line: int,
    r_eye_line: int,
    led_driver_line: int,
    extra_roles: dict[str, int] | None = None,
    window_mode: str = "i",
    start_index: int | str = 0,
    end_index: int | str = -1,
    start_sample: int | None = None,
    end_sample: int | None = None,
    gap_threshold_ms: float = 1000.0,
    arena_channel_name: str = "Arena_TTL",
) -> tuple[dict[str, int], dict[str, int]]:
    manual_line_map: dict[str, int] = {
        arena_channel_name: int(arena_line),
        "L_eye_TTL": int(l_eye_line),
        "R_eye_TTL": int(r_eye_line),
        "LED_driver": int(led_driver_line),
    }
    if extra_roles:
        manual_line_map.update({str(k): int(v) for k, v in extra_roles.items()})

    assigned_lines = list(manual_line_map.values())
    if len(set(assigned_lines)) != len(assigned_lines):
        raise ValueError(
            "Each TTL role must map to a distinct line number "
            f"(got duplicates in {manual_line_map})."
        )

    arena_window = compute_arena_window(
        blocksync,
        events_csv_path,
        manual_line_map[arena_channel_name],
        mode=window_mode,
        start_index=start_index,
        end_index=end_index,
        start_sample=start_sample,
        end_sample=end_sample,
        gap_threshold_ms=gap_threshold_ms,
    )
    return manual_line_map, arena_window


def save_ttl_manual_sidecar(
    events_csv_path: Path,
    blocksync: BlockSync,
    manual_line_map: dict[str, int],
    arena_window: dict[str, int],
) -> Path:
    sidecar = events_csv_path.parent / "ttl_manual_mapping.json"
    payload = {
        "block_path": str(blocksync.block_path),
        "events_csv": str(events_csv_path),
        "sample_rate": float(blocksync.sample_rate),
        "manual_line_map": manual_line_map,
        "arena_window": arena_window,
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar


class TtlRasterPlot(QtWidgets.QWidget):
    """Embedded pyqtgraph raster of TTL rising edges (mirrors ``_plot_ttl_raster``)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "TTL line")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.plot_widget)

    def set_events_csv(self, events_csv_path: Path, sample_rate: float) -> None:
        df = pd.read_csv(events_csv_path)
        df_on = df[df["state"] == 1].copy()
        self.plot_widget.clear()
        if df_on.empty:
            return

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        lines = sorted(df_on["line"].unique().tolist())
        for i, ln in enumerate(lines):
            s = df_on.loc[df_on["line"] == ln, "sample_number"].to_numpy(dtype=np.int64)
            s.sort()
            t = s / float(sample_rate)
            y = np.full_like(t, fill_value=float(ln), dtype=float)
            self.plot_widget.plot(
                t,
                y,
                pen=None,
                symbol="o",
                symbolSize=4,
                symbolBrush=pg.mkBrush(palette[i % len(palette)]),
                name=f"line {ln}",
            )


class ManualTtlDialog(QtWidgets.QDialog):
    """Collect ``manual_line_map`` + ``arena_window`` without ``input()`` prompts."""

    def __init__(
        self,
        blocksync: BlockSync,
        events_csv_path: Path | None = None,
        *,
        arena_channel_name: str = "Arena_TTL",
        mapping_sources: MappingSources | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._blocksync = blocksync
        self._events_csv_path = Path(events_csv_path or _events_csv_path_for_block(blocksync))
        self._arena_channel_name = arena_channel_name
        self._mapping_sources = list(mapping_sources or [])
        self._manual_line_map: dict[str, int] | None = None
        self._arena_window: dict[str, int] | None = None

        self.setWindowTitle(f"Manual TTL mapping — block {blocksync.block_num}")
        self.resize(920, 720)
        self._build_ui()
        self._populate_summary()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        self._summary = QtWidgets.QTableWidget(0, 6)
        self._summary.setHorizontalHeaderLabels(
            ["line", "n_rising", "est_hz", "median_dt_ms", "t_first_s", "t_last_s"]
        )
        self._summary.horizontalHeader().setStretchLastSection(True)
        root.addWidget(QtWidgets.QLabel("TTL line summary (rising edges):"))
        root.addWidget(self._summary)

        raster_row = QtWidgets.QHBoxLayout()
        self._raster = TtlRasterPlot()
        self._raster.setMinimumHeight(180)
        raster_row.addWidget(self._raster, stretch=1)
        self._btn_browser_raster = QtWidgets.QPushButton("Open raster\nin browser")
        raster_row.addWidget(self._btn_browser_raster)
        root.addLayout(raster_row)

        if self._mapping_sources:
            leech_row = QtWidgets.QHBoxLayout()
            leech_row.addWidget(QtWidgets.QLabel("Copy mapping from block:"))
            self._leech_block = QtWidgets.QComboBox()
            self._leech_block.addItem("(select block)", None)
            for label, line_map in self._mapping_sources:
                self._leech_block.addItem(label, line_map)
            self._btn_leech_apply = QtWidgets.QPushButton("Apply")
            leech_row.addWidget(self._leech_block, stretch=1)
            leech_row.addWidget(self._btn_leech_apply)
            root.addLayout(leech_row)

        map_box = QtWidgets.QGroupBox("Role → line mapping")
        form = QtWidgets.QFormLayout(map_box)
        self._arena_line = QtWidgets.QSpinBox()
        self._arena_line.setRange(0, 128)
        self._l_eye_line = QtWidgets.QSpinBox()
        self._l_eye_line.setRange(0, 128)
        self._r_eye_line = QtWidgets.QSpinBox()
        self._r_eye_line.setRange(0, 128)
        self._led_driver_line = QtWidgets.QSpinBox()
        self._led_driver_line.setRange(0, 128)
        self._btn_auto_eye = QtWidgets.QPushButton("Auto-assign L/R from two lines")
        self._eye_a = QtWidgets.QSpinBox()
        self._eye_a.setRange(0, 128)
        self._eye_b = QtWidgets.QSpinBox()
        self._eye_b.setRange(0, 128)
        auto_row = QtWidgets.QHBoxLayout()
        auto_row.addWidget(QtWidgets.QLabel("Lines:"))
        auto_row.addWidget(self._eye_a)
        auto_row.addWidget(self._eye_b)
        auto_row.addWidget(self._btn_auto_eye)
        form.addRow(f"{self._arena_channel_name} line:", self._arena_line)
        form.addRow("LED_driver line:", self._led_driver_line)
        form.addRow("L_eye_TTL line:", self._l_eye_line)
        form.addRow("R_eye_TTL line:", self._r_eye_line)
        form.addRow("", auto_row)
        root.addWidget(map_box)

        extra_box = QtWidgets.QGroupBox("Additional TTL channels (optional)")
        extra_layout = QtWidgets.QVBoxLayout(extra_box)
        self._extra_table = QtWidgets.QTableWidget(0, 2)
        self._extra_table.setHorizontalHeaderLabels(["Role name", "Line"])
        self._extra_table.horizontalHeader().setStretchLastSection(True)
        extra_layout.addWidget(self._extra_table)
        extra_btns = QtWidgets.QHBoxLayout()
        self._btn_add_extra = QtWidgets.QPushButton("Add channel")
        self._btn_remove_extra = QtWidgets.QPushButton("Remove selected")
        extra_btns.addWidget(self._btn_add_extra)
        extra_btns.addWidget(self._btn_remove_extra)
        extra_btns.addStretch(1)
        extra_layout.addLayout(extra_btns)
        root.addWidget(extra_box)

        win_box = QtWidgets.QGroupBox("Arena sync window")
        win_layout = QtWidgets.QVBoxLayout(win_box)
        mode_row = QtWidgets.QHBoxLayout()
        self._mode_index = QtWidgets.QRadioButton("Index (rising-edge 0..N-1)")
        self._mode_sample = QtWidgets.QRadioButton("Sample number")
        self._mode_auto = QtWidgets.QRadioButton("Automatic (gap detection)")
        self._mode_auto.setChecked(True)
        mode_row.addWidget(self._mode_index)
        mode_row.addWidget(self._mode_sample)
        mode_row.addWidget(self._mode_auto)
        win_layout.addLayout(mode_row)

        idx_form = QtWidgets.QFormLayout()
        self._start_index = QtWidgets.QLineEdit("0")
        self._end_index = QtWidgets.QLineEdit("-1")
        idx_form.addRow("Start index:", self._start_index)
        idx_form.addRow("End index:", self._end_index)
        win_layout.addLayout(idx_form)

        sample_form = QtWidgets.QFormLayout()
        self._start_sample = QtWidgets.QSpinBox()
        self._start_sample.setRange(0, 2_000_000_000)
        self._end_sample = QtWidgets.QSpinBox()
        self._end_sample.setRange(0, 2_000_000_000)
        sample_form.addRow("Start sample:", self._start_sample)
        sample_form.addRow("End sample:", self._end_sample)
        win_layout.addLayout(sample_form)

        gap_row = QtWidgets.QHBoxLayout()
        self._gap_ms = QtWidgets.QDoubleSpinBox()
        self._gap_ms.setRange(1.0, 60_000.0)
        self._gap_ms.setValue(1000.0)
        gap_row.addWidget(QtWidgets.QLabel("Gap threshold (ms):"))
        gap_row.addWidget(self._gap_ms)
        gap_row.addStretch(1)
        win_layout.addLayout(gap_row)
        root.addWidget(win_box)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_ok = QtWidgets.QPushButton("Apply mapping")
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_cancel)
        root.addLayout(btn_row)

        self._btn_auto_eye.clicked.connect(self._on_auto_assign_eyes)
        self._btn_browser_raster.clicked.connect(self._on_open_browser_raster)
        if self._mapping_sources:
            self._btn_leech_apply.clicked.connect(self._on_apply_leech_mapping)
        self._btn_add_extra.clicked.connect(self._add_extra_channel_row)
        self._btn_remove_extra.clicked.connect(self._remove_selected_extra_channels)
        self._btn_ok.clicked.connect(self._on_accept)
        self._btn_cancel.clicked.connect(self.reject)
        self._mode_index.toggled.connect(self._refresh_mode_enabled)
        self._mode_sample.toggled.connect(self._refresh_mode_enabled)
        self._mode_auto.toggled.connect(self._refresh_mode_enabled)
        self._refresh_mode_enabled()

    def _add_extra_channel_row(self, role: str = "", line: int = 0) -> None:
        row = self._extra_table.rowCount()
        self._extra_table.insertRow(row)
        role_edit = QtWidgets.QLineEdit(str(role))
        line_spin = QtWidgets.QSpinBox()
        line_spin.setRange(0, 128)
        line_spin.setValue(int(line))
        self._extra_table.setCellWidget(row, 0, role_edit)
        self._extra_table.setCellWidget(row, 1, line_spin)

    def _remove_selected_extra_channels(self) -> None:
        rows = sorted({idx.row() for idx in self._extra_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._extra_table.removeRow(row)

    def _extra_roles_from_table(self) -> dict[str, int]:
        extra: dict[str, int] = {}
        for row in range(self._extra_table.rowCount()):
            role_widget = self._extra_table.cellWidget(row, 0)
            line_widget = self._extra_table.cellWidget(row, 1)
            if not isinstance(role_widget, QtWidgets.QLineEdit):
                continue
            if not isinstance(line_widget, QtWidgets.QSpinBox):
                continue
            role = role_widget.text().strip()
            if not role:
                raise ValueError(f"Additional channel row {row + 1} is missing a role name.")
            if role in REQUIRED_TTL_ROLES:
                raise ValueError(
                    f"Role {role!r} is already mapped above; use the required fields instead."
                )
            if role in extra:
                raise ValueError(f"Duplicate additional role name: {role!r}")
            extra[role] = int(line_widget.value())
        return extra

    def _populate_extra_channels(self) -> None:
        self._extra_table.setRowCount(0)
        seen: set[str] = set()
        channeldict = getattr(self._blocksync, "channeldict", None) or {}
        for line, name in sorted(channeldict.items(), key=lambda kv: int(kv[0])):
            role = str(name)
            if role in REQUIRED_TTL_ROLES or role in seen:
                continue
            self._add_extra_channel_row(role, int(line))
            seen.add(role)

        oe_dirname = getattr(self._blocksync, "oe_dirname", None)
        if oe_dirname:
            sidecar = (
                self._blocksync.block_path / "oe_files" / oe_dirname / "ttl_manual_mapping.json"
            )
            if sidecar.is_file():
                try:
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                    for role, line in payload.get("manual_line_map", {}).items():
                        if role in REQUIRED_TTL_ROLES or role in seen:
                            continue
                        self._add_extra_channel_row(str(role), int(line))
                        seen.add(str(role))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass

    def _refresh_mode_enabled(self) -> None:
        idx = self._mode_index.isChecked()
        sample = self._mode_sample.isChecked()
        auto = self._mode_auto.isChecked()
        self._start_index.setEnabled(idx)
        self._end_index.setEnabled(idx)
        self._start_sample.setEnabled(sample)
        self._end_sample.setEnabled(sample)
        self._gap_ms.setEnabled(auto)

    def _populate_summary(self) -> None:
        summary = self._blocksync._summarize_ttl_lines_from_events_csv(self._events_csv_path)
        self._summary.setRowCount(len(summary))
        for row, rec in summary.iterrows():
            for col, key in enumerate(
                ["line", "n_rising", "est_hz", "median_dt_ms", "t_first_s", "t_last_s"]
            ):
                val = rec[key]
                text = "" if pd.isna(val) else str(val)
                self._summary.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        self._raster.set_events_csv(self._events_csv_path, float(self._blocksync.sample_rate))

        self._arena_line.setValue(
            default_ttl_line_for_role(
                self._blocksync, self._arena_channel_name, fallback=0
            )
        )
        self._led_driver_line.setValue(
            default_ttl_line_for_role(self._blocksync, "LED_driver", fallback=4)
        )
        self._l_eye_line.setValue(
            default_ttl_line_for_role(self._blocksync, "L_eye_TTL", fallback=0)
        )
        self._r_eye_line.setValue(
            default_ttl_line_for_role(self._blocksync, "R_eye_TTL", fallback=0)
        )

        if not summary.empty:
            if self._arena_line.value() == 0:
                self._arena_line.setValue(int(summary.iloc[0]["line"]))
            unused = [
                int(summary.iloc[i]["line"])
                for i in range(len(summary))
                if int(summary.iloc[i]["line"])
                not in {
                    self._arena_line.value(),
                    self._led_driver_line.value(),
                }
            ]
            if len(unused) >= 1 and self._l_eye_line.value() == 0:
                self._l_eye_line.setValue(unused[0])
            if len(unused) >= 2 and self._r_eye_line.value() == 0:
                self._r_eye_line.setValue(unused[1])
            if len(unused) >= 2:
                self._eye_a.setValue(unused[0])
                self._eye_b.setValue(unused[1])
        self._populate_extra_channels()

    def apply_line_map(self, manual_line_map: dict[str, int]) -> None:
        """Fill role spinboxes and extra channels from a role→line map (not arena window)."""
        arena_role = self._arena_channel_name
        if arena_role in manual_line_map:
            self._arena_line.setValue(int(manual_line_map[arena_role]))
        if "LED_driver" in manual_line_map:
            self._led_driver_line.setValue(int(manual_line_map["LED_driver"]))
        if "L_eye_TTL" in manual_line_map:
            self._l_eye_line.setValue(int(manual_line_map["L_eye_TTL"]))
        if "R_eye_TTL" in manual_line_map:
            self._r_eye_line.setValue(int(manual_line_map["R_eye_TTL"]))

        self._extra_table.setRowCount(0)
        for role, line in sorted(manual_line_map.items(), key=lambda kv: kv[0]):
            if role in REQUIRED_TTL_ROLES or role == arena_role:
                continue
            self._add_extra_channel_row(str(role), int(line))

    def _on_apply_leech_mapping(self) -> None:
        line_map = self._leech_block.currentData()
        if not isinstance(line_map, dict):
            QtWidgets.QMessageBox.information(
                self,
                "Copy mapping",
                "Select a block to copy its TTL line mapping from.",
            )
            return
        self.apply_line_map({str(k): int(v) for k, v in line_map.items()})

    def _window_mode(self) -> str:
        if self._mode_sample.isChecked():
            return "s"
        if self._mode_auto.isChecked():
            return "a"
        return "i"

    def _on_auto_assign_eyes(self) -> None:
        try:
            assigned = auto_assign_eye_ttl_lines(
                self._blocksync,
                self._events_csv_path,
                int(self._eye_a.value()),
                int(self._eye_b.value()),
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Auto-assign", str(e))
            return
        self._l_eye_line.setValue(assigned["L_eye_TTL"])
        self._r_eye_line.setValue(assigned["R_eye_TTL"])

    def _on_open_browser_raster(self) -> None:
        try:
            self._blocksync._plot_ttl_raster(
                self._events_csv_path,
                title=f"{self._blocksync.block_num} TTL raster",
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Raster plot", str(e))

    def build_payload(self) -> tuple[dict[str, int], dict[str, int]]:
        """Validate current widget state and return mapping dicts (no side effects)."""
        return build_manual_ttl_payload(
            self._blocksync,
            self._events_csv_path,
            arena_line=int(self._arena_line.value()),
            l_eye_line=int(self._l_eye_line.value()),
            r_eye_line=int(self._r_eye_line.value()),
            led_driver_line=int(self._led_driver_line.value()),
            extra_roles=self._extra_roles_from_table(),
            window_mode=self._window_mode(),
            start_index=self._start_index.text().strip() or "0",
            end_index=self._end_index.text().strip() or "-1",
            start_sample=int(self._start_sample.value()),
            end_sample=int(self._end_sample.value()),
            gap_threshold_ms=float(self._gap_ms.value()),
            arena_channel_name=self._arena_channel_name,
        )

    def set_mapping_for_test(
        self,
        *,
        arena_line: int,
        l_eye_line: int,
        r_eye_line: int,
        led_driver_line: int = 4,
        window_mode: str = "i",
        start_index: str = "0",
        end_index: str = "-1",
        start_sample: int = 0,
        end_sample: int = 0,
        gap_threshold_ms: float = 1000.0,
    ) -> None:
        """Programmatic preset used by pytest."""
        self._arena_line.setValue(int(arena_line))
        self._l_eye_line.setValue(int(l_eye_line))
        self._r_eye_line.setValue(int(r_eye_line))
        self._led_driver_line.setValue(int(led_driver_line))
        self._start_index.setText(str(start_index))
        self._end_index.setText(str(end_index))
        self._start_sample.setValue(int(start_sample))
        self._end_sample.setValue(int(end_sample))
        self._gap_ms.setValue(float(gap_threshold_ms))
        if window_mode == "s":
            self._mode_sample.setChecked(True)
        elif window_mode == "a":
            self._mode_auto.setChecked(True)
        else:
            self._mode_index.setChecked(True)

    def payload(self) -> tuple[dict[str, int], dict[str, int]] | None:
        return self._manual_line_map, self._arena_window

    def _on_accept(self) -> None:
        try:
            manual_line_map, arena_window = self.build_payload()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Manual TTL mapping", str(e))
            return
        save_ttl_manual_sidecar(
            self._events_csv_path,
            self._blocksync,
            manual_line_map,
            arena_window,
        )
        self._manual_line_map = manual_line_map
        self._arena_window = arena_window
        self.accept()


def parse_open_ephys_with_manual_override(
    blocksync: BlockSync,
    manual_line_map: dict[str, int],
    arena_window: dict[str, int],
    *,
    overwrite: bool = True,
    arena_channel_name: str = "Arena_TTL",
    gap_threshold_ms: float = 1000.0,
    align_to_zero: bool = True,
) -> None:
    """Re-parse events with GUI-provided overrides (no ``input()`` fallback)."""
    blocksync.oe_events_to_csv(align_to_zero=align_to_zero)
    events_csv_path = blocksync.block_path / "oe_files" / blocksync.oe_dirname / "events.csv"
    ex_path = blocksync.block_path / "oe_files" / blocksync.oe_dirname / "parsed_events.csv"
    if overwrite and ex_path.is_file():
        ex_path.unlink()

    blocksync.oe_events, blocksync.arena_vid_first_t, blocksync.arena_vid_last_t = (
        blocksync.oe_events_parser(
            events_csv_path,
            blocksync.channeldict,
            arena_channel_name=arena_channel_name,
            export_path=ex_path,
            auto_break_selection=False,
            manual_line_map=manual_line_map,
            arena_window=arena_window,
            gap_threshold_ms=gap_threshold_ms,
        )
    )
    blocksync.l_vid_first_t = blocksync.oe_events["R_eye_TTL"].loc[
        blocksync.oe_events["R_eye_TTL_frame"].idxmin()
    ]
    blocksync.l_vid_last_t = blocksync.oe_events["R_eye_TTL"].loc[
        blocksync.oe_events["R_eye_TTL_frame"].idxmax()
    ]
    blocksync.r_vid_first_t = blocksync.oe_events["L_eye_TTL"].loc[
        blocksync.oe_events["L_eye_TTL_frame"].idxmin()
    ]
    blocksync.r_vid_last_t = blocksync.oe_events["L_eye_TTL"].loc[
        blocksync.oe_events["L_eye_TTL_frame"].idxmax()
    ]
