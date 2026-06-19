import json
import math
import os
import sys
import time
import copy
from functools import partial

import numpy as np
import pandas as pd
from vispy import app, scene

app.use_app('pyqt5')
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from vispy.scene import Mesh
from vispy.visuals.transforms import MatrixTransform
from vispy.visuals.filters import ShadingFilter
from vispy.geometry import Rect

# ---------- Optional (MP4 export) ----------
try:
    import imageio
    import imageio_ffmpeg

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

# ---------- Optional: trimesh for 3D file loading (external OBJ) ----------
try:
    import trimesh

    HAVE_TRIMESH = True
except Exception:
    HAVE_TRIMESH = False

# ===================== I/O & Columns =====================
TIME_COLS = ["ms_axis", "Unix Time", "Sample Unix Time", "Timestamp", "time", "Time (s)", "Time"]
ROLL_COLS = ["VQF Roll", "Roll (deg)", "Roll"]
PITCH_COLS = ["VQF Pitch", "Pitch (deg)", "Pitch"]
YAW_COLS = ["VQF Yaw", "Yaw (deg)", "Yaw"]

FRAME_COLS = ["frame", "Frame", "Unnamed: 0", "index", "Index", "", "Arena_frame"]

ARENA_VIDEO_FRAME_COLS = ["Arena_frame"]
LEFT_VIDEO_FRAME_COLS = ["L_eye_frame", "Left_eye_frame", "left_eye_frame"]
RIGHT_VIDEO_FRAME_COLS = ["R_eye_frame", "Right_eye_frame", "right_eye_frame"]

EYE_YAW_LEFT_COLS = ["yaw_left", "Yaw_left", "Yaw Left", "Left_Yaw", "LeftYaw", "yaw_L", "Yaw_L"]
EYE_PITCH_LEFT_COLS = ["pitch_left", "Pitch_left", "Pitch Left", "Left_Pitch", "LeftPitch", "pitch_L", "Pitch_L"]
EYE_YAW_RIGHT_COLS = ["yaw_right", "Yaw_right", "Yaw Right", "Right_Yaw", "RightYaw", "yaw_R", "Yaw_R"]
EYE_PITCH_RIGHT_COLS = ["pitch_right", "Pitch_right", "Pitch Right", "Right_Pitch", "RightPitch", "pitch_R", "Pitch_R"]


def read_table(path):
    return pd.read_excel(path, sheet_name=0) if path.lower().endswith(".xlsx") else pd.read_csv(path)


def pick_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Missing any of {candidates}. Columns present: {list(df.columns)}")


# ===================== Rotation mapping =====================
def RzRyRx(yaw_deg, pitch_deg, roll_deg, yaw_sign=+1.0):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_sign * yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


ALIGN_WORLD = np.diag([1.0, 1.0, 1.0])  # pre-multiply
ALIGN_LOCAL = np.diag([1.0, 1.0, 1.0])  # post-multiply

# --- Euler sign conventions (adjust here if a channel looks mirrored) ---
ROLL_SIGN = +1.0
PITCH_SIGN = +1.0
YAW_SIGN = +1.0


def rotation_from_ypr(roll_deg, pitch_deg, yaw_deg):
    R = RzRyRx(
        YAW_SIGN * yaw_deg,
        PITCH_SIGN * pitch_deg,
        ROLL_SIGN * roll_deg,
        yaw_sign=+1.0
    )
    return ALIGN_WORLD @ R @ ALIGN_LOCAL  # 3x3


# ===================== Geometry helpers =====================
def cuboid(center, size):
    cx, cy, cz = center;
    lx, ly, lz = size
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    quads = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    faces = []
    for a, b, c, d in quads: faces += [[a, b, c], [a, c, d]]
    return v, np.array(faces, dtype=np.uint32)


def cuboid_edge_positions_inflated(center, size, eps=0.003):
    center = np.asarray(center, np.float32)
    v, _ = cuboid(center, size)  # (8,3)
    diag = float(np.max(size)) if hasattr(size, "__iter__") else float(size)
    d = v - center
    n = np.linalg.norm(d, axis=1, keepdims=True);
    n[n == 0] = 1.0
    v_out = v + (d / n) * (eps * diag)
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    pts = np.empty((len(E) * 2, 3), dtype=np.float32)
    for k, (i, j) in enumerate(E): pts[2 * k], pts[2 * k + 1] = v_out[i], v_out[j]
    return pts


class NoWheelFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel:
            event.ignore()
            return True
        return False


# ===================== Axes cube with ticks =====================
class AxesCube(scene.Node):
    """Grid/ticks cube drawing three faces (left-view friendly defaults)."""

    def __init__(self, extent=6.0, step=1.0,
                 x_face='max', y_face='min', floor='min',
                 grid_color=(0.65, 0.65, 0.65, 0.55),
                 edge_color=(0.35, 0.35, 0.35, 1.0),
                 label_color='white',
                 base_font_size=96,
                 pixel_ratio=1.0,
                 viewport_height=800,
                 parent=None):
        super().__init__(parent=parent)
        self.step = float(step)
        self.base_font_size = int(base_font_size)
        self.pixel_ratio = float(pixel_ratio)
        self.viewport_height = int(viewport_height)
        self._x_face, self._y_face, self._floor = x_face, y_face, floor

        self.grids = scene.visuals.Line(color=grid_color, width=1.0, connect='segments', parent=self)
        self.grids.set_gl_state(depth_test=True, blend=True)
        self.edges = scene.visuals.Line(color=edge_color, width=2.0, connect='segments', parent=self)
        self.edges.set_gl_state(depth_test=True, blend=True)

        fs = self._font_px()
        self.labelX = scene.visuals.Text("X", color=label_color, method='cpu', font_size=fs,
                                         anchor_x='left', anchor_y='top', parent=self)
        self.labelY = scene.visuals.Text("Y", color=label_color, method='cpu', font_size=fs,
                                         anchor_x='left', anchor_y='top', parent=self)
        self.labelZ = scene.visuals.Text("Z", color=label_color, method='cpu', font_size=fs,
                                         anchor_x='left', anchor_y='top', parent=self)
        for t in (self.labelX, self.labelY, self.labelZ): t.set_gl_state(depth_test=True, blend=True)

        self.xticks, self.yticks, self.zticks = [], [], []
        self.set_extent(extent)

    def _font_px(self):
        scale_h = max(0.6, min(2.2, self.viewport_height / 800.0))
        return int(max(12, min(192, round(self.base_font_size * self.pixel_ratio * scale_h))))

    def set_screen_params(self, pixel_ratio: float, viewport_height: int):
        self.pixel_ratio = float(max(0.5, pixel_ratio))
        self.viewport_height = int(max(300, viewport_height))
        fs = self._font_px()
        for t in (self.labelX, self.labelY, self.labelZ): t.font_size = fs
        for coll in (self.xticks, self.yticks, self.zticks):
            for t in coll: t.font_size = fs

    def set_base_font_size(self, size: int):
        self.base_font_size = int(size)
        self.set_screen_params(self.pixel_ratio, self.viewport_height)

    def _clear_ticks(self):
        for coll in (self.xticks, self.yticks, self.zticks):
            for t in coll: t.parent = None
        self.xticks, self.yticks, self.zticks = [], [], []

    @staticmethod
    def _fmt_tick(v):
        if abs(v) < 1e-9: return "0"
        r = round(v)
        return f"{int(r)}" if abs(v - r) < 1e-6 else f"{v:.2f}"

    def set_extent(self, extent):
        if np.isscalar(extent):
            xmin = ymin = zmin = -float(extent);
            xmax = ymax = zmax = +float(extent)
        else:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = extent
        s = self.step

        xw = xmin if self._x_face == 'min' else xmax
        yw = ymin if self._y_face == 'min' else ymax
        zf = zmin if self._floor == 'min' else zmax

        segs = []
        # floor
        for x in np.arange(xmin, xmax + 1e-9, s): segs += [[x, ymin, zf], [x, ymax, zf]]
        for y in np.arange(ymin, ymax + 1e-9, s): segs += [[xmin, y, zf], [xmax, y, zf]]
        # X wall
        for y in np.arange(ymin, ymax + 1e-9, s): segs += [[xw, y, zmin], [xw, y, zmax]]
        for z in np.arange(zmin, zmax + 1e-9, s): segs += [[xw, ymin, z], [xw, ymax, z]]
        # Y wall
        for x in np.arange(xmin, xmax + 1e-9, s): segs += [[x, yw, zmin], [x, yw, zmax]]
        for z in np.arange(zmin, zmax + 1e-9, s): segs += [[xmin, yw, z], [xmax, yw, z]]

        self.grids.set_data(np.asarray(segs, np.float32), connect='segments')

        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
        ], dtype=np.float32)
        E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        edge_pts = np.empty((len(E) * 2, 3), np.float32)
        for k, (i, j) in enumerate(E): edge_pts[2 * k], edge_pts[2 * k + 1] = corners[i], corners[j]
        self.edges.set_data(edge_pts, connect='segments')

        L = max(xmax - xmin, ymax - ymin, zmax - zmin);
        off = 0.06 * L;
        eps = 0.025 * L
        self.labelX.pos = (xmax - off, yw + off, zf + off)
        self.labelY.pos = (xw - off, ymax - off if self._y_face == 'max' else ymin + off, zf + off)
        self.labelZ.pos = (xw - off, yw + off, zmax - off if self._floor == 'min' else zmin + off)

        self._clear_ticks();
        fs = self._font_px()
        for x in np.arange(xmin, xmax + 1e-9, s):
            t = scene.visuals.Text(self._fmt_tick(x), color='white', method='cpu', font_size=fs,
                                   anchor_x='center', anchor_y='top', parent=self)
            t.pos = (x, yw + eps if self._y_face == 'min' else yw - eps, zf + eps);
            t.set_gl_state(depth_test=True, blend=True)
            self.xticks.append(t)
        for y in np.arange(ymin, ymax + 1e-9, s):
            t = scene.visuals.Text(self._fmt_tick(y), color='white', method='cpu', font_size=fs,
                                   anchor_x='right', anchor_y='center', parent=self)
            x_pos = xw - eps if self._x_face == 'max' else xw + eps
            t.pos = (x_pos, y, zf + eps);
            t.set_gl_state(depth_test=True, blend=True)
            self.yticks.append(t)
        for z in np.arange(zmin, zmax + 1e-9, s):
            t = scene.visuals.Text(self._fmt_tick(z), color='white', method='cpu', font_size=fs,
                                   anchor_x='right', anchor_y='top', parent=self)
            x_pos = xw - eps if self._x_face == 'max' else xw + eps
            y_pos = yw + eps if self._y_face == 'min' else yw - eps
            t.pos = (x_pos, y_pos, z);
            t.set_gl_state(depth_test=True, blend=True)
            self.zticks.append(t)


# ---------- Export-range dialog ----------
class ExportRangeDialog(QtWidgets.QDialog):
    def __init__(self, parent, n_frames, current_idx):
        super().__init__(parent)
        self.setWindowTitle("Export range…");
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)
        self.start = QtWidgets.QSpinBox();
        self.start.setRange(0, n_frames - 1);
        self.start.setValue(current_idx)
        self.end = QtWidgets.QSpinBox();
        self.end.setRange(0, n_frames - 1);
        self.end.setValue(n_frames - 1)
        self.step = QtWidgets.QSpinBox();
        self.step.setRange(1, max(1, n_frames // 10));
        self.step.setValue(1)
        self.fps = QtWidgets.QSpinBox();
        self.fps.setRange(1, 120);
        self.fps.setValue(30)
        form.addRow("Start frame:", self.start);
        form.addRow("End frame:", self.end)
        form.addRow("Step:", self.step);
        form.addRow("FPS:", self.fps)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept);
        btns.rejected.connect(self.reject);
        form.addRow(btns)

    def values(self):
        s = int(self.start.value());
        e = int(self.end.value())
        if e < s: s, e = e, s
        return s, e, int(self.step.value()), int(self.fps.value())


class ColumnSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent, columns, current_map):
        super().__init__(parent)
        self.setWindowTitle("Select Data Columns")
        self.resize(300, 150)
        layout = QtWidgets.QFormLayout(self)

        self.combos = {}
        self.keys = ["Time", "Roll", "Pitch", "Yaw"]

        for key in self.keys:
            cb = QtWidgets.QComboBox()
            cb.addItems(columns)
            curr = current_map.get(key)
            if curr in columns: cb.setCurrentIndex(columns.index(curr))
            layout.addRow(key, cb)
            self.combos[key] = cb

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_selection(self):
        return {key: cb.currentText() for key, cb in self.combos.items()}


class AxisGizmo3D:
    """
    Small 3D inset showing the WORLD axes as seen by the current camera.

    Implemented as a VisPy ViewBox that is rendered inside the same SceneCanvas.
    """

    def __init__(self, canvas: scene.SceneCanvas, parent_scene, size: int = 140, margin: int = 12):
        self.canvas = canvas
        self.size = int(size)
        self.margin = int(margin)

        # Inset view rendered on top of the main view
        self.view = scene.widgets.ViewBox(parent=parent_scene)
        self.view.order = 10
        self.view.interactive = False
        self.view.camera = scene.cameras.TurntableCamera(fov=0.0, distance=3.5, elevation=0.0, azimuth=0.0, up='+z')
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.interactive = False

        # Background (subtle dark quad)
        bg = scene.visuals.Rectangle(center=(0, 0), width=2.8, height=2.8,
                                     color=(0, 0, 0, 0.35), parent=self.view.scene)
        bg.set_gl_state(depth_test=False, blend=True)

        L = 1.3  # axis length (in gizmo local units)
        o = np.array([0.0, 0.0, 0.0], np.float32)

        # Lines (X=red, Y=green, Z=blue)
        self.x_line = scene.visuals.Line(pos=np.vstack([o, [L, 0, 0]]).astype(np.float32),
                                         color=(1.0, 0.3, 0.3, 1.0), width=4, parent=self.view.scene)
        self.y_line = scene.visuals.Line(pos=np.vstack([o, [0, L, 0]]).astype(np.float32),
                                         color=(0.3, 1.0, 0.4, 1.0), width=4, parent=self.view.scene)
        self.z_line = scene.visuals.Line(pos=np.vstack([o, [0, 0, L]]).astype(np.float32),
                                         color=(0.35, 0.6, 1.0, 1.0), width=4, parent=self.view.scene)
        for ln in (self.x_line, self.y_line, self.z_line):
            ln.set_gl_state(depth_test=False, blend=True)

        # Labels
        self.x_lbl = scene.visuals.Text("X", color="white", font_size=18, anchor_x='left', anchor_y='center',
                                        parent=self.view.scene)
        self.y_lbl = scene.visuals.Text("Y", color="white", font_size=18, anchor_x='left', anchor_y='center',
                                        parent=self.view.scene)
        self.z_lbl = scene.visuals.Text("Z", color="white", font_size=18, anchor_x='left', anchor_y='center',
                                        parent=self.view.scene)

        self.x_lbl.pos = (L + 0.12, 0.0, 0.0)
        self.y_lbl.pos = (0.0, L + 0.12, 0.0)
        self.z_lbl.pos = (0.0, 0.0, L + 0.12)

        for t in (self.x_lbl, self.y_lbl, self.z_lbl):
            t.set_gl_state(depth_test=False, blend=True)

        self.update_rect()

    def update_rect(self):
        w, h = self.canvas.size
        x = int(w - self.size - self.margin)
        y = int(self.margin)
        self.view.rect = Rect(x, y, int(self.size), int(self.size))

    def sync_from_camera(self, cam):
        # Match the main camera's orientation (so the inset rotates with your view)
        self.view.camera.azimuth = float(getattr(cam, "azimuth", 0.0))
        self.view.camera.elevation = float(getattr(cam, "elevation", 0.0))
        self.view.camera.up = getattr(cam, "up", "+z")


# ===================== Player =====================
class Player(QtWidgets.QMainWindow):
    # Default to RIGHT view & zoomed out
    CAM_ELEV = 0
    CAM_AZIM = 90
    CAM_DIST = 24.0
    CAM_CENTER = (0, 0, 0)
    CAM_RANGE = dict(x=(-6, 6), y=(-6, 6), z=(-6, 6))

    @staticmethod
    def _fmt_clock(seconds: float) -> str:
        cs_total = int(round(max(0.0, float(seconds)) * 100.0))
        cs = cs_total % 100;
        s_total = cs_total // 100;
        s = s_total % 60
        m_total = s_total // 60;
        m = m_total % 60;
        h = m_total // 60
        return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}" if h else f"{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def _fmt_hms(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600);
        m = int((seconds % 3600) // 60);
        s = int(seconds % 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def __init__(self, file_path=None):
        super().__init__()
        self.setWindowTitle("Interactive 3D Reconstruction")

        # ---- Load data (optional) ----
        if file_path:
            try:
                self.df = read_table(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()

        # ---- Detect Defaults & Compute ----
        self.col_map = self._detect_default_columns(self.df)
        self._recalc_trajectory()

        # ---- Playback marks / looping ----
        self.mark_in, self.mark_out = 0, max(0, self.n - 1)
        self.loop_selection = False
        self.hud_marks = f"IN {self.mark_in + 1}  OUT {self.mark_out + 1}" if self.n > 0 else ""

        # ---- Embedded videos state ----
        self.video_cache_limit = 32
        self.video_scale_to_fit = True
        self.video_panel_visible = True

        self.arena_video_variants = []
        self.active_arena_video_idx = 0
        self.max_arena_videos = 4

        self.video_slots = {
            "left": {
                "title": "Left Eye Video",
                "path": None,
                "reader": None,
                "meta": {},
                "nframes": 0,
                "cache": {},
                "rotation_quadrants": 0,
                "flip_h": False,
                "flip_v": False,
            },
            "right": {
                "title": "Right Eye Video",
                "path": None,
                "reader": None,
                "meta": {},
                "nframes": 0,
                "cache": {},
                "rotation_quadrants": 0,
                "flip_h": False,
                "flip_v": False,
            },
        }

        self.arena_video_frame_data = np.array([], dtype=float)
        self.left_video_frame_data = np.array([], dtype=float)
        self.right_video_frame_data = np.array([], dtype=float)

        # ---- UI ----
        central = QtWidgets.QWidget(self);
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central);
        vbox.setContentsMargins(8, 8, 8, 8);
        vbox.setSpacing(6)

        # ===== Main row: canvas (left) + side panels (right) =====
        main_row = QtWidgets.QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(8)
        vbox.addLayout(main_row, stretch=1)

        # Right-side panel container (scrollable)
        self.side_scroll = QtWidgets.QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setMinimumWidth(520)
        self.side_scroll.setMaximumWidth(800)
        self.side_panel = QtWidgets.QFrame()
        self.side_vbox = QtWidgets.QVBoxLayout(self.side_panel)
        self.side_vbox.setContentsMargins(6, 6, 6, 6)
        self.side_vbox.setSpacing(8)
        self.side_scroll.setWidget(self.side_panel)

        # Hidden until any panel is opened
        self.side_scroll.setVisible(False)

        try:
            self.canvas = scene.SceneCanvas(keys='interactive', size=(1000, 800), show=False, config={'samples': 8})
        except TypeError:
            self.canvas = scene.SceneCanvas(keys='interactive', size=(1000, 800), show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(elevation=self.CAM_ELEV, azimuth=self.CAM_AZIM,
                                                         fov=45, distance=self.CAM_DIST, up='+z')
        # ---- keep axis gizmo synced with mouse-driven camera motion ----
        try:
            self.view.camera.events.transform_change.connect(lambda ev: self._update_axis_gizmo())
        except Exception:
            pass

        # Fallback: update on canvas mouse events too
        try:
            self.canvas.events.mouse_move.connect(lambda ev: self._update_axis_gizmo())
            self.canvas.events.mouse_wheel.connect(lambda ev: self._update_axis_gizmo())
            self.canvas.events.mouse_press.connect(lambda ev: self._update_axis_gizmo())
            self.canvas.events.mouse_release.connect(lambda ev: self._update_axis_gizmo())
        except Exception:
            pass

        self.view.camera._elevation_lim = (-180.0, 180.0)  # allow full range
        self.view.camera.center = self.CAM_CENTER

        # ---- 3D axis gizmo (inset, rendered inside the VisPy canvas) ----
        self.axis_gizmo = AxisGizmo3D(self.canvas, self.canvas.scene, size=150, margin=12)
        self.axis_gizmo.sync_from_camera(self.view.camera)
        # keep inset placed correctly on resize
        self.canvas.events.resize.connect(lambda _ev: self.axis_gizmo.update_rect())
        # keep it synced while you interact with the camera
        self.canvas.events.draw.connect(lambda _ev: self.axis_gizmo.sync_from_camera(self.view.camera))

        # ---- Scene graph ----
        self.root = scene.Node(parent=self.view.scene)  # animated by CSV rotations

        # BNO-origin correction node lives between root and model
        self.bno_origin_node = scene.Node(parent=self.root)
        self.bno_origin_transform = MatrixTransform()
        self.bno_origin_node.transform = self.bno_origin_transform

        self.model_node = scene.Node(parent=self.bno_origin_node)  # moved under bno_origin_node
        self.model_transform = MatrixTransform()
        self.model_node.transform = self.model_transform
        self.model_scale = 1.0

        # BNO-origin state (editable in UI)
        self.bno_origin_enabled = True
        self.bno_origin_rpy = np.array([0.0, 0.0, 0.0], dtype=float)  # [roll0, pitch0, yaw0] in deg
        self.bno_origin_xyz = np.array([0.0, 0.0, 0.0], dtype=float)  # [x0, y0, z0] in scene units

        # ---- Geometry in local space ----
        BOARD_LEN, BOARD_WID, BOARD_HEI = 1.0, 4.0, 5.0
        BNO_LEN, BNO_WID, BNO_HEI = 2.5, 0.5, 2.0
        STAND_LEN, STAND_WID, STAND_HEI = 1.5, 0.25, 1.0

        BOARD_CENTER_LOCAL = np.array([-3.25, 0.0, 0.0], dtype=np.float32)
        BNO_CENTER_LOCAL = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        STAND_CENTER_LOCAL = np.array([-2.0, 0.0, 0.0], dtype=np.float32)

        board_v, board_f = cuboid(BOARD_CENTER_LOCAL, (BOARD_LEN, BOARD_WID, BOARD_HEI))
        bno_v, bno_f = cuboid(BNO_CENTER_LOCAL, (BNO_LEN, BNO_WID, BNO_HEI))
        stand_v, stand_f = cuboid(STAND_CENTER_LOCAL, (STAND_LEN, STAND_WID, STAND_HEI))

        BOARD_COLOR = (0.78, 0.86, 0.94, 1.0)
        BNO_COLOR = (0.12, 0.46, 0.95, 1.0)
        STAND_COLOR = (0.55, 0.72, 0.85, 1.0)

        self.board = Mesh(vertices=board_v, faces=board_f, color=BOARD_COLOR, shading=None, parent=self.model_node)
        self.board.set_gl_state(preset='opaque', depth_test=True, blend=False, polygon_offset_fill=True,
                                polygon_offset=(6.0, 6.0))
        self.stand = Mesh(vertices=stand_v, faces=stand_f, color=STAND_COLOR, shading=None, parent=self.model_node)
        self.stand.set_gl_state(preset='opaque', depth_test=True, blend=False, polygon_offset_fill=True,
                                polygon_offset=(6.0, 6.0))
        self.bno = Mesh(vertices=bno_v, faces=bno_f, color=BNO_COLOR, shading=None, parent=self.model_node)
        self.bno.set_gl_state(preset='opaque', depth_test=True, blend=False, polygon_offset_fill=True,
                              polygon_offset=(6.0, 6.0))

        EDGE_COLOR = (0.25, 0.30, 0.35, 1.0);
        EDGE_WIDTH = 2.0;
        _eps = 0.003
        self.board_edges = scene.visuals.Line(
            cuboid_edge_positions_inflated(BOARD_CENTER_LOCAL, (BOARD_LEN, BOARD_WID, BOARD_HEI), eps=_eps),
            color=EDGE_COLOR, width=EDGE_WIDTH, connect='segments', parent=self.model_node)
        self.board_edges.set_gl_state(depth_test=True, blend=True)
        self.bno_edges = scene.visuals.Line(
            cuboid_edge_positions_inflated(BNO_CENTER_LOCAL, (BNO_LEN, BNO_WID, BNO_HEI), eps=_eps),
            color=EDGE_COLOR, width=EDGE_WIDTH, connect='segments', parent=self.model_node)
        self.bno_edges.set_gl_state(depth_test=True, blend=True)
        self.stand_edges = scene.visuals.Line(
            cuboid_edge_positions_inflated(STAND_CENTER_LOCAL, (STAND_LEN, STAND_WID, STAND_HEI), eps=_eps),
            color=EDGE_COLOR, width=EDGE_WIDTH, connect='segments', parent=self.model_node)
        self.stand_edges.set_gl_state(depth_test=True, blend=True)

        # ----- External object (OBJ) holder -----
        self.external_node = scene.Node(parent=self.root)
        self.external_transform = MatrixTransform()
        self.external_node.transform = self.external_transform
        self.external_mesh_visuals = []
        self.external_visible = True
        self._external_auto_scale = True
        self._external_manual_scale = 1.0
        self.last_obj_path = None  # track last loaded path for settings

        # --- BNO axis triad (OFF by default) ---
        hx, hy, hz = BNO_LEN * 0.5, BNO_WID * 0.5, BNO_HEI * 0.5
        ax_base = 1.4 * max(BNO_LEN, BNO_WID, BNO_HEI)
        axis_lengths = [ax_base, ax_base, ax_base]
        eps = 0.02 * ax_base

        origins = [np.array([+hx + eps, 0.0, 0.0], np.float32),
                   np.array([0.0, hy + eps, 0.0], np.float32),
                   np.array([0.0, 0.0, +hz + eps], np.float32)]
        dirs = [np.array([1, 0, 0], np.float32),
                np.array([0, 1, 0], np.float32),
                np.array([0, 0, 1], np.float32)]
        PURPLE = (0.75, 0.35, 1.00, 1.0)
        colors = [PURPLE, (0.30, 0.95, 0.40, 1.0), (0.35, 0.60, 1.00, 1.0)]

        self.bno_axes = []
        for o, d, c, L in zip(origins, dirs, colors, axis_lengths):
            pos = np.vstack([o, o + d * L]).astype(np.float32)
            a = scene.visuals.Line(pos=pos, color=c, width=4, connect='segments', parent=self.model_node)
            a.set_gl_state(depth_test=True, blend=True);
            a.visible = False
            self.bno_axes.append(a)

        # --- Direction vector ---
        dir_origin = np.array([+hx + eps, 0.0, 0.0], np.float32)
        dir_vec = np.array([+1.0, 0.0, 0.0], np.float32)
        dir_len = ax_base
        dir_pos = np.vstack([dir_origin, dir_origin + dir_vec * dir_len]).astype(np.float32)
        self.dir_arrow = scene.visuals.Line(pos=dir_pos, color=(1.00, 0.25, 0.25, 1.0),
                                            width=5, connect='segments', parent=self.model_node)
        self.dir_arrow.set_gl_state(depth_test=True, blend=True)
        self.dir_arrow.visible = True

        # --- Manual eye vectors (world-relative, NOT animated by self.root) ---
        # Parameters live in scene/world coordinates (same as axes cube / camera).
        self.eye_params = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "length": 2.0},
            {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "length": 2.0},
        ]
        # Keep a default snapshot for resets
        self.eye_defaults = copy.deepcopy(self.eye_params)

        # Create the visuals as children of view.scene, so they stay fixed in world axes
        # (unlike self.root which is animated by CSV rotations).
        # Create the visuals as children of view.scene, so they stay fixed in world axes
        self.eye_lines = []
        eye_colors = [
            (0.20, 0.90, 0.20, 1.0),  # Eye A: green
            (0.20, 0.60, 1.00, 1.0),  # Eye B: blue
        ]
        for k in range(2):
            pos = np.zeros((2, 3), dtype=np.float32)
            v = scene.visuals.Line(
                pos=pos,
                color=eye_colors[k],
                width=5,
                connect='segments',
                parent=self.view.scene
            )
            v.set_gl_state(depth_test=True, blend=True)
            v.visible = False
            self.eye_lines.append(v)

        # Initialize geometry
        self._update_eye_vector(0)
        self._update_eye_vector(1)

        # ---- Eye vectors follow-model (capture world -> model_local, then animate via model transform) ----
        self.eye_follow_model = False  # global toggle
        self.eye_local_segments = [None, None]  # each: (origin_local(3,), end_local(3,))

        # ---- Per-row eye direction mode (anchor fixed on head, direction varies by DATA ROW) ----
        self.eye_pf_enabled = False
        self.eye_pf_source_path = ""
        self.eye_pf_series = [None, None]  # per eye: ndarray shape (n_data, 2) -> [yaw, pitch]
        self.eye_pf_anchor_local = [None, None]
        self.eye_pf_last_dirH = [None, None]
        self.eye_pf_hold_last = True
        self.eye_pf_swap_lr = False
        self.eye_pf_first_row = [None, None]  # first valid DATA ROW for each eye
        self.eye_pf_prev_i = None

        self.prepend_zero_pose_frame = True
        self.data_index_offset = 0

        self.arena_frame_data = None  # actual Arena_frame values aligned to data rows
        self.t_abs_data = None  # absolute ms_axis values aligned to data rows
        self.gui_time = None  # displayed GUI time, including virtual frame
        self.first_head_ms = None

        # Root transform & first frame index
        self.root.transform = MatrixTransform();
        self.i = 0
        self.neutral_pose = False  # show objects at Yaw=Pitch=Roll=0 when True

        # ---- Axes cube ----
        all_verts = np.vstack([board_v, bno_v, stand_v])
        R = float(np.linalg.norm(all_verts, axis=1).max()) + 0.5
        pr = getattr(self.canvas, 'pixel_ratio', 1.0);
        vh = self.canvas.size[1]
        self.axes_base_extent = R
        self.axes_scale = 1.0
        self.axes_cube = AxesCube(
            extent=R,
            step=1.0,
            x_face='min',
            y_face='min',
            floor='min',
            base_font_size=96,
            pixel_ratio=pr,
            viewport_height=vh,
            parent=self.view.scene
        )

        # ===== Canvas area + HUD =====
        self.canvas_frame = QtWidgets.QFrame()
        grid = QtWidgets.QGridLayout(self.canvas_frame);
        grid.setContentsMargins(0, 0, 0, 0);
        grid.setSpacing(0)
        grid.addWidget(self.canvas.native, 0, 0)
        self.hud_lbl = QtWidgets.QLabel("")
        self.hud_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.hud_lbl.setStyleSheet(
            "QLabel { color: white; background: rgba(0,0,0,120); padding: 6px 8px; font: 13pt 'Menlo','Courier New',monospace; }")
        self.hud_lbl.setVisible(True)
        grid.addWidget(self.hud_lbl, 0, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # ===== Reconstruction + Videos area =====
        # New layout:
        #   left  -> reconstruction canvas
        #   right -> video column
        #              top    : large Arena video
        #              bottom : Left/Right eye videos side by side
        self.visuals_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # left: reconstruction canvas
        self.visuals_splitter.addWidget(self.canvas_frame)

        # right: video column
        self.video_panel = QtWidgets.QFrame()
        self.video_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.video_panel_vbox = QtWidgets.QVBoxLayout(self.video_panel)
        self.video_panel_vbox.setContentsMargins(4, 4, 4, 4)
        self.video_panel_vbox.setSpacing(6)

        self.video_widgets = {}

        def make_video_box(key, min_w, min_h):
            box = QtWidgets.QFrame()
            box.setFrameShape(QtWidgets.QFrame.StyledPanel)
            box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

            box_vbox = QtWidgets.QVBoxLayout(box)
            box_vbox.setContentsMargins(4, 4, 4, 4)
            box_vbox.setSpacing(4)

            title_row = QtWidgets.QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(6)

            if key == "arena":
                prev_btn = QtWidgets.QToolButton()
                prev_btn.setText("◀")
                prev_btn.clicked.connect(self._prev_arena_video)

                next_btn = QtWidgets.QToolButton()
                next_btn.setText("▶")
                next_btn.clicked.connect(self._next_arena_video)

                title = QtWidgets.QLabel("Arena Video")
                title.setStyleSheet("font-weight: bold;")
                title.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
                title.setWordWrap(False)
                title.setMinimumWidth(0)

                title_row.addWidget(prev_btn)
                title_row.addWidget(title, stretch=1)
                title_row.addWidget(next_btn)
            else:
                title = QtWidgets.QLabel(self.video_slots[key]["title"])
                title.setStyleSheet("font-weight: bold;")
                title.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
                title.setWordWrap(False)
                title.setMinimumWidth(0)

                prev_btn = None
                next_btn = None
                title_row.addWidget(title, stretch=1)

            label = QtWidgets.QLabel("No video loaded")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumSize(min_w, min_h)
            label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            label.setStyleSheet("background: black; color: white;")

            info = QtWidgets.QLabel("")
            info.setWordWrap(False)
            info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            info.setMinimumWidth(0)
            info.setMaximumHeight(22)
            info.setStyleSheet("QLabel { padding-top: 2px; }")

            box_vbox.addLayout(title_row)
            box_vbox.addWidget(label, stretch=1)
            box_vbox.addWidget(info)

            self.video_widgets[key] = {
                "frame": box,
                "title": title,
                "label": label,
                "info": info,
                "prev_btn": prev_btn,
                "next_btn": next_btn,
            }
            return box

        # Large arena video on top
        arena_box = make_video_box("arena", 420, 280)
        self.video_panel_vbox.addWidget(arena_box, stretch=3)

        # Two eye videos below
        eyes_row = QtWidgets.QHBoxLayout()
        eyes_row.setContentsMargins(0, 0, 0, 0)
        eyes_row.setSpacing(6)

        left_box = make_video_box("left", 220, 160)
        right_box = make_video_box("right", 220, 160)

        eyes_row.addWidget(right_box, stretch=1)
        eyes_row.addWidget(left_box, stretch=1)

        eyes_row_widget = QtWidgets.QWidget()
        eyes_row_widget.setLayout(eyes_row)
        eyes_row_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.video_panel_vbox.addWidget(eyes_row_widget, stretch=2)

        self.visuals_splitter.addWidget(self.video_panel)

        # Give most width to the 3D view, but keep a substantial video column
        self.visuals_splitter.setSizes([1150, 650])

        main_row.addWidget(self.visuals_splitter, stretch=1)
        main_row.addWidget(self.side_scroll, stretch=0)

        # ===== Top controls row =====
        controls = QtWidgets.QHBoxLayout();
        vbox.addLayout(controls)
        self.btn_back = QtWidgets.QPushButton("⟨⟨ Step Back")
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_fwd = QtWidgets.QPushButton("Step Fwd ⟩⟩")

        self.step_jump_lbl = QtWidgets.QLabel("Step size:")
        self.step_jump = QtWidgets.QSpinBox()
        self.step_jump.setRange(1, 1000000)
        self.step_jump.setValue(1)
        self.step_jump.setSingleStep(1)

        self.speed_lbl = QtWidgets.QLabel("Speed (x):")
        self.speed = QtWidgets.QDoubleSpinBox();
        self.speed.setRange(0.1, 10.0);
        self.speed.setValue(1.0)
        self.fps_lbl = QtWidgets.QLabel("FPS:")
        self.fps = QtWidgets.QDoubleSpinBox()
        self.fps.setRange(1.0, 240.0);
        self.fps.setDecimals(2);
        self.fps.setSingleStep(0.01);
        self.fps.setValue(60)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal);
        self.slider.setRange(0, self.n - 1);
        self.slider.setValue(0)

        # ---- Data Menu ----
        self.data_menu = QtWidgets.QMenu(self)
        self.data_menu.addAction("Load File…", self._load_new_file)
        self.data_menu.addAction("Select Columns…", self._prompt_columns)
        self.data_menu.addSeparator()
        self.data_menu.addAction("Export CSV with Eye Directions (World)…", self._export_csv_with_eye_dirs_world)
        self.data_btn = QtWidgets.QToolButton()
        self.data_btn.setText("Data ▾")
        self.data_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.data_btn.setMenu(self.data_menu)

        # ---- Scene menu (Save/Load settings) ----
        self.scene_menu = QtWidgets.QMenu(self)
        self.scene_menu.addAction("Save Scene Settings…", self._save_scene_settings)
        self.scene_menu.addAction("Load Scene Settings…", self._load_scene_settings)
        self.scene_btn = QtWidgets.QToolButton();
        self.scene_btn.setText("Scene ▾")
        self.scene_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup);
        self.scene_btn.setMenu(self.scene_menu)

        # ---- View menu ----
        self.view_menu = QtWidgets.QMenu(self)
        self.view_menu.addAction("Reset view", self._reset_view)
        self.act_lock_view = self.view_menu.addAction("Lock view")
        self.act_lock_view.setCheckable(True);
        self.act_lock_view.toggled.connect(self._set_lock_view)
        self.view_menu.addSeparator()
        self.view_menu.addAction("Flip view (roll 180°)", self._flip_view_roll)
        self.act_axes_cube = self.view_menu.addAction("Show axes cube")
        self.act_axes_cube.setCheckable(True);
        self.act_axes_cube.setChecked(True)
        self.act_axes_cube.toggled.connect(lambda v: setattr(self.axes_cube, "visible", bool(v)))
        self.act_bno_axes = self.view_menu.addAction("Show BNO axes")
        self.act_bno_axes.setCheckable(True);
        self.act_bno_axes.setChecked(False)
        self.act_bno_axes.toggled.connect(self._toggle_bno_axes)
        self.act_dir_vec = self.view_menu.addAction("Show direction vector")
        self.act_dir_vec.setCheckable(True);
        self.act_dir_vec.setChecked(True)
        self.act_dir_vec.toggled.connect(self._toggle_dir_vec)
        self.act_tick_size = self.view_menu.addAction("Tick size…", self._change_tick_size)
        self.view_menu.addSeparator()
        self.act_show_view_controls = self.view_menu.addAction("Show View Controls")
        self.act_show_view_controls.setCheckable(True);
        self.act_show_view_controls.setChecked(False)
        self.act_show_view_controls.toggled.connect(lambda v: self.view_controls_group.setVisible(bool(v)))
        self.act_show_view_controls.toggled.connect(lambda _v: self._update_side_panel_visibility())
        self.act_neutral = self.view_menu.addAction("Neutral orientation (Yaw=Pitch=Roll=0)")
        self.act_neutral.setCheckable(True);
        self.act_neutral.setChecked(False)
        self.act_neutral.toggled.connect(self._toggle_neutral)
        self.view_menu.addSeparator()
        self.act_hide_board_only = self.view_menu.addAction("Hide board only")
        self.act_hide_board_only.setCheckable(True)
        self.act_hide_board_only.toggled.connect(self._toggle_board_only)
        self.act_hide_board_only.setChecked(True)
        self._toggle_board_only(True)
        self.view_menu.addSeparator()
        self.view_menu.addAction("Top view (1)", partial(self._snap_view, 90, 90))
        self.view_menu.addAction("Front view (2)", partial(self._snap_view, 0, 90))
        self.view_menu.addAction("Right view (3)", partial(self._snap_view, 0, 180))
        self.view_menu.addSeparator()
        self.act_show_hud = self.view_menu.addAction("Show HUD")
        self.act_show_hud.setCheckable(True);
        self.act_show_hud.setChecked(True)
        self.act_show_hud.toggled.connect(self.hud_lbl.setVisible)
        self.view_btn = QtWidgets.QToolButton();
        self.view_btn.setText("View ▾")
        self.view_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup);
        self.view_btn.setMenu(self.view_menu)

        # ---- Playback menu ----
        self.play_menu = QtWidgets.QMenu(self)
        self.play_menu.addAction("Mark In    [", self._mark_in)
        self.play_menu.addAction("Mark Out   ]", self._mark_out)
        self.act_loop = self.play_menu.addAction("Loop selection (L)")
        self.act_loop.setCheckable(True);
        self.act_loop.setChecked(False)
        self.act_loop.toggled.connect(self._toggle_loop)
        self.play_btn = QtWidgets.QToolButton();
        self.play_btn.setText("Playback ▾")
        self.play_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup);
        self.play_btn.setMenu(self.play_menu)

        # ---- Export menu ----
        self.export_menu = QtWidgets.QMenu(self)
        self.export_menu.addAction("Export MP4", self._export_mp4)
        self.export_menu.addAction("Export range…", self._export_range)
        self.export_menu.addSeparator()
        self.act_include_hud = self.export_menu.addAction("Include HUD in export")
        self.act_include_hud.setCheckable(True);
        self.act_include_hud.setChecked(False)
        self.export_menu.addSeparator()
        self.act_cancel_export = self.export_menu.addAction("Cancel export", self._cancel_export)
        self.act_cancel_export.setEnabled(False)
        self.export_btn = QtWidgets.QToolButton();
        self.export_btn.setText("Export ▾")
        self.export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup);
        self.export_btn.setMenu(self.export_menu)

        # ---- Objects menu (external OBJ + visibility + transform panel toggle) ----
        self.objects_menu = QtWidgets.QMenu(self)
        self.act_load_obj = self.objects_menu.addAction("Load .OBJ…", self._browse_obj)
        self.objects_menu.addSeparator()
        self.act_show_obj = self.objects_menu.addAction("Show external object")
        self.act_show_obj.setCheckable(True);
        self.act_show_obj.setChecked(True)
        self.act_show_obj.toggled.connect(self._toggle_external_visibility)
        self.act_hide_boards = self.objects_menu.addAction("Hide boards/BNO")
        self.act_hide_boards.setCheckable(True);
        self.act_hide_boards.setChecked(False)
        self.act_hide_boards.toggled.connect(self._toggle_board_visibility)
        self.objects_menu.addSeparator()
        self.act_show_transform = self.objects_menu.addAction("Show Transform Controls")
        self.act_show_transform.setCheckable(True);
        self.act_show_transform.setChecked(False)

        self.objects_btn = QtWidgets.QToolButton();
        self.objects_btn.setText("Objects ▾")
        self.objects_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup);
        self.objects_btn.setMenu(self.objects_menu)

        # ---- Video menu ----
        self.video_menu = QtWidgets.QMenu(self)

        self.video_menu.addAction("Load Arena Videos…", self._browse_arena_videos)
        self.video_menu.addAction("Load Left Eye Video…", lambda: self._browse_video("left"))
        self.video_menu.addAction("Load Right Eye Video…", lambda: self._browse_video("right"))

        self.video_menu.addSeparator()

        self.video_menu.addAction("Clear Arena Video", lambda: self._clear_video("arena"))
        self.video_menu.addAction("Clear Left Eye Video", lambda: self._clear_video("left"))
        self.video_menu.addAction("Clear Right Eye Video", lambda: self._clear_video("right"))

        self.video_menu.addSeparator()

        self.act_show_video = self.video_menu.addAction("Show embedded videos")
        self.act_show_video.setCheckable(True)
        self.act_show_video.setChecked(True)
        self.act_show_video.toggled.connect(self._toggle_video_visibility)

        self.act_show_video_controls = self.video_menu.addAction("Show video controls")
        self.act_show_video_controls.setCheckable(True)
        self.act_show_video_controls.setChecked(False)

        self.video_btn = QtWidgets.QToolButton()
        self.video_btn.setText("Video ▾")
        self.video_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.video_btn.setMenu(self.video_menu)

        # Pack controls line
        for w in (self.btn_back, self.btn_play, self.btn_fwd,
                  self.step_jump_lbl, self.step_jump,
                  self.speed_lbl, self.speed,
                  self.fps_lbl, self.fps,
                  self.slider,
                  self.data_btn, self.scene_btn, self.view_btn, self.play_btn, self.export_btn, self.video_btn):
            controls.addWidget(w)
        controls.addWidget(self.objects_btn)

        # ===== View Controls (sliders) =====
        self.view_controls_group = self._build_view_controls_ui()
        # Initialize BNO-origin UI and apply identity
        self._sync_bno_ui_from_state()
        self._apply_bno_origin_transform()
        self.side_vbox.addWidget(self.view_controls_group)
        self.view_controls_group.setVisible(False)

        # ===== External OBJ Transform panel (hidden until toggled) =====
        self.obj_transform_group = self._build_obj_transform_ui()
        self.side_vbox.addWidget(self.obj_transform_group)
        self.obj_transform_group.setVisible(False)

        # ===== Eye vectors panel (hidden until toggled) =====
        self.eye_vectors_group = self._build_eye_vectors_ui()
        self.side_vbox.addWidget(self.eye_vectors_group)
        self.eye_vectors_group.setVisible(False)

        # ===== Video controls panel (hidden until toggled) =====
        self.video_controls_group = self._build_video_controls_ui()
        self.side_vbox.addWidget(self.video_controls_group)
        self.video_controls_group.setVisible(False)
        self.act_show_video_controls.toggled.connect(self.video_controls_group.setVisible)
        self.act_show_video_controls.toggled.connect(lambda _v: self._update_side_panel_visibility())

        self.act_show_transform.toggled.connect(self.obj_transform_group.setVisible)
        self.act_show_transform.toggled.connect(self.eye_vectors_group.setVisible)
        self.act_show_transform.toggled.connect(lambda _v: self._update_side_panel_visibility())

        self.side_vbox.addStretch(1)

        self._update_side_panel_visibility()
        self._disable_wheel_on_controls()

        # Frame the scene & apply first frame
        self._reset_view();
        self._apply_frame(0)
        self._update_axis_gizmo()

        # Timer & signals
        self.timer = QtCore.QTimer(self);
        self.timer.timeout.connect(self._on_timer)
        self.fps.valueChanged.connect(self._update_timer_interval)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_back.clicked.connect(self._step_back)
        self.btn_fwd.clicked.connect(self._step_fwd)
        self.slider.valueChanged.connect(self._on_slider)
        # --- playback timing + FPS measurement ---
        self.play_start_wall = None  # wall-clock when play started
        self.play_start_idx = 0  # frame index at play start
        self.actual_fps = 0.0  # measured render FPS
        self._fps_samples = 0
        self._fps_last_t = time.perf_counter()

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self._toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence('1'), self, activated=partial(self._snap_view, 90, 90))
        QtWidgets.QShortcut(QtGui.QKeySequence('2'), self, activated=partial(self._snap_view, 0, 90))
        QtWidgets.QShortcut(QtGui.QKeySequence('3'), self, activated=partial(self._snap_view, 0, 180))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=self._step_fwd)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, activated=self._step_back)
        QtWidgets.QShortcut(QtGui.QKeySequence('['), self, activated=self._mark_in)
        QtWidgets.QShortcut(QtGui.QKeySequence(']'), self, activated=self._mark_out)
        QtWidgets.QShortcut(QtGui.QKeySequence('L'), self,
                            activated=lambda: self.act_loop.setChecked(not self.act_loop.isChecked()))

        # DPI changes → update tick sizes
        self.canvas.events.resize.connect(self._on_canvas_resize)

        self.view_locked = False
        self._exporting = False
        self._cancel_flag = False
        self._progress = None

        self.resize(1200, 1000);
        self.show()

    def _detect_default_columns(self, df):
        if df.empty: return {"Time": "", "Roll": "", "Pitch": "", "Yaw": ""}
        try:
            return {
                "Time": pick_col(df, TIME_COLS), "Roll": pick_col(df, ROLL_COLS),
                "Pitch": pick_col(df, PITCH_COLS), "Yaw": pick_col(df, YAW_COLS)
            }
        except KeyError:
            c = list(df.columns)
            return {"Time": c[0] if len(c) > 0 else "", "Roll": c[1] if len(c) > 1 else "",
                    "Pitch": c[2] if len(c) > 2 else "", "Yaw": c[3] if len(c) > 3 else ""}

    def _get_frame_ids(self, df: pd.DataFrame):
        """
        Return int frame-ids for each row in df.
        Priority:
          1) any known frame column (FRAME_COLS)
          2) if there's an unnamed first column, use it
          3) fallback: row index [0..n-1]
        """
        if df is None or df.empty:
            return np.array([], dtype=np.int64)

        # Try explicit candidates
        for name in FRAME_COLS:
            if name in df.columns:
                try:
                    return df[name].to_numpy(dtype=np.int64, copy=True)
                except Exception:
                    pass

        # Try first column if it looks like "Unnamed: 0" or empty header
        first = df.columns[0]
        if str(first).strip() in ("", "Unnamed: 0"):
            try:
                return df[first].to_numpy(dtype=np.int64, copy=True)
            except Exception:
                pass

        # Fallback
        return np.arange(len(df), dtype=np.int64)

    def _recalc_trajectory(self):
        if self.df.empty:
            self.t = np.array([], dtype=float)
            self.roll = np.array([], dtype=float)
            self.pitch = np.array([], dtype=float)
            self.yaw = np.array([], dtype=float)
            self.t_abs_data = np.array([], dtype=float)
            self.gui_time = np.array([], dtype=float)
            self.arena_frame_data = np.array([], dtype=float)
            self.arena_video_frame_data = np.array([], dtype=float)
            self.left_video_frame_data = np.array([], dtype=float)
            self.right_video_frame_data = np.array([], dtype=float)
            self.eye_pf_series = [None, None]
            self.eye_pf_first_row = [None, None]
            self.first_head_ms = None
            self.n = 0
            return

        try:
            time_col = self.col_map["Time"]
            roll_col = self.col_map["Roll"]
            pitch_col = self.col_map["Pitch"]
            yaw_col = self.col_map["Yaw"]

            t_all = self.df[time_col].to_numpy(float)
            roll_all = self.df[roll_col].to_numpy(float)
            pitch_all = self.df[pitch_col].to_numpy(float)
            yaw_all = self.df[yaw_col].to_numpy(float)
        except Exception:
            self.n = 0
            return

        # Keep only rows that actually have valid head data.
        head_mask = (
                np.isfinite(t_all) &
                np.isfinite(roll_all) &
                np.isfinite(pitch_all) &
                np.isfinite(yaw_all)
        )

        if not np.any(head_mask):
            self.t = np.array([], dtype=float)
            self.roll = np.array([], dtype=float)
            self.pitch = np.array([], dtype=float)
            self.yaw = np.array([], dtype=float)
            self.t_abs_data = np.array([], dtype=float)
            self.gui_time = np.array([], dtype=float)
            self.arena_frame_data = np.array([], dtype=float)
            self.arena_video_frame_data = np.array([], dtype=float)
            self.left_video_frame_data = np.array([], dtype=float)
            self.right_video_frame_data = np.array([], dtype=float)
            self.eye_pf_series = [None, None]
            self.eye_pf_first_row = [None, None]
            self.first_head_ms = None
            self.n = 0
            return

        df_data = self.df.loc[head_mask].reset_index(drop=True)

        self.t_abs_data = df_data[time_col].to_numpy(float)
        self.first_head_ms = float(self.t_abs_data[0])

        self.roll = df_data[roll_col].to_numpy(float)
        self.pitch = df_data[pitch_col].to_numpy(float)
        self.yaw = df_data[yaw_col].to_numpy(float)

        # Keep a relative time too, useful for video sync deltas if needed.
        self.t = self.t_abs_data - float(self.t_abs_data[0])

        # Actual Arena_frame values for display only. They may repeat.
        try:
            frame_col = pick_col(df_data, FRAME_COLS)
            self.arena_frame_data = df_data[frame_col].to_numpy(float)
        except Exception:
            self.arena_frame_data = np.arange(len(df_data), dtype=float)

        # Exact video-frame indices to show for each data row.
        try:
            arena_vcol = pick_col(df_data, ARENA_VIDEO_FRAME_COLS)
            self.arena_video_frame_data = df_data[arena_vcol].to_numpy(float)
        except Exception:
            self.arena_video_frame_data = np.array([np.nan] * len(df_data), dtype=float)

        try:
            left_vcol = pick_col(df_data, LEFT_VIDEO_FRAME_COLS)
            self.left_video_frame_data = df_data[left_vcol].to_numpy(float)
        except Exception:
            self.left_video_frame_data = np.array([np.nan] * len(df_data), dtype=float)

        try:
            right_vcol = pick_col(df_data, RIGHT_VIDEO_FRAME_COLS)
            self.right_video_frame_data = df_data[right_vcol].to_numpy(float)
        except Exception:
            self.right_video_frame_data = np.array([np.nan] * len(df_data), dtype=float)

        # Build per-eye row-aligned series from the SAME loaded data file
        def _make_eye_series(yaw_candidates, pitch_candidates):
            try:
                ycol = pick_col(df_data, yaw_candidates)
                pcol = pick_col(df_data, pitch_candidates)
            except KeyError:
                return None, None

            yv = df_data[ycol].to_numpy(float)
            pv = df_data[pcol].to_numpy(float)
            arr = np.column_stack([yv, pv]).astype(float)

            valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
            first_row = int(np.where(valid)[0][0]) if np.any(valid) else None
            return arr, first_row

        left_arr, left_first = _make_eye_series(EYE_YAW_LEFT_COLS, EYE_PITCH_LEFT_COLS)
        right_arr, right_first = _make_eye_series(EYE_YAW_RIGHT_COLS, EYE_PITCH_RIGHT_COLS)

        self.eye_pf_series = [left_arr, right_arr]
        self.eye_pf_first_row = [left_first, right_first]
        self.eye_pf_last_dirH = [None, None]
        self.eye_pf_prev_i = None

        self.n = len(self.t_abs_data)

        self.total_duration = float(self.t_abs_data[-1]) if self.n else 0.0
        self.total_duration_str = self._fmt_clock(self.total_duration)

        self.transforms = np.repeat(np.eye(4, dtype=np.float32)[None, ...], self.n, axis=0)
        for i in range(self.n):
            self.transforms[i, 0:3, 0:3] = rotation_from_ypr(
                self.roll[i], self.pitch[i], self.yaw[i]
            ).astype(np.float32)

        self.data_index_offset = 0
        self.gui_time = self.t_abs_data.copy()

        if getattr(self, "prepend_zero_pose_frame", False) and self.n > 0:
            I4 = np.eye(4, dtype=np.float32)[None, ...]
            self.transforms = np.concatenate([I4, self.transforms], axis=0)

            # GUI shows virtual frame at 0.0, then jumps to first real head timestamp.
            self.gui_time = np.concatenate([[0.0], self.t_abs_data], axis=0)

            self.n = int(self.transforms.shape[0])
            self.data_index_offset = 1

        if hasattr(self, 'slider'):
            self.slider.setRange(0, max(0, self.n - 1))

        self.mark_in, self.mark_out = 0, max(0, self.n - 1)
        self.hud_marks = ""

    def _load_new_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Data File", "",
                                                         "Data Files (*.csv *.xlsx);;All Files (*)")
        if not fname:
            return

        try:
            new_df = read_table(fname)
            if new_df.empty:
                raise ValueError("File is empty")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Could not load file:\n{e}")
            return

        self.df = new_df
        self.col_map = self._detect_default_columns(self.df)
        self.eye_pf_source_path = fname
        self._recalc_trajectory()

        if hasattr(self, "lbl_eye_pf_file"):
            self.lbl_eye_pf_file.setText(f"Eye data source: {os.path.basename(fname)}")

        if getattr(self, "eye_pf_enabled", False):
            self._capture_eye_anchors_world_to_model_local()

        self._apply_frame(0)
        self.statusBar().showMessage(f"Loaded: {os.path.basename(fname)}", 5000)

    def _prompt_columns(self):
        if self.df.empty: return
        dlg = ColumnSelectionDialog(self, list(self.df.columns), self.col_map)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.col_map = dlg.get_selection()
            self._recalc_trajectory()
            self._apply_frame(0)

    def _export_csv_with_eye_dirs_world(self):
        """
        Export the loaded dataframe with two new columns:
        one per eye vector, containing "(PitchDeg, YawDeg)" per row,
        where Pitch/Yaw are relative to WORLD axes.

        Important:
        - output rows match the ORIGINAL loaded dataframe exactly
        - rows without valid head data export as blank strings
        - per-row eye mode uses the SAME loaded file, row by row
        """
        if getattr(self, "df", None) is None or self.df.empty:
            QtWidgets.QMessageBox.warning(self, "No data", "No dataframe loaded.")
            return

        pf = bool(getattr(self, "eye_pf_enabled", False))

        if pf:
            if self.eye_pf_series[0] is None and self.eye_pf_series[1] is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No eye directions",
                    "No row-based eye direction data was found in the loaded file."
                )
                return
            if any(a is None for a in getattr(self, "eye_pf_anchor_local", [None, None])):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing anchors",
                    "Enable per-row eye mode once to capture anchors."
                )
                return
        else:
            if not getattr(self, "eye_follow_model", False):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Eye vectors not locked",
                    "Lock the eye vectors to the model first (Follow model / capture) before exporting."
                )
                return

            if getattr(self, "eye_local_segments", None) is None or any(s is None for s in self.eye_local_segments):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing capture",
                    "Eye vectors were not captured. Toggle Follow model (capture) once, then export."
                )
                return

        default_name = "with_eye_dirs_world.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV with Eye Directions (World)", default_name, "CSV Files (*.csv)"
        )
        if not path:
            return

        # Preserve current GUI state
        cur_i = int(getattr(self, "i", 0))
        cur_root = self.root.transform.matrix.copy()
        cur_neutral = bool(getattr(self, "neutral_pose", False))

        try:
            df_out = self.df.copy()
            out_left_eye = [""] * len(df_out)
            out_right_eye = [""] * len(df_out)

            # Rebuild the same head-valid mask used by _recalc_trajectory()
            time_col = self.col_map["Time"]
            roll_col = self.col_map["Roll"]
            pitch_col = self.col_map["Pitch"]
            yaw_col = self.col_map["Yaw"]

            t_all = self.df[time_col].to_numpy(float)
            roll_all = self.df[roll_col].to_numpy(float)
            pitch_all = self.df[pitch_col].to_numpy(float)
            yaw_all = self.df[yaw_col].to_numpy(float)

            head_mask = (
                    np.isfinite(t_all) &
                    np.isfinite(roll_all) &
                    np.isfinite(pitch_all) &
                    np.isfinite(yaw_all)
            )

            valid_row_indices = np.flatnonzero(head_mask)

            # valid_data_idx runs over filtered head-data rows
            for valid_data_idx, original_row_idx in enumerate(valid_row_indices):
                k_vis = int(valid_data_idx + getattr(self, "data_index_offset", 0))

                if k_vis < 0 or k_vis >= len(self.transforms):
                    continue

                # Apply frame rotation
                if cur_neutral:
                    self.root.transform.matrix = np.eye(4, dtype=np.float32)
                else:
                    self.root.transform.matrix = self.transforms[k_vis]

                T = self._model_T_world()

                if pf:
                    R = self._linear3_from_transform(T)

                    # Export by anatomical source after applying the current swap state.
                    left_src_idx = 0 if getattr(self, "eye_pf_swap_lr", False) else 1
                    right_src_idx = 1 if getattr(self, "eye_pf_swap_lr", False) else 0

                    for src_idx, out_list in [(left_src_idx, out_left_eye), (right_src_idx, out_right_eye)]:
                        arr = self.eye_pf_series[src_idx]

                        if arr is None or valid_data_idx >= len(arr):
                            out_list[original_row_idx] = ""
                            continue

                        yaw_deg_H = float(arr[valid_data_idx, 0])
                        pitch_deg_H = float(arr[valid_data_idx, 1])

                        if not (np.isfinite(yaw_deg_H) and np.isfinite(pitch_deg_H)):
                            out_list[original_row_idx] = ""
                            continue

                        d_H = self._eye_dir_from_yaw_pitch(yaw_deg_H, pitch_deg_H)
                        d_W = (R @ np.asarray(d_H, dtype=np.float32).reshape(3))

                        yawW, pitchW = self._yaw_pitch_from_world_dir(d_W)
                        out_list[original_row_idx] = f"({pitchW:.3f}, {yawW:.3f})"

                else:
                    for idx, out_list in [(0, out_left_eye), (1, out_right_eye)]:
                        origin_l, end_l = self.eye_local_segments[idx]
                        origin_w = self._as_vec3(T.map(origin_l))
                        end_w = self._as_vec3(T.map(end_l))
                        d_w = end_w - origin_w
                        yaw_deg, pitch_deg = self._yaw_pitch_from_world_dir(d_w)
                        out_list[original_row_idx] = f"({pitch_deg:.3f}, {yaw_deg:.3f})"

            df_out["EyeL_World_PitchYaw"] = out_left_eye
            df_out["EyeR_World_PitchYaw"] = out_right_eye
            df_out.to_csv(path, index=False)

            QtWidgets.QMessageBox.information(
                self,
                "Export complete",
                "Saved CSV with EyeL_World_PitchYaw and EyeR_World_PitchYaw."
            )

        finally:
            self.root.transform.matrix = cur_root
            self.i = cur_i
            try:
                self._apply_frame(self.i)
            except Exception:
                self.canvas.update()

    # ----- Video imbedding panel -----
    def _build_video_controls_ui(self):
        gb = QtWidgets.QGroupBox("Video Controls")
        outer = QtWidgets.QVBoxLayout(gb)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.video_control_widgets = {}

        for key in ("arena", "left", "right"):
            if key == "arena":
                group_title = "Arena Video"
            else:
                group_title = self.video_slots[key]["title"]

            row_box = QtWidgets.QGroupBox(group_title)
            grid = QtWidgets.QGridLayout(row_box)

            btn_load = QtWidgets.QPushButton("Load")
            btn_clear = QtWidgets.QPushButton("Clear")
            btn_cw = QtWidgets.QPushButton("Rotate CW")
            btn_ccw = QtWidgets.QPushButton("Rotate CCW")
            btn_flip_h = QtWidgets.QPushButton("Flip H")
            btn_flip_v = QtWidgets.QPushButton("Flip V")

            grid.addWidget(btn_load, 0, 0)
            grid.addWidget(btn_clear, 0, 1)
            grid.addWidget(btn_cw, 1, 0)
            grid.addWidget(btn_ccw, 1, 1)
            grid.addWidget(btn_flip_h, 2, 0)
            grid.addWidget(btn_flip_v, 2, 1)

            btn_load.clicked.connect(lambda _=False, k=key: self._browse_video(k))
            btn_clear.clicked.connect(lambda _=False, k=key: self._clear_video(k))
            btn_cw.clicked.connect(lambda _=False, k=key: self._rotate_video_cw(k))
            btn_ccw.clicked.connect(lambda _=False, k=key: self._rotate_video_ccw(k))
            btn_flip_h.clicked.connect(lambda _=False, k=key: self._flip_video_h(k))
            btn_flip_v.clicked.connect(lambda _=False, k=key: self._flip_video_v(k))

            outer.addWidget(row_box)

        outer.addStretch(1)
        return gb

    def _make_empty_video_slot(self, title):
        return {
            "title": title,
            "path": None,
            "reader": None,
            "meta": {},
            "nframes": 0,
            "cache": {},
            "rotation_quadrants": 0,
            "flip_h": False,
            "flip_v": False,
        }

    def _get_active_arena_slot(self):
        if not self.arena_video_variants:
            return None
        idx = max(0, min(self.active_arena_video_idx, len(self.arena_video_variants) - 1))
        return self.arena_video_variants[idx]

    def _get_video_slot(self, slot_key):
        if slot_key == "arena":
            return self._get_active_arena_slot()
        return self.video_slots[slot_key]

    def _update_arena_title(self):
        widgets = self.video_widgets.get("arena")
        if not widgets:
            return

        total = len(self.arena_video_variants)
        if total <= 0:
            widgets["title"].setText("Arena Video")
            if widgets.get("prev_btn") is not None:
                widgets["prev_btn"].setEnabled(False)
            if widgets.get("next_btn") is not None:
                widgets["next_btn"].setEnabled(False)
            return

        idx = max(0, min(self.active_arena_video_idx, total - 1))
        widgets["title"].setText(f"Arena Video ({idx + 1}/{total})")

        enable_switch = total > 1
        if widgets.get("prev_btn") is not None:
            widgets["prev_btn"].setEnabled(enable_switch)
        if widgets.get("next_btn") is not None:
            widgets["next_btn"].setEnabled(enable_switch)

    def _prev_arena_video(self):
        total = len(self.arena_video_variants)
        if total <= 1:
            return
        self.active_arena_video_idx = (self.active_arena_video_idx - 1) % total
        self._update_arena_title()
        self._update_video_frame("arena")

    def _next_arena_video(self):
        total = len(self.arena_video_variants)
        if total <= 1:
            return
        self.active_arena_video_idx = (self.active_arena_video_idx + 1) % total
        self._update_arena_title()
        self._update_video_frame("arena")

    def _browse_video(self, slot_key):
        if slot_key == "arena":
            self._browse_arena_videos()
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Load {self.video_slots[slot_key]['title']}",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if not path:
            return
        self._load_video(slot_key, path)

    def _browse_arena_videos(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Arena Videos (up to 4)",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if not paths:
            return

        paths = paths[:self.max_arena_videos]
        self._clear_video("arena")

        self.arena_video_variants = []
        for k, path in enumerate(paths, start=1):
            slot = self._make_empty_video_slot(f"Arena Video {k}")
            self._load_video_into_slot(slot, path)
            self.arena_video_variants.append(slot)

        self.active_arena_video_idx = 0
        self._update_arena_title()
        self._update_video_frame("arena")
        self.statusBar().showMessage(f"Loaded {len(self.arena_video_variants)} arena video(s)", 5000)

    def _load_video(self, slot_key, path):
        slot = self._get_video_slot(slot_key)
        widgets = self.video_widgets[slot_key]

        try:
            if slot is None:
                raise RuntimeError(f"No slot available for '{slot_key}'.")

            self._load_video_into_slot(slot, path)

            widgets["info"].setText(
                f"Frames: {slot['nframes'] if slot['nframes'] > 0 else 'unknown'}"
            )

            if slot_key == "arena":
                self._update_arena_title()
            else:
                widgets["title"].setText(slot["title"])

            self._update_video_frame(slot_key)
            self.statusBar().showMessage(f"Loaded {slot['title']}: {os.path.basename(path)}", 5000)

        except Exception as e:
            if slot is not None:
                slot["reader"] = None
                slot["path"] = None
                slot["meta"] = {}
                slot["nframes"] = 0
                slot["cache"] = {}

            widgets["label"].setText("Failed to load video")
            QtWidgets.QMessageBox.critical(self, "Video Load Error", str(e))

    def _load_video_into_slot(self, slot, path):
        if not HAVE_IMAGEIO:
            raise RuntimeError("imageio / imageio-ffmpeg is not available.")

        if slot["reader"] is not None:
            try:
                slot["reader"].close()
            except Exception:
                pass

        slot["reader"] = imageio.get_reader(path)
        slot["path"] = path
        slot["meta"] = slot["reader"].get_meta_data() or {}
        slot["cache"] = {}

        nframes = slot["meta"].get("nframes", None)
        if nframes is None or nframes == float("inf"):
            try:
                nframes = slot["reader"].count_frames()
            except Exception:
                nframes = 0

        slot["nframes"] = int(nframes) if nframes is not None else 0

    def _clear_video(self, slot_key):
        if slot_key == "arena":
            for slot in self.arena_video_variants:
                if slot["reader"] is not None:
                    try:
                        slot["reader"].close()
                    except Exception:
                        pass
            self.arena_video_variants = []
            self.active_arena_video_idx = 0

            widgets = self.video_widgets["arena"]
            widgets["label"].clear()
            widgets["label"].setText("No video loaded")
            widgets["info"].setText("")
            self._update_arena_title()
            return

        slot = self._get_video_slot(slot_key)
        widgets = self.video_widgets[slot_key]

        if slot["reader"] is not None:
            try:
                slot["reader"].close()
            except Exception:
                pass

        slot["path"] = None
        slot["reader"] = None
        slot["meta"] = {}
        slot["nframes"] = 0
        slot["cache"] = {}
        slot["rotation_quadrants"] = 0
        slot["flip_h"] = False
        slot["flip_v"] = False

        widgets["label"].clear()
        widgets["label"].setText("No video loaded")
        widgets["info"].setText("")
        widgets["title"].setText(slot["title"])

    def _toggle_video_visibility(self, visible):
        self.video_panel_visible = bool(visible)
        self.video_panel.setVisible(bool(visible))

    def _video_frame_index_from_gui_index(self, slot_key, idx):
        idx = int(idx)

        # Virtual frame has no corresponding video frame.
        if idx < int(getattr(self, "data_index_offset", 0)):
            return None

        data_idx = idx - int(getattr(self, "data_index_offset", 0))
        if data_idx < 0:
            return None

        if slot_key == "arena":
            arr = getattr(self, "arena_video_frame_data", None)
        elif slot_key == "left":
            arr = getattr(self, "left_video_frame_data", None)
        elif slot_key == "right":
            arr = getattr(self, "right_video_frame_data", None)
        else:
            return None

        if arr is None or data_idx >= len(arr):
            return None

        v = arr[data_idx]
        if not np.isfinite(v):
            return None

        return int(round(float(v)))

    def _read_video_frame(self, slot_key, frame_idx):
        slot = self._get_video_slot(slot_key)
        if slot is None:
            return None

        if slot["reader"] is None or frame_idx is None:
            return None

        frame_idx = int(frame_idx)
        if frame_idx < 0:
            return None

        if slot["nframes"] > 0 and frame_idx >= slot["nframes"]:
            return None

        if frame_idx in slot["cache"]:
            return slot["cache"][frame_idx]

        frame = slot["reader"].get_data(frame_idx)

        slot["cache"][frame_idx] = frame
        if len(slot["cache"]) > self.video_cache_limit:
            oldest_key = next(iter(slot["cache"]))
            del slot["cache"][oldest_key]

        return frame

    def _transform_video_array(self, slot_key, arr):
        slot = self._get_video_slot(slot_key)
        if slot is None:
            return np.ascontiguousarray(arr)
        out = arr

        q = int(slot["rotation_quadrants"]) % 4
        if q:
            out = np.rot90(out, q)

        if slot["flip_h"]:
            out = np.fliplr(out)

        if slot["flip_v"]:
            out = np.flipud(out)

        return np.ascontiguousarray(out)

    def _numpy_rgb_to_qpixmap(self, arr):
        if arr is None:
            return QtGui.QPixmap()

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.shape[2] == 4:
            fmt = QtGui.QImage.Format_RGBA8888
        else:
            fmt = QtGui.QImage.Format_RGB888

        h, w, c = arr.shape
        arr = np.ascontiguousarray(arr)
        qimg = QtGui.QImage(arr.data, w, h, arr.strides[0], fmt)
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _update_video_frame(self, slot_key):
        slot = self._get_video_slot(slot_key)
        widgets = self.video_widgets[slot_key]

        if slot is None:
            widgets["label"].clear()
            widgets["label"].setText("No video loaded")
            widgets["info"].setText("")
            return

        if slot["reader"] is None:
            return

        vf = self._video_frame_index_from_gui_index(slot_key, self.i)

        if vf is None:
            widgets["label"].clear()
            widgets["label"].setText("No frame for this timestamp")
            text = f"GUI: {self.i} | Video: none"

            if widgets["info"].text() != text:
                widgets["info"].setText(text)
            return

        try:
            frame = self._read_video_frame(slot_key, vf)
            if frame is None:
                widgets["label"].clear()
                widgets["label"].setText("No matching video frame")
                text = f"GUI: {self.i} | Video: {vf}"
                if widgets["info"].text() != text:
                    widgets["info"].setText(text)
                return

            frame = self._transform_video_array(slot_key, frame)

            pix = self._numpy_rgb_to_qpixmap(frame)
            if pix.isNull():
                widgets["label"].clear()
                widgets["label"].setText("Failed to decode video frame")
                return

            if self.video_scale_to_fit:
                pix = pix.scaled(
                    widgets["label"].size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )

            widgets["label"].setPixmap(pix)
            text = f"GUI: {self.i} | Video: {vf}"
            if widgets["info"].text() != text:
                widgets["info"].setText(text)

        except Exception as e:
            widgets["label"].clear()
            widgets["label"].setText(f"Video frame error:\n{e}")

    def _update_all_videos(self):
        for key in ("arena", "left", "right"):
            self._update_video_frame(key)

    def _rotate_video_cw(self, slot_key):
        slot = self._get_video_slot(slot_key)
        if slot is None:
            return
        slot["rotation_quadrants"] = (slot["rotation_quadrants"] - 1) % 4
        self._update_video_frame(slot_key)

    def _rotate_video_ccw(self, slot_key):
        slot = self._get_video_slot(slot_key)
        if slot is None:
            return
        slot["rotation_quadrants"] = (slot["rotation_quadrants"] + 1) % 4
        self._update_video_frame(slot_key)

    def _flip_video_h(self, slot_key):
        slot = self._get_video_slot(slot_key)
        if slot is None:
            return
        slot["flip_h"] = not slot["flip_h"]
        self._update_video_frame(slot_key)

    def _flip_video_v(self, slot_key):
        slot = self._get_video_slot(slot_key)
        if slot is None:
            return
        slot["flip_v"] = not slot["flip_v"]
        self._update_video_frame(slot_key)

    # ----- View Controls panel -----
    def _build_view_controls_ui(self):
        gb = QtWidgets.QGroupBox("View / Scene Controls")
        grid = QtWidgets.QGridLayout(gb);
        grid.setContentsMargins(10, 8, 10, 8);
        grid.setHorizontalSpacing(10)

        def make_slider(min_i, max_i, tick=10):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(min_i, max_i);
            s.setTickInterval(tick);
            s.setSingleStep(1)
            return s

        def bind_pair(slider, spin, to_spin, to_slider, on_change):
            def _s2p(val):
                spin.blockSignals(True);
                spin.setValue(to_spin(val));
                spin.blockSignals(False);
                on_change()

            slider.valueChanged.connect(_s2p)

            def _p2s(val):
                slider.blockSignals(True);
                slider.setValue(to_slider(val));
                slider.blockSignals(False);
                on_change()

            spin.valueChanged.connect(_p2s)

        # Cube scale (0.10 .. 10.00)
        self.cube_scale_slider = make_slider(10, 1000, tick=25)
        self.cube_scale_spin = QtWidgets.QDoubleSpinBox()
        self.cube_scale_spin.setRange(0.10, 10.00);
        self.cube_scale_spin.setDecimals(2);
        self.cube_scale_spin.setSingleStep(0.01)
        bind_pair(self.cube_scale_slider, self.cube_scale_spin,
                  to_spin=lambda v: v / 100.0, to_slider=lambda v: int(round(v * 100)),
                  on_change=self._apply_axes_scale)

        # Model scale (0.10 .. 10.00)
        self.model_scale_slider = make_slider(10, 1000, tick=25)
        self.model_scale_spin = QtWidgets.QDoubleSpinBox()
        self.model_scale_spin.setRange(0.10, 10.00);
        self.model_scale_spin.setDecimals(2);
        self.model_scale_spin.setSingleStep(0.01)
        bind_pair(self.model_scale_slider, self.model_scale_spin,
                  to_spin=lambda v: v / 100.0, to_slider=lambda v: int(round(v * 100)),
                  on_change=self._apply_model_scale)

        r = 0
        grid.addWidget(QtWidgets.QLabel("Cube scale"), r, 0);
        grid.addWidget(self.cube_scale_slider, r, 1);
        grid.addWidget(self.cube_scale_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Model scale"), r, 0);
        grid.addWidget(self.model_scale_slider, r, 1);
        grid.addWidget(self.model_scale_spin, r, 2);
        r += 1

        # Defaults
        self.cube_scale_spin.setValue(self.axes_scale)
        self.model_scale_spin.setValue(self.model_scale)

        # ----- Camera controls -----
        grid.addWidget(QtWidgets.QLabel("— Camera —"), r, 0);
        r += 1

        # Elevation (-180..+180) & Azimuth (-180..+180)
        self.cam_elev_slider = make_slider(-180, 180, tick=10)
        self.cam_elev_spin = QtWidgets.QDoubleSpinBox()
        self.cam_elev_spin.setRange(-180.0, 180.0);
        self.cam_elev_spin.setDecimals(1);
        self.cam_elev_spin.setSingleStep(0.5)

        self.cam_azim_slider = make_slider(-180, 180, tick=15)
        self.cam_azim_spin = QtWidgets.QDoubleSpinBox()
        self.cam_azim_spin.setRange(-180.0, 180.0);
        self.cam_azim_spin.setDecimals(1);
        self.cam_azim_spin.setSingleStep(0.5)

        bind_pair(self.cam_elev_slider, self.cam_elev_spin,
                  to_spin=float, to_slider=int, on_change=self._apply_camera_from_ui)
        bind_pair(self.cam_azim_slider, self.cam_azim_spin,
                  to_spin=float, to_slider=int, on_change=self._apply_camera_from_ui)

        grid.addWidget(QtWidgets.QLabel("Elevation (°)"), r, 0);
        grid.addWidget(self.cam_elev_slider, r, 1);
        grid.addWidget(self.cam_elev_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Azimuth (°)"), r, 0);
        grid.addWidget(self.cam_azim_slider, r, 1);
        grid.addWidget(self.cam_azim_spin, r, 2);
        r += 1

        # Distance (zoom)  (1 .. 100)
        self.cam_dist_slider = make_slider(1, 100, tick=5)
        self.cam_dist_spin = QtWidgets.QDoubleSpinBox()
        self.cam_dist_spin.setRange(1.0, 100.0);
        self.cam_dist_spin.setDecimals(2);
        self.cam_dist_spin.setSingleStep(0.25)

        bind_pair(self.cam_dist_slider, self.cam_dist_spin,
                  to_spin=lambda v: float(v), to_slider=lambda v: int(round(v)),
                  on_change=self._apply_camera_from_ui)

        grid.addWidget(QtWidgets.QLabel("Distance"), r, 0);
        grid.addWidget(self.cam_dist_slider, r, 1);
        grid.addWidget(self.cam_dist_spin, r, 2);
        r += 1

        # Center X/Y/Z (-10 .. +10)
        self.cam_cx_slider = make_slider(-1000, 1000, tick=50)
        self.cam_cy_slider = make_slider(-1000, 1000, tick=50)
        self.cam_cz_slider = make_slider(-1000, 1000, tick=50)

        self.cam_cx_spin = QtWidgets.QDoubleSpinBox();
        self.cam_cx_spin.setRange(-10.0, 10.0);
        self.cam_cx_spin.setDecimals(2);
        self.cam_cx_spin.setSingleStep(0.05)
        self.cam_cy_spin = QtWidgets.QDoubleSpinBox();
        self.cam_cy_spin.setRange(-10.0, 10.0);
        self.cam_cy_spin.setDecimals(2);
        self.cam_cy_spin.setSingleStep(0.05)
        self.cam_cz_spin = QtWidgets.QDoubleSpinBox();
        self.cam_cz_spin.setRange(-10.0, 10.0);
        self.cam_cz_spin.setDecimals(2);
        self.cam_cz_spin.setSingleStep(0.05)

        bind_pair(self.cam_cx_slider, self.cam_cx_spin, to_spin=lambda v: v / 100.0,
                  to_slider=lambda v: int(round(v * 100)), on_change=self._apply_camera_from_ui)
        bind_pair(self.cam_cy_slider, self.cam_cy_spin, to_spin=lambda v: v / 100.0,
                  to_slider=lambda v: int(round(v * 100)), on_change=self._apply_camera_from_ui)
        bind_pair(self.cam_cz_slider, self.cam_cz_spin, to_spin=lambda v: v / 100.0,
                  to_slider=lambda v: int(round(v * 100)), on_change=self._apply_camera_from_ui)

        grid.addWidget(QtWidgets.QLabel("Center X"), r, 0);
        grid.addWidget(self.cam_cx_slider, r, 1);
        grid.addWidget(self.cam_cx_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Center Y"), r, 0);
        grid.addWidget(self.cam_cy_slider, r, 1);
        grid.addWidget(self.cam_cy_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Center Z"), r, 0);
        grid.addWidget(self.cam_cz_slider, r, 1);
        grid.addWidget(self.cam_cz_spin, r, 2);
        r += 1

        # Defaults from class constants
        self.cam_elev_spin.setValue(self.CAM_ELEV)
        self.cam_azim_spin.setValue(self.CAM_AZIM)
        self.cam_dist_spin.setValue(self.CAM_DIST)
        self.cam_cx_spin.setValue(self.CAM_CENTER[0])
        self.cam_cy_spin.setValue(self.CAM_CENTER[1])
        self.cam_cz_spin.setValue(self.CAM_CENTER[2])

        # ----- BNO Origin (pose + position) -----
        bno_row = r
        grid.addWidget(QtWidgets.QLabel("— BNO Origin —"), bno_row, 0)
        r += 1

        self.chk_bno_enable = QtWidgets.QCheckBox("Enable BNO origin")
        self.chk_bno_enable.setChecked(True)
        self.chk_bno_enable.toggled.connect(
            lambda v: (setattr(self, "bno_origin_enabled", bool(v)), self._apply_bno_origin_transform()))
        grid.addWidget(self.chk_bno_enable, r, 0, 1, 3)
        r += 1

        # Position X/Y/Z sliders (-5.00 .. +5.00)
        self.bno_tx_slider = make_slider(-500, 500, tick=50)
        self.bno_ty_slider = make_slider(-500, 500, tick=50)
        self.bno_tz_slider = make_slider(-500, 500, tick=50)

        self.bno_tx = QtWidgets.QDoubleSpinBox()
        self.bno_tx.setRange(-5.00, 5.00)
        self.bno_tx.setDecimals(2)
        self.bno_tx.setSingleStep(0.01)
        self.bno_ty = QtWidgets.QDoubleSpinBox()
        self.bno_ty.setRange(-5.00, 5.00)
        self.bno_ty.setDecimals(2)
        self.bno_ty.setSingleStep(0.01)
        self.bno_tz = QtWidgets.QDoubleSpinBox()
        self.bno_tz.setRange(-5.00, 5.00)
        self.bno_tz.setDecimals(2)
        self.bno_tz.setSingleStep(0.01)

        bind_pair(self.bno_tx_slider, self.bno_tx, to_spin=lambda v: v / 100.0, to_slider=lambda v: int(round(v * 100)),
                  on_change=self._on_bno_origin_changed)
        bind_pair(self.bno_ty_slider, self.bno_ty, to_spin=lambda v: v / 100.0, to_slider=lambda v: int(round(v * 100)),
                  on_change=self._on_bno_origin_changed)
        bind_pair(self.bno_tz_slider, self.bno_tz, to_spin=lambda v: v / 100.0, to_slider=lambda v: int(round(v * 100)),
                  on_change=self._on_bno_origin_changed)

        grid.addWidget(QtWidgets.QLabel("Origin X"), r, 0)
        grid.addWidget(self.bno_tx_slider, r, 1)
        grid.addWidget(self.bno_tx, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Origin Y"), r, 0)
        grid.addWidget(self.bno_ty_slider, r, 1)
        grid.addWidget(self.bno_ty, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Origin Z"), r, 0)
        grid.addWidget(self.bno_tz_slider, r, 1)
        grid.addWidget(self.bno_tz, r, 2)
        r += 1

        # Roll/Pitch/Yaw sliders (-180 .. +180)
        self.bno_roll_slider = make_slider(-180, 180, tick=30)
        self.bno_pitch_slider = make_slider(-180, 180, tick=30)
        self.bno_yaw_slider = make_slider(-180, 180, tick=30)

        def make_deg():
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-180.0, 180.0)
            s.setDecimals(2)
            s.setSingleStep(0.1)
            return s

        self.bno_roll = make_deg()
        self.bno_pitch = make_deg()
        self.bno_yaw = make_deg()

        bind_pair(self.bno_roll_slider, self.bno_roll, to_spin=float, to_slider=int,
                  on_change=self._on_bno_origin_changed)
        bind_pair(self.bno_pitch_slider, self.bno_pitch, to_spin=float, to_slider=int,
                  on_change=self._on_bno_origin_changed)
        bind_pair(self.bno_yaw_slider, self.bno_yaw, to_spin=float, to_slider=int,
                  on_change=self._on_bno_origin_changed)

        grid.addWidget(QtWidgets.QLabel("Roll0 (°)"), r, 0)
        grid.addWidget(self.bno_roll_slider, r, 1)
        grid.addWidget(self.bno_roll, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Pitch0 (°)"), r, 0)
        grid.addWidget(self.bno_pitch_slider, r, 1)
        grid.addWidget(self.bno_pitch, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Yaw0 (°)"), r, 0)
        grid.addWidget(self.bno_yaw_slider, r, 1)
        grid.addWidget(self.bno_yaw, r, 2)
        r += 1

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.btn_bno_set_from_current = QtWidgets.QPushButton("Set from current frame")
        self.btn_bno_reset = QtWidgets.QPushButton("Reset BNO origin")
        self.btn_bno_set_from_current.clicked.connect(self._bno_set_from_current)
        self.btn_bno_reset.clicked.connect(self._bno_reset)
        btns.addWidget(self.btn_bno_set_from_current)
        btns.addWidget(self.btn_bno_reset)
        grid.addLayout(btns, r, 0, 1, 3)
        r += 1

        return gb

    def _update_side_panel_visibility(self):
        any_open = (
                not self.view_controls_group.isHidden() or
                not self.obj_transform_group.isHidden() or
                not self.eye_vectors_group.isHidden() or
                not self.video_controls_group.isHidden()
        )

        self.side_scroll.setVisible(any_open)

    def _apply_axes_scale(self):
        self.axes_scale = float(self.cube_scale_spin.value())
        self.axes_cube.set_extent(self.axes_base_extent * self.axes_scale)

    def _apply_bno_origin_transform(self):
        """Apply user-defined origin: translation and inverse of origin RPY."""
        if not getattr(self, "bno_origin_enabled", True):
            M = np.eye(4, dtype=np.float32)
        else:
            r0, p0, y0 = map(float, self.bno_origin_rpy)

            # Use the SAME convention as the CSV motion (ALIGN_WORLD/ALIGN_LOCAL included)
            R0 = rotation_from_ypr(r0, p0, y0).astype(np.float32)  # 3x3

            # We want to apply the inverse of the "origin" rotation.
            # Scene convention: rotations stored as R (no transpose).
            # If you ever want the inverse origin rotation, use R0.T explicitly.
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R0

            M[3, 0:3] = self.bno_origin_xyz.astype(np.float32)

        self.bno_origin_transform.matrix = M
        self.canvas.update()

    def _apply_model_scale(self):
        self.model_scale = float(self.model_scale_spin.value())
        M = np.eye(4, dtype=np.float32);
        M[:3, :3] = np.eye(3, dtype=np.float32) * self.model_scale
        self.model_transform.matrix = M
        self.canvas.update()

    # ----- Manual eye vectors (world-relative) -----
    def _eye_dir_from_yaw_pitch(self, yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """
        Unit direction in WORLD axes.
        """
        # Flip signs to match the rest of the app's perceived slider directions
        yaw = -yaw_deg
        pitch = -pitch_deg

        R = rotation_from_ypr(0.0, pitch, yaw).astype(np.float32)  # roll=0
        d = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)  # NO transpose
        n = float(np.linalg.norm(d))
        if n <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (d / n).astype(np.float32)

    def _yaw_pitch_from_world_dir(self, d_world: np.ndarray):
        """
        Inverse of _eye_dir_from_yaw_pitch for WORLD direction vectors.

        Returns (yaw_deg, pitch_deg) in the SAME slider convention used by the app
        (because _eye_dir_from_yaw_pitch internally flips signs).
        """
        d = np.asarray(d_world, dtype=np.float32).reshape(-1)
        if d.shape[0] == 4:
            d = d[:3]
        n = float(np.linalg.norm(d))
        if n <= 1e-12:
            return 0.0, 0.0
        d = d / n

        # In the forward mapping:
        # yaw  = -yaw_deg, pitch = -pitch_deg
        # d = [cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), sin(pitch)]
        yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
        pitch = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))

        yaw_deg = -yaw
        pitch_deg = pitch
        return float(yaw_deg), float(pitch_deg)

    def _update_eye_vector(self, idx: int):
        # If we're in "follow model" mode, visuals come from captured local endpoints per-frame.
        if getattr(self, "eye_follow_model", False):
            self._update_eyes_from_model_local()
            return

        p = self.eye_params[idx]
        origin = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
        d = self._eye_dir_from_yaw_pitch(float(p["yaw"]), float(p["pitch"]))
        end = origin + d * float(p["length"])
        pos = np.vstack([origin, end]).astype(np.float32)
        self.eye_lines[idx].set_data(pos=pos)
        self.canvas.update()

    def _as_vec3(self, v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] == 4:
            return v[:3]
        return v[:3]

    def _linear3_from_transform(self, T):
        """
        Robustly extract a 3x3 linear map from a VisPy transform that supports .map(),
        without relying on .matrix being available.

        We compute:
          A[:,0] = T([1,0,0]) - T([0,0,0])
          A[:,1] = T([0,1,0]) - T([0,0,0])
          A[:,2] = T([0,0,1]) - T([0,0,0])

        For pure rigid transforms this is the rotation matrix (possibly with tiny numerical noise).
        """
        o = self._as_vec3(T.map(np.array([0.0, 0.0, 0.0], dtype=np.float32)))
        ex = self._as_vec3(T.map(np.array([1.0, 0.0, 0.0], dtype=np.float32))) - o
        ey = self._as_vec3(T.map(np.array([0.0, 1.0, 0.0], dtype=np.float32))) - o
        ez = self._as_vec3(T.map(np.array([0.0, 0.0, 1.0], dtype=np.float32))) - o

        # Normalize columns (handles any slight scaling)
        def _safe_norm(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-12 else v

        ex = _safe_norm(ex)
        ey = _safe_norm(ey)
        ez = _safe_norm(ez)

        # Optional: re-orthogonalize to reduce drift (Gram-Schmidt)
        ex = _safe_norm(ex)
        ey = ey - ex * float(np.dot(ex, ey))
        ey = _safe_norm(ey)
        ez = np.cross(ex, ey)
        ez = _safe_norm(ez)

        A = np.stack([ex, ey, ez], axis=1).astype(np.float32)
        return A

    def _model_T_world(self):
        # model_node -> world(scene)
        # This includes: root(frame rotation) -> bno_origin -> model_scale
        return self.model_node.node_transform(self.view.scene)

    def _capture_eye_anchors_world_to_model_local(self):
        """
        Capture ONLY the eye anchor points into model/head-local coordinates.
        Uses current pose/frame, but because we invert the full model->world transform,
        the result is stable head-local anchors.
        """
        T = self._model_T_world()
        for idx in (0, 1):
            p = self.eye_params[idx]
            origin_w = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
            origin_l = self._as_vec3(T.imap(origin_w))
            self.eye_pf_anchor_local[idx] = origin_l

    def _update_eyes_from_per_frame(self):
        """
        For the current GUI index self.i:
          - eye anchor is fixed in head-local coords
          - eye direction comes from the SAME loaded data file, row by row
          - duplicated Arena_frame values do not matter
        """
        if not getattr(self, "eye_pf_enabled", False):
            return
        if self.n == 0:
            return
        if self.eye_pf_anchor_local[0] is None or self.eye_pf_anchor_local[1] is None:
            return

        # Virtual frame: hide eye vectors
        if self.i < self.data_index_offset:
            for idx in (0, 1):
                self.eye_lines[idx].visible = False
            return

        data_idx = int(self.i - self.data_index_offset)
        if data_idx < 0 or self.eye_pf_series[0] is None and self.eye_pf_series[1] is None:
            for idx in (0, 1):
                self.eye_lines[idx].visible = False
            return

        going_backward = (self.eye_pf_prev_i is not None and self.i < self.eye_pf_prev_i)
        self.eye_pf_prev_i = int(self.i)

        T = self._model_T_world()
        R = self._linear3_from_transform(T)

        for idx in (0, 1):
            want_visible = False
            if hasattr(self, "eye_enabled_chk") and self.eye_enabled_chk[idx] is not None:
                want_visible = bool(self.eye_enabled_chk[idx].isChecked())

            src_idx = 1 - idx if getattr(self, "eye_pf_swap_lr", False) else idx
            arr = self.eye_pf_series[src_idx]
            first_row = self.eye_pf_first_row[src_idx]

            if arr is None or data_idx >= len(arr):
                self.eye_lines[idx].visible = False
                continue

            if first_row is not None and data_idx < int(first_row):
                self.eye_lines[idx].visible = False
                continue

            yaw_deg = float(arr[data_idx, 0])
            pitch_deg = float(arr[data_idx, 1])

            if not (np.isfinite(yaw_deg) and np.isfinite(pitch_deg)):
                if self.eye_pf_hold_last and not going_backward and self.eye_pf_last_dirH[idx] is not None:
                    d_H = self.eye_pf_last_dirH[idx]
                else:
                    self.eye_lines[idx].visible = False
                    continue
            else:
                d_H = self._eye_dir_from_yaw_pitch(yaw_deg, pitch_deg)
                self.eye_pf_last_dirH[idx] = d_H

            origin_l = self.eye_pf_anchor_local[idx]
            origin_w = self._as_vec3(T.map(origin_l))

            d_W = (R @ np.asarray(d_H, dtype=np.float32).reshape(3))
            nrm = float(np.linalg.norm(d_W))
            if nrm > 1e-12:
                d_W = d_W / nrm

            L = float(self.eye_params[idx].get("length", 2.0))
            end_w = origin_w + d_W * L

            pos = np.vstack([origin_w, end_w]).astype(np.float32)
            self.eye_lines[idx].set_data(pos=pos)
            self.eye_lines[idx].visible = bool(want_visible)

    def _capture_eyes_world_to_model_local(self):
        """
        Step (3): Capture current eye vectors (defined in world axes) and convert them into model-local endpoints.
        After this, we can re-render them each frame so they stay rigidly attached to the model/BNO frame.
        """
        T = self._model_T_world()

        for idx in (0, 1):
            p = self.eye_params[idx]
            origin_w = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
            d_w = self._eye_dir_from_yaw_pitch(float(p["yaw"]), float(p["pitch"]))
            end_w = origin_w + d_w * float(p["length"])

            origin_l = self._as_vec3(T.imap(origin_w))
            end_l = self._as_vec3(T.imap(end_w))

            self.eye_local_segments[idx] = (origin_l, end_l)

    def _update_eyes_from_model_local(self):
        """
        Step (4): For the current frame, map captured local endpoints back to world and update the visuals.
        """
        if not self.eye_follow_model:
            return

        T = self._model_T_world()

        for idx in (0, 1):
            seg = self.eye_local_segments[idx]
            if seg is None:
                continue

            origin_l, end_l = seg
            origin_w = self._as_vec3(T.map(origin_l))
            end_w = self._as_vec3(T.map(end_l))

            pos = np.vstack([origin_w, end_w]).astype(np.float32)
            self.eye_lines[idx].set_data(pos=pos)

    def _apply_eye_from_ui(self, idx: int):
        p = self.eye_params[idx]
        p["x"] = float(self.eye_pos_spin[idx][0].value())
        p["y"] = float(self.eye_pos_spin[idx][1].value())
        p["z"] = float(self.eye_pos_spin[idx][2].value())
        p["yaw"] = float(self.eye_yaw_spin[idx].value())
        p["pitch"] = float(self.eye_pitch_spin[idx].value())
        p["length"] = float(self.eye_len_spin[idx].value())
        self._update_eye_vector(idx)

    def _set_eye_visible(self, idx: int, visible: bool):
        # If visuals not built yet, ignore
        if hasattr(self, "eye_lines") and self.eye_lines and idx < len(self.eye_lines):
            self.eye_lines[idx].visible = bool(visible)
        self.canvas.update()

    def _reset_eye_vector(self, idx: int):
        d = self.eye_defaults[idx]
        # Update widgets (sliders will follow because they are bound to spins)
        self.eye_pos_spin[idx][0].setValue(float(d["x"]))
        self.eye_pos_spin[idx][1].setValue(float(d["y"]))
        self.eye_pos_spin[idx][2].setValue(float(d["z"]))
        self.eye_yaw_spin[idx].setValue(float(d["yaw"]))
        self.eye_pitch_spin[idx].setValue(float(d["pitch"]))
        self.eye_len_spin[idx].setValue(float(d["length"]))
        self._apply_eye_from_ui(idx)

    def _flip_view_roll(self):
        cam = self.view.camera
        # Toggle the camera's "up" axis so the whole scene appears upside-down/upright
        cam.up = '-z' if cam.up == '+z' else '+z'

        # Spin azimuth 180° so the scene keeps roughly the same facing
        def _norm180(a):  # normalize to [-180, 180]
            return ((float(a) + 180.0) % 360.0) - 180.0

        cam.azimuth = _norm180(cam.azimuth + 180.0)

        # Reflect into the UI (so the numbers match what you see)
        if hasattr(self, "cam_azim_spin"):
            self.cam_azim_spin.blockSignals(True)
            self.cam_azim_slider.blockSignals(True)
            self.cam_azim_spin.setValue(cam.azimuth)
            self.cam_azim_slider.setValue(int(round(cam.azimuth)))
            self.cam_azim_spin.blockSignals(False)
            self.cam_azim_slider.blockSignals(False)

        self._update_axis_gizmo()
        self.canvas.update()

    @staticmethod
    def _fold_turntable_elev_azim(elev_deg: float, azim_deg: float):
        e, a = float(elev_deg), float(azim_deg)
        # Fold elevation back into [-90, 90] while rolling azimuth by 180° when crossing a pole
        while e > 90.0:
            e = 180.0 - e
            a += 180.0
        while e < -90.0:
            e = -180.0 - e
            a += 180.0
        # Normalize azimuth to [-180, 180] (nicer numbers)
        a = ((a + 180.0) % 360.0) - 180.0
        return e, a

    def _apply_camera_from_ui(self):
        cam = self.view.camera

        elev_ui = float(self.cam_elev_spin.value())
        azim_ui = float(self.cam_azim_spin.value())
        elev_cam, azim_cam = self._fold_turntable_elev_azim(elev_ui, azim_ui)

        # Apply to camera
        cam.elevation = elev_cam
        cam.azimuth = azim_cam
        cam.distance = float(self.cam_dist_spin.value())
        cam.center = (
            float(self.cam_cx_spin.value()),
            float(self.cam_cy_spin.value()),
            float(self.cam_cz_spin.value()),
        )

        # Optional but recommended: reflect folded values back into the UI, so it matches the view
        if (abs(elev_cam - elev_ui) > 1e-6) or (abs(azim_cam - azim_ui) > 1e-6):
            self.cam_elev_spin.blockSignals(True)
            self.cam_azim_spin.blockSignals(True)
            self.cam_elev_spin.setValue(elev_cam)
            self.cam_azim_spin.setValue(azim_cam)
            self.cam_elev_spin.blockSignals(False)
            self.cam_azim_spin.blockSignals(False)

            # keep sliders in sync too
            self.cam_elev_slider.blockSignals(True)
            self.cam_azim_slider.blockSignals(True)
            self.cam_elev_slider.setValue(int(round(elev_cam)))
            self.cam_azim_slider.setValue(int(round(azim_cam)))
            self.cam_elev_slider.blockSignals(False)
            self.cam_azim_slider.blockSignals(False)

        self._update_axis_gizmo()
        self.canvas.update()

    def _toggle_neutral(self, on: bool):
        self.neutral_pose = bool(on)
        self._apply_frame(self.i)

    # ----- OBJ Transform panel -----
    def _build_obj_transform_ui(self):
        gb = QtWidgets.QGroupBox("External OBJ Transform")
        grid = QtWidgets.QGridLayout(gb);
        grid.setContentsMargins(10, 8, 10, 8);
        grid.setHorizontalSpacing(10)

        def make_slider(min_i, max_i, tick=10):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(min_i, max_i)
            s.setTickInterval(tick);
            s.setSingleStep(1)
            return s

        def bind_pair(slider, spin, to_spin, to_slider, on_change):
            def _s2p(val):
                spin.blockSignals(True);
                spin.setValue(to_spin(val));
                spin.blockSignals(False);
                on_change()

            slider.valueChanged.connect(_s2p)

            def _p2s(val):
                slider.blockSignals(True);
                slider.setValue(to_slider(val));
                slider.blockSignals(False);
                on_change()

            spin.valueChanged.connect(_p2s)

        # Scale (0.10 .. 5.00)
        self.scale_slider = make_slider(10, 500, tick=25)
        self.scale_spin = QtWidgets.QDoubleSpinBox();
        self.scale_spin.setRange(0.10, 5.00);
        self.scale_spin.setDecimals(2);
        self.scale_spin.setSingleStep(0.01)
        bind_pair(self.scale_slider, self.scale_spin,
                  to_spin=lambda v: v / 100.0,
                  to_slider=lambda v: int(round(v * 100)),
                  on_change=self._apply_external_transform)

        # Position X/Y/Z (-10.00 .. +10.00)
        self.tx_slider = make_slider(-1000, 1000, tick=50)
        self.ty_slider = make_slider(-1000, 1000, tick=50)
        self.tz_slider = make_slider(-1000, 1000, tick=50)
        self.tx_spin = QtWidgets.QDoubleSpinBox();
        self.tx_spin.setRange(-5.00, 5.00);
        self.tx_spin.setDecimals(2);
        self.tx_spin.setSingleStep(0.01)
        self.ty_spin = QtWidgets.QDoubleSpinBox();
        self.ty_spin.setRange(-5.00, 5.00);
        self.ty_spin.setDecimals(2);
        self.ty_spin.setSingleStep(0.01)
        self.tz_spin = QtWidgets.QDoubleSpinBox();
        self.tz_spin.setRange(-5.00, 5.00);
        self.tz_spin.setDecimals(2);
        self.tz_spin.setSingleStep(0.01)

        bind_pair(self.tx_slider, self.tx_spin, lambda v: v / 100.0, lambda v: int(round(v * 100)),
                  self._apply_external_transform)
        bind_pair(self.ty_slider, self.ty_spin, lambda v: v / 100.0, lambda v: int(round(v * 100)),
                  self._apply_external_transform)
        bind_pair(self.tz_slider, self.tz_spin, lambda v: v / 100.0, lambda v: int(round(v * 100)),
                  self._apply_external_transform)

        # Rotation Roll/Pitch/Yaw (-180 .. +180 deg)
        self.roll_slider = make_slider(-180, 180, tick=30)
        self.pitch_slider = make_slider(-180, 180, tick=30)
        self.yaw_slider = make_slider(-180, 180, tick=30)
        self.roll_spin = QtWidgets.QDoubleSpinBox();
        self.roll_spin.setRange(-180.0, 180.0);
        self.roll_spin.setDecimals(1);
        self.roll_spin.setSingleStep(0.1)
        self.pitch_spin = QtWidgets.QDoubleSpinBox();
        self.pitch_spin.setRange(-180.0, 180.0);
        self.pitch_spin.setDecimals(1);
        self.pitch_spin.setSingleStep(0.1)
        self.yaw_spin = QtWidgets.QDoubleSpinBox();
        self.yaw_spin.setRange(-180.0, 180.0);
        self.yaw_spin.setDecimals(1);
        self.yaw_spin.setSingleStep(0.1)

        bind_pair(self.roll_slider, self.roll_spin, float, int, self._apply_external_transform)
        bind_pair(self.pitch_slider, self.pitch_spin, float, int, self._apply_external_transform)
        bind_pair(self.yaw_slider, self.yaw_spin, float, int, self._apply_external_transform)

        # Layout
        r = 0
        grid.addWidget(QtWidgets.QLabel("Scale"), r, 0);
        grid.addWidget(self.scale_slider, r, 1);
        grid.addWidget(self.scale_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("X"), r, 0);
        grid.addWidget(self.tx_slider, r, 1);
        grid.addWidget(self.tx_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Y"), r, 0);
        grid.addWidget(self.ty_slider, r, 1);
        grid.addWidget(self.ty_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Z"), r, 0);
        grid.addWidget(self.tz_slider, r, 1);
        grid.addWidget(self.tz_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Roll (°)"), r, 0);
        grid.addWidget(self.roll_slider, r, 1);
        grid.addWidget(self.roll_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Pitch (°)"), r, 0);
        grid.addWidget(self.pitch_slider, r, 1);
        grid.addWidget(self.pitch_spin, r, 2);
        r += 1
        grid.addWidget(QtWidgets.QLabel("Yaw (°)"), r, 0);
        grid.addWidget(self.yaw_slider, r, 1);
        grid.addWidget(self.yaw_spin, r, 2);
        r += 1

        self.reset_btn = QtWidgets.QPushButton("Reset OBJ Transform")
        self.reset_btn.clicked.connect(self._reset_external_transform)
        grid.addWidget(self.reset_btn, r, 0, 1, 3)

        # Defaults
        self.scale_spin.setValue(1.00)
        self.tx_spin.setValue(0.0);
        self.ty_spin.setValue(0.0);
        self.tz_spin.setValue(0.0)
        self.roll_spin.setValue(0.0);
        self.pitch_spin.setValue(0.0);
        self.yaw_spin.setValue(0.0)
        return gb

    # ----- Eye vectors panel -----
    def _build_eye_vectors_ui(self):
        gb = QtWidgets.QGroupBox("Eye Vectors (World → Capture to BNO)")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)

        def make_slider(min_i, max_i, tick=10):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(min_i, max_i)
            s.setTickInterval(tick)
            s.setSingleStep(1)
            return s

        def bind_pair(slider, spin, to_spin, to_slider, on_change):
            def _s2p(val):
                spin.blockSignals(True)
                spin.setValue(to_spin(val))
                spin.blockSignals(False)
                on_change()

            slider.valueChanged.connect(_s2p)

            def _p2s(val):
                slider.blockSignals(True)
                slider.setValue(to_slider(val))
                slider.blockSignals(False)
                on_change()

            spin.valueChanged.connect(_p2s)

        r = 0
        grid.addWidget(QtWidgets.QLabel("Manual vectors are defined in WORLD axes."), r, 0, 1, 3)
        r += 1

        # Storage for widgets (these names are REQUIRED by your _apply_eye_from_ui)
        self.eye_pos_slider = [[None, None, None], [None, None, None]]
        self.eye_pos_spin = [[None, None, None], [None, None, None]]
        self.eye_yaw_slider = [None, None]
        self.eye_pitch_slider = [None, None]
        self.eye_len_slider = [None, None]
        self.eye_yaw_spin = [None, None]
        self.eye_pitch_spin = [None, None]
        self.eye_len_spin = [None, None]
        self.eye_enabled_chk = [None, None]

        # ---- Follow-model toggle (captures current world vectors into model-local and then animates them) ----
        self.chk_eye_follow_model = QtWidgets.QCheckBox("Follow model (capture world → BNO now)")
        self.chk_eye_follow_model.setChecked(False)

        def _on_follow_toggle(v):
            self.eye_follow_model = bool(v)
            if self.eye_follow_model:
                # capture once at the current pose/frame
                self._capture_eyes_world_to_model_local()
                # ensure they are visible if user enabled them
                self._update_eyes_from_model_local()
            else:
                # back to world-edit mode: redraw from eye_params
                self._update_eye_vector(0)
                self._update_eye_vector(1)
            self.canvas.update()

        self.chk_eye_follow_model.toggled.connect(_on_follow_toggle)
        grid.addWidget(self.chk_eye_follow_model, r, 0, 1, 3)
        r += 1

        # ---- Per-timestamp direction mode controls ----
        self.lbl_eye_pf_file = QtWidgets.QLabel("Eye data source: (same loaded data file)")
        self.chk_eye_pf_enable = QtWidgets.QCheckBox("Use per-row eye directions from loaded data file")
        self.chk_eye_pf_hold = QtWidgets.QCheckBox("Hold last valid direction through gaps")
        self.chk_eye_pf_hold.setChecked(False)
        self.eye_pf_hold_last = False

        self.chk_eye_pf_swap = QtWidgets.QCheckBox("Swap L/R directions (green ↔ blue)")
        self.chk_eye_pf_swap.setChecked(False)

        def _on_pf_swap(v):
            self.eye_pf_swap_lr = bool(v)
            if getattr(self, "eye_pf_enabled", False):
                self._update_eyes_from_per_frame()
                self.canvas.update()

        self.chk_eye_pf_swap.toggled.connect(_on_pf_swap)

        def _on_pf_hold(v):
            self.eye_pf_hold_last = bool(v)

        self.chk_eye_pf_hold.toggled.connect(_on_pf_hold)

        def _on_pf_enable(v):
            self.eye_pf_enabled = bool(v)

            if self.eye_pf_enabled:
                self.eye_follow_model = False
                self.chk_eye_follow_model.blockSignals(True)
                self.chk_eye_follow_model.setChecked(False)
                self.chk_eye_follow_model.blockSignals(False)

                self._capture_eye_anchors_world_to_model_local()
                self.eye_pf_last_dirH = [None, None]
                self._update_eyes_from_per_frame()
            else:
                self._update_eye_vector(0)
                self._update_eye_vector(1)

            self.canvas.update()

        self.chk_eye_pf_enable.toggled.connect(_on_pf_enable)

        grid.addWidget(self.lbl_eye_pf_file, r, 0, 1, 3);
        r += 1
        grid.addWidget(self.chk_eye_pf_enable, r, 0, 1, 3);
        r += 1
        grid.addWidget(self.chk_eye_pf_hold, r, 0, 1, 3);
        r += 1
        grid.addWidget(self.chk_eye_pf_swap, r, 0, 1, 3);
        r += 1

        def add_eye_block(idx: int, title: str):
            nonlocal r
            title_lbl = QtWidgets.QLabel(title)

            chk = QtWidgets.QCheckBox("Show")
            chk.setChecked(False)  # OFF by default
            chk.toggled.connect(lambda v, _idx=idx: self._set_eye_visible(_idx, v))
            self.eye_enabled_chk[idx] = chk

            btn_reset = QtWidgets.QPushButton("Reset")
            btn_reset.clicked.connect(lambda _=False, _idx=idx: self._reset_eye_vector(_idx))

            roww = QtWidgets.QWidget()
            rowl = QtWidgets.QHBoxLayout(roww)
            rowl.setContentsMargins(0, 0, 0, 0)
            rowl.addWidget(title_lbl)
            rowl.addStretch(1)
            rowl.addWidget(chk)
            rowl.addWidget(btn_reset)

            grid.addWidget(roww, r, 0, 1, 3)
            r += 1

            # Position: -10..+10 (slider -1000..1000 mapped /100)
            for j, lab in enumerate(["Pos X", "Pos Y", "Pos Z"]):
                s = make_slider(-1000, 1000, tick=100)
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(-10.0, 10.0)
                sp.setDecimals(2)
                sp.setSingleStep(0.05)

                bind_pair(
                    s, sp,
                    to_spin=lambda v: v / 100.0,
                    to_slider=lambda v: int(round(v * 100)),
                    on_change=lambda _idx=idx: self._apply_eye_from_ui(_idx),
                )

                self.eye_pos_slider[idx][j] = s
                self.eye_pos_spin[idx][j] = sp

                grid.addWidget(QtWidgets.QLabel(lab), r, 0)
                grid.addWidget(s, r, 1)
                grid.addWidget(sp, r, 2)
                r += 1

            # Direction: yaw/pitch in degrees (world)
            self.eye_yaw_slider[idx] = make_slider(-180, 180, tick=30)
            self.eye_pitch_slider[idx] = make_slider(-180, 180, tick=30)

            self.eye_yaw_spin[idx] = QtWidgets.QDoubleSpinBox()
            self.eye_pitch_spin[idx] = QtWidgets.QDoubleSpinBox()
            for sp in (self.eye_yaw_spin[idx], self.eye_pitch_spin[idx]):
                sp.setRange(-180.0, 180.0)
                sp.setDecimals(1)
                sp.setSingleStep(0.1)

            bind_pair(
                self.eye_yaw_slider[idx], self.eye_yaw_spin[idx],
                to_spin=float, to_slider=int,
                on_change=lambda _idx=idx: self._apply_eye_from_ui(_idx),
            )
            bind_pair(
                self.eye_pitch_slider[idx], self.eye_pitch_spin[idx],
                to_spin=float, to_slider=int,
                on_change=lambda _idx=idx: self._apply_eye_from_ui(_idx),
            )

            grid.addWidget(QtWidgets.QLabel("Yaw"), r, 0)
            grid.addWidget(self.eye_yaw_slider[idx], r, 1)
            grid.addWidget(self.eye_yaw_spin[idx], r, 2)
            r += 1

            grid.addWidget(QtWidgets.QLabel("Pitch"), r, 0)
            grid.addWidget(self.eye_pitch_slider[idx], r, 1)
            grid.addWidget(self.eye_pitch_spin[idx], r, 2)
            r += 1

            # Length: 0..20 (slider 0..2000 mapped /100)
            self.eye_len_slider[idx] = make_slider(0, 2000, tick=100)
            self.eye_len_spin[idx] = QtWidgets.QDoubleSpinBox()
            self.eye_len_spin[idx].setRange(0.0, 20.0)
            self.eye_len_spin[idx].setDecimals(2)
            self.eye_len_spin[idx].setSingleStep(0.05)

            bind_pair(
                self.eye_len_slider[idx], self.eye_len_spin[idx],
                to_spin=lambda v: v / 100.0,
                to_slider=lambda v: int(round(v * 100)),
                on_change=lambda _idx=idx: self._apply_eye_from_ui(_idx),
            )

            grid.addWidget(QtWidgets.QLabel("Length"), r, 0)
            grid.addWidget(self.eye_len_slider[idx], r, 1)
            grid.addWidget(self.eye_len_spin[idx], r, 2)
            r += 1

        add_eye_block(0, "Eye vector A")
        add_eye_block(1, "Eye vector B")

        # Initialize UI from current params + keep OFF by default
        for idx in (0, 1):
            self.eye_pos_spin[idx][0].setValue(float(self.eye_params[idx]["x"]))
            self.eye_pos_spin[idx][1].setValue(float(self.eye_params[idx]["y"]))
            self.eye_pos_spin[idx][2].setValue(float(self.eye_params[idx]["z"]))
            self.eye_yaw_spin[idx].setValue(float(self.eye_params[idx]["yaw"]))
            self.eye_pitch_spin[idx].setValue(float(self.eye_params[idx]["pitch"]))
            self.eye_len_spin[idx].setValue(float(self.eye_params[idx]["length"]))

            # OFF by default
            self.eye_enabled_chk[idx].setChecked(False)
            self._set_eye_visible(idx, False)

            # Still compute geometry so it is ready when toggled on
            self._apply_eye_from_ui(idx)

        return gb

    def _reset_external_transform(self):
        self.scale_spin.setValue(1.00)
        self.tx_spin.setValue(0.0);
        self.ty_spin.setValue(0.0);
        self.tz_spin.setValue(0.0)
        self.roll_spin.setValue(0.0);
        self.pitch_spin.setValue(0.0);
        self.yaw_spin.setValue(0.0)
        self._apply_external_transform()

    def _apply_external_transform(self):
        """Uniform scale + yaw/pitch/roll + translation (MATCHES CSV/BNO mapping)."""
        S = float(self.scale_spin.value())
        roll = float(self.roll_spin.value())
        pitch = float(self.pitch_spin.value())
        yaw = float(self.yaw_spin.value())
        tx, ty, tz = float(self.tx_spin.value()), float(self.ty_spin.value()), float(self.tz_spin.value())

        # Use the SAME convention as CSV/BNO (ALIGN_WORLD/ALIGN_LOCAL)
        R = rotation_from_ypr(roll, pitch, yaw).astype(np.float32)  # 3x3

        M = np.eye(4, dtype=np.float32)

        # Match BNO origin convention (no transpose)
        M[:3, :3] = (R * S)

        # Keep your existing translation convention
        M[3, 0:3] = [tx, ty, tz]

        self.external_transform.matrix = M
        self.canvas.update()

    # ----- camera helpers -----
    def _reset_view(self):
        cam = self.view.camera
        cam.elevation = self.CAM_ELEV
        cam.azimuth = self.CAM_AZIM
        cam.distance = self.CAM_DIST
        cam.center = self.CAM_CENTER
        cam.set_range(**self.CAM_RANGE)
        self._update_axis_gizmo()
        self.canvas.update()
        # sync UI
        if hasattr(self, "cam_elev_spin"):
            self.cam_elev_spin.setValue(self.CAM_ELEV)
            self.cam_azim_spin.setValue(self.CAM_AZIM)
            self.cam_dist_spin.setValue(self.CAM_DIST)
            self.cam_cx_spin.setValue(self.CAM_CENTER[0])
            self.cam_cy_spin.setValue(self.CAM_CENTER[1])
            self.cam_cz_spin.setValue(self.CAM_CENTER[2])

    def _snap_view(self, elev, azim):
        elev_cam, azim_cam = self._fold_turntable_elev_azim(float(elev), float(azim))
        cam = self.view.camera
        cam.elevation = elev_cam
        cam.azimuth = azim_cam
        self._update_axis_gizmo()
        self.canvas.update()

        # sync UI (without re-triggering handlers)
        if hasattr(self, "cam_elev_spin"):
            self.cam_elev_spin.blockSignals(True)
            self.cam_azim_spin.blockSignals(True)
            self.cam_elev_spin.setValue(elev_cam)
            self.cam_azim_spin.setValue(azim_cam)
            self.cam_elev_spin.blockSignals(False)
            self.cam_azim_spin.blockSignals(False)

            self.cam_elev_slider.blockSignals(True)
            self.cam_azim_slider.blockSignals(True)
            self.cam_elev_slider.setValue(int(round(elev_cam)))
            self.cam_azim_slider.setValue(int(round(azim_cam)))
            self.cam_elev_slider.blockSignals(False)
            self.cam_azim_slider.blockSignals(False)

    def _set_lock_view(self, checked: bool):
        self.view_locked = bool(checked)
        self.view.camera.interactive = not self.view_locked
        for w in (self.cam_elev_slider, self.cam_elev_spin,
                  self.cam_azim_slider, self.cam_azim_spin,
                  self.cam_dist_slider, self.cam_dist_spin,
                  self.cam_cx_slider, self.cam_cx_spin,
                  self.cam_cy_slider, self.cam_cy_spin,
                  self.cam_cz_slider, self.cam_cz_spin):
            w.setEnabled(not self.view_locked)
        self.canvas.update()

    def _toggle_board_only(self, hidden: bool):
        vis = not bool(hidden)
        self.board.visible = vis
        self.board_edges.visible = vis
        self.canvas.update()

    # ----- External OBJ -----
    def _browse_obj(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .obj model", "", "OBJ Files (*.obj)")
        if path:
            try:
                self._load_external_obj(path)
                self.statusBar().showMessage(f"Loaded OBJ: {os.path.basename(path)}", 5000)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "OBJ load error", str(e))

    def _clear_external_mesh(self):
        for m in self.external_mesh_visuals:
            try:
                m.parent = None; m.destroy()
            except Exception:
                pass
        self.external_mesh_visuals = []

    def _load_external_obj(self, path: str):
        self._clear_external_mesh()
        self.last_obj_path = path
        # Load geometry
        if HAVE_TRIMESH:
            tm = trimesh.load(path, force='scene')
            geoms = [tm] if isinstance(tm, trimesh.Trimesh) else list(tm.geometry.values())
            if not geoms: raise RuntimeError("OBJ contains no geometry")
            all_v = np.concatenate([g.vertices for g in geoms], axis=0)
            vmin, vmax = all_v.min(axis=0), all_v.max(axis=0)
            size = float((vmax - vmin).max()) or 1.0
            scale = (4.0 / size) if self._external_auto_scale else float(self._external_manual_scale)
            for g in geoms:
                v = (g.vertices - 0.5 * (vmin + vmax)) * scale
                f = g.faces.astype(np.uint32)
                mesh_vis = Mesh(
                    vertices=v.astype(np.float32),
                    faces=f,
                    color=(0.45, 0.45, 0.45, 1.0),
                    shading=None,
                    parent=self.external_node
                )

                mesh_vis.attach(
                    ShadingFilter(
                        # Camera-facing light so whichever side you look at is lit
                        light_dir=(-1.0, 0.0, 0.0),

                        # More ambient to avoid huge +X vs -X contrast
                        ambient_light=(0.85, 0.85, 0.85, 1.0),
                        diffuse_light=(0.35, 0.35, 0.35, 1.0),

                        # Keep specular subtle (high specular can exaggerate “one side”)
                        specular_light=(0.08, 0.08, 0.08, 1.0),
                        shininess=20.0
                    )
                )

                # SOLID / OPAQUE
                mesh_vis.set_gl_state(preset='opaque', depth_test=True, blend=False, depth_mask=True, cull_face=False)
                self.external_mesh_visuals.append(mesh_vis)
        else:
            verts, faces = [], []
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if not line or line.startswith("#"): continue
                    sp = line.strip().split()
                    if not sp: continue
                    if sp[0] == "v" and len(sp) >= 4:
                        verts.append([float(sp[1]), float(sp[2]), float(sp[3])])
                    elif sp[0] == "f" and len(sp) >= 4:
                        tri = []
                        for tok in sp[1:4]: tri.append(int(tok.split("/")[0]) - 1)
                        faces.append(tri)
            if not verts or not faces:
                raise RuntimeError("OBJ missing vertices or faces; or not triangular")
            v = np.asarray(verts, np.float32)
            vmin, vmax = v.min(axis=0), v.max(axis=0)
            size = float((vmax - vmin).max()) or 1.0
            scale = (4.0 / size) if self._external_auto_scale else float(self._external_manual_scale)
            v = (v - 0.5 * (vmin + vmax)) * scale
            f = np.asarray(faces, np.uint32)
            mesh_vis = Mesh(
                vertices=v.astype(np.float32),
                faces=f,
                color=(0.45, 0.45, 0.45, 1.0),
                shading=None,
                parent=self.external_node
            )

            mesh_vis.attach(
                ShadingFilter(
                    # Camera-facing light so whichever side you look at is lit
                    light_dir=(-1.0, 0.0, 0.0),

                    # More ambient to avoid huge +X vs -X contrast
                    ambient_light=(0.85, 0.85, 0.85, 1.0),
                    diffuse_light=(0.35, 0.35, 0.35, 1.0),

                    # Keep specular subtle (high specular can exaggerate “one side”)
                    specular_light=(0.08, 0.08, 0.08, 1.0),
                    shininess=20.0
                )
            )

            mesh_vis.set_gl_state(preset='opaque', depth_test=True, blend=False, depth_mask=True, cull_face=False)
            self.external_mesh_visuals.append(mesh_vis)

        self._reset_external_transform()
        self._toggle_external_visibility(self.act_show_obj.isChecked())
        self.view.camera.set_range()

    # ----- visibility / UI handlers -----
    def _toggle_board_visibility(self, hidden: bool):
        vis = not bool(hidden)
        for w in (self.board, self.board_edges, self.bno, self.bno_edges, self.stand, self.stand_edges, *self.bno_axes):
            w.visible = vis
        self.canvas.update()

    def _toggle_external_visibility(self, visible: bool):
        for m in self.external_mesh_visuals: m.visible = bool(visible)
        self.canvas.update()

    # ----- toggles -----
    def _toggle_bno_axes(self, visible: bool):
        for a in self.bno_axes: a.visible = bool(visible)
        self.canvas.update()

    def _toggle_dir_vec(self, visible: bool):
        self.dir_arrow.visible = bool(visible);
        self.canvas.update()

    def _change_tick_size(self):
        val, ok = QtWidgets.QInputDialog.getInt(self, "Tick size", "Logical font size (8–192):", value=96, min=8,
                                                max=192, step=1)
        if ok: self.axes_cube.set_base_font_size(val)

    def _on_canvas_resize(self, event=None):
        pr = getattr(self.canvas, 'pixel_ratio', 1.0);
        vh = self.canvas.size[1]
        self.axes_cube.set_screen_params(pr, vh)

    def _update_axis_gizmo(self):
        if hasattr(self, "axis_gizmo"):
            try:
                self.axis_gizmo.sync_from_camera(self.view.camera)
            except Exception:
                pass

    # ----- playback marks -----
    def _update_hud_marks(self):
        self.hud_marks = f"IN {self.mark_in - self.data_index_offset}  OUT {self.mark_out - self.data_index_offset}" if self.n > 0 else ""
        self._apply_frame(self.i)

    def _mark_in(self):
        self.mark_in = int(self.i)
        if self.mark_out < self.mark_in: self.mark_out = self.mark_in
        self._update_hud_marks()

    def _mark_out(self):
        self.mark_out = int(self.i)
        if self.mark_out < self.mark_in: self.mark_in = self.mark_out
        self._update_hud_marks()

    def _toggle_loop(self, on: bool):
        self.loop_selection = bool(on)

    # ----- HUD composer -----
    def _compose_hud_text(self, idx=None):
        if idx is None:
            idx = self.i
        if self.n == 0:
            return ""

        cur_clock = self._fmt_clock(self.gui_time[idx])
        offset = int(getattr(self, "data_index_offset", 0))
        n_data = int(len(self.roll))

        if offset == 1 and idx == 0:
            frame_line = "Virtual Identity Pose"
            roll = 0.00
            pitch = 0.00
            yaw = 0.00
            arena_txt = "-"
        else:
            data_idx = int(idx - offset)

            arena_val = self.arena_frame_data[data_idx]
            arena_video_val = self.arena_video_frame_data[data_idx] if data_idx < len(
                self.arena_video_frame_data) else np.nan
            left_video_val = self.left_video_frame_data[data_idx] if data_idx < len(
                self.left_video_frame_data) else np.nan
            right_video_val = self.right_video_frame_data[data_idx] if data_idx < len(
                self.right_video_frame_data) else np.nan

            def _fmt_vid(v):
                if np.isfinite(v):
                    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.3f}"
                return "-"
            if np.isfinite(arena_val):
                if abs(arena_val - round(arena_val)) < 1e-6:
                    arena_txt = str(int(round(arena_val)))
                else:
                    arena_txt = f"{arena_val:.3f}"
            else:
                arena_txt = "NaN"

            frame_line = (
                f"Data row {data_idx}/{n_data - 1} | "
                f"Arena_frame {arena_txt} | "
                f"Video A/L/R: {_fmt_vid(arena_video_val)} / {_fmt_vid(left_video_val)} / {_fmt_vid(right_video_val)}"
            )
            roll = float(self.roll[data_idx])
            pitch = float(self.pitch[data_idx])
            yaw = float(self.yaw[data_idx])

        hud = (
            f"Time: {cur_clock} / {self.total_duration_str}\n"
            f"{frame_line}\n"
            f"Roll {roll:.2f}°  Pitch {pitch:.2f}°  Yaw {yaw:.2f}°"
        )

        if self.hud_marks:
            hud += f"\n{self.hud_marks}"
        if hasattr(self, "actual_fps"):
            hud += f"\nActual FPS: {self.actual_fps:.1f}"

        return hud
    # ----- animation & frames -----
    def _apply_frame(self, idx):
        if self.n == 0: return
        self.i = max(0, min(int(idx), self.n - 1))
        if self.neutral_pose:
            self.root.transform.matrix = np.eye(4, dtype=np.float32)
        else:
            self.root.transform.matrix = self.transforms[self.i]

        # Update eyes for this frame (per-frame mode takes priority)
        if getattr(self, "eye_pf_enabled", False):
            self._update_eyes_from_per_frame()
        else:
            self._update_eyes_from_model_local()

        self._update_all_videos()

        self.hud_lbl.setText(self._compose_hud_text(self.i))
        self.canvas.update()
        self.slider.blockSignals(True);
        self.slider.setValue(self.i);
        self.slider.blockSignals(False)
        offset = int(getattr(self, "data_index_offset", 0))

        if offset == 1 and self.i == 0:
            self.setWindowTitle("Virtual Identity Pose")
        else:
            data_idx = int(self.i - offset)
            arena_val = self.arena_frame_data[data_idx]

            if np.isfinite(arena_val) and abs(arena_val - round(arena_val)) < 1e-6:
                arena_txt = str(int(round(arena_val)))
            elif np.isfinite(arena_val):
                arena_txt = f"{arena_val:.3f}"
            else:
                arena_txt = "NaN"

            self.setWindowTitle(f"Data row {data_idx} | Arena_frame {arena_txt} | t={self.gui_time[self.i]:.4f}s")

    def _on_timer(self):
        if self.n == 0:
            return

        now = time.perf_counter()

        if self.play_start_wall is None:
            self.play_start_wall = now
            self.play_start_idx = int(self.i)
            self.play_start_gui_time = float(self.gui_time[self.i])

        elapsed = (now - self.play_start_wall) * float(self.speed.value())
        target_time = self.play_start_gui_time + elapsed

        if self.loop_selection and self.mark_out >= self.mark_in:
            t0 = float(self.gui_time[self.mark_in])
            t1 = float(self.gui_time[self.mark_out])
            span = max(1e-9, t1 - t0)
            target_time = t0 + ((target_time - t0) % span)

            nxt = int(np.searchsorted(self.gui_time, target_time, side='right') - 1)
            nxt = max(self.mark_in, min(nxt, self.mark_out))
        else:
            nxt = int(np.searchsorted(self.gui_time, target_time, side='right') - 1)
            nxt = max(0, min(nxt, self.n - 1))

            if nxt >= self.n - 1 and target_time >= float(self.gui_time[-1]):
                self.timer.stop()
                self.btn_play.setText("▶ Play")
                self.play_start_wall = None

        if nxt != self.i:
            self._apply_frame(nxt)

        self._fps_samples += 1
        dt = now - self._fps_last_t
        if dt >= 0.5:
            self.actual_fps = self._fps_samples / dt
            self._fps_samples = 0
            self._fps_last_t = now
            self.hud_lbl.setText(self._compose_hud_text(self.i))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._update_all_videos()
        except Exception:
            pass

    # ----- fractional FPS management -----
    def _current_interval_ms(self):
        fps_val = float(self.fps.value())
        return max(1, int(round(1000.0 / fps_val)))

    def _update_timer_interval(self, *_):
        if self.timer.isActive(): self.timer.start(self._current_interval_ms())

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ Play")
            self.play_start_wall = None
            return

        if self.n == 0:
            return

        # If we are on the virtual frame, jump immediately to the first real data frame.
        if int(getattr(self, "data_index_offset", 0)) == 1 and int(self.i) == 0:
            if self.n > 1:
                self._apply_frame(1)
            else:
                return

        self.play_start_wall = time.perf_counter()
        self.play_start_idx = int(self.i)
        self.play_start_gui_time = float(self.gui_time[self.i])
        self._fps_samples = 0
        self._fps_last_t = self.play_start_wall
        self.timer.start(self._current_interval_ms())
        self.btn_play.setText("⏸ Pause")

    def _step_size(self):
        return max(1, int(self.step_jump.value()))

    def _step_back(self):
        self.timer.stop()
        self.btn_play.setText("▶ Play")

        step = self._step_size()
        prv = self.i - step

        if self.loop_selection and self.mark_out >= self.mark_in:
            if prv < self.mark_in:
                span = self.mark_out - self.mark_in + 1
                if span > 0:
                    prv = self.mark_in + ((prv - self.mark_in) % span)

        self._apply_frame(prv)

    def _step_fwd(self):
        self.timer.stop()
        self.btn_play.setText("▶ Play")

        step = self._step_size()
        nxt = self.i + step

        if self.loop_selection and self.mark_out >= self.mark_in:
            if nxt > self.mark_out:
                span = self.mark_out - self.mark_in + 1
                if span > 0:
                    nxt = self.mark_in + ((nxt - self.mark_in) % span)

        self._apply_frame(nxt)

    def _on_slider(self, val):
        self.timer.stop();
        self.btn_play.setText("▶ Play")
        self._apply_frame(val)

    # ======= Save/Load Scene Settings (JSON) =======
    def _gather_scene_settings(self):
        return {
            "cube_scale": float(self.axes_scale),
            "model_scale": float(self.model_scale),
            "external": {
                "scale": float(self.scale_spin.value()),
                "tx": float(self.tx_spin.value()),
                "ty": float(self.ty_spin.value()),
                "tz": float(self.tz_spin.value()),
                "roll": float(self.roll_spin.value()),
                "pitch": float(self.pitch_spin.value()),
                "yaw": float(self.yaw_spin.value())
            },
            "external_obj_path": self.last_obj_path or "",
            "bno_origin": {
                "enabled": bool(self.bno_origin_enabled),
                "rpy_deg": [float(self.bno_origin_rpy[0]), float(self.bno_origin_rpy[1]),
                            float(self.bno_origin_rpy[2])],
                "xyz": [float(self.bno_origin_xyz[0]), float(self.bno_origin_xyz[1]), float(self.bno_origin_xyz[2])]
            },
            "eye_vectors": [
                {
                    "enabled": bool(self.eye_enabled_chk[0].isChecked()),
                    "x": float(self.eye_pos_spin[0][0].value()),
                    "y": float(self.eye_pos_spin[0][1].value()),
                    "z": float(self.eye_pos_spin[0][2].value()),
                    "yaw": float(self.eye_yaw_spin[0].value()),
                    "pitch": float(self.eye_pitch_spin[0].value()),
                    "length": float(self.eye_len_spin[0].value()),
                },
                {
                    "enabled": bool(self.eye_enabled_chk[1].isChecked()),
                    "x": float(self.eye_pos_spin[1][0].value()),
                    "y": float(self.eye_pos_spin[1][1].value()),
                    "z": float(self.eye_pos_spin[1][2].value()),
                    "yaw": float(self.eye_yaw_spin[1].value()),
                    "pitch": float(self.eye_pitch_spin[1].value()),
                    "length": float(self.eye_len_spin[1].value()),
                },
            ],
            # _gather_scene_settings
            "camera": {
                "elev": float(self.cam_elev_spin.value()),
                "azim": float(self.cam_azim_spin.value()),
                "dist": float(self.cam_dist_spin.value()),
                "center": [float(self.cam_cx_spin.value()),
                           float(self.cam_cy_spin.value()),
                           float(self.cam_cz_spin.value())]
            },
        }

    def _apply_scene_settings(self, data: dict):
        try:
            # optional OBJ autoload
            obj_path = data.get("external_obj_path") or ""
            if obj_path and os.path.exists(obj_path):
                self._load_external_obj(obj_path)

            # cube/model scales
            cube_scale = float(data.get("cube_scale", 1.0))
            model_scale = float(data.get("model_scale", 1.0))
            self.cube_scale_spin.setValue(cube_scale)
            self.model_scale_spin.setValue(model_scale)

            # external transform
            ext = data.get("external", {})
            self.scale_spin.setValue(float(ext.get("scale", 1.0)))
            self.tx_spin.setValue(float(ext.get("tx", 0.0)))
            self.ty_spin.setValue(float(ext.get("ty", 0.0)))
            self.tz_spin.setValue(float(ext.get("tz", 0.0)))
            self.roll_spin.setValue(float(ext.get("roll", 0.0)))
            self.pitch_spin.setValue(float(ext.get("pitch", 0.0)))
            self.yaw_spin.setValue(float(ext.get("yaw", 0.0)))
            self._apply_external_transform()

            # BNO origin (optional, independent of external object)
            bno = data.get("bno_origin")
            if isinstance(bno, dict):
                self.bno_origin_enabled = bool(bno.get("enabled", True))
                rpy = bno.get("rpy_deg")
                if isinstance(rpy, (list, tuple)) and len(rpy) == 3:
                    self.bno_origin_rpy[:] = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
                xyz = bno.get("xyz")
                if isinstance(xyz, (list, tuple)) and len(xyz) == 3:
                    self.bno_origin_xyz[:] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                self._sync_bno_ui_from_state()
                self._apply_bno_origin_transform()

            # eye vectors
            ev = data.get("eye_vectors", None)
            if isinstance(ev, list) and len(ev) >= 2:
                for idx in (0, 1):
                    item = ev[idx] if isinstance(ev[idx], dict) else {}
                    self.eye_pos_spin[idx][0].setValue(float(item.get("x", self.eye_params[idx]["x"])))
                    self.eye_pos_spin[idx][1].setValue(float(item.get("y", self.eye_params[idx]["y"])))
                    self.eye_pos_spin[idx][2].setValue(float(item.get("z", self.eye_params[idx]["z"])))
                    self.eye_yaw_spin[idx].setValue(float(item.get("yaw", self.eye_params[idx]["yaw"])))
                    self.eye_pitch_spin[idx].setValue(float(item.get("pitch", self.eye_params[idx]["pitch"])))
                    self.eye_len_spin[idx].setValue(float(item.get("length", self.eye_params[idx]["length"])))
                    self._apply_eye_from_ui(idx)

                    enabled = bool(item.get("enabled", False))
                    self.eye_enabled_chk[idx].setChecked(enabled)
                    self._set_eye_visible(idx, enabled)

            # _apply_scene_settings
            cam = data.get("camera")
            if isinstance(cam, dict):
                self.cam_elev_spin.setValue(float(cam.get("elev", self.CAM_ELEV)))
                self.cam_azim_spin.setValue(float(cam.get("azim", self.CAM_AZIM)))
                self.cam_dist_spin.setValue(float(cam.get("dist", self.CAM_DIST)))
                ctr = cam.get("center", list(self.CAM_CENTER))
                if isinstance(ctr, (list, tuple)) and len(ctr) == 3:
                    self.cam_cx_spin.setValue(float(ctr[0]))
                    self.cam_cy_spin.setValue(float(ctr[1]))
                    self.cam_cz_spin.setValue(float(ctr[2]))
                self._apply_camera_from_ui()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load settings", f"Failed to apply settings:\n{e}")

    def _save_scene_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Scene Settings", "scene_settings.json",
                                                        "JSON (*.json);;Text (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._gather_scene_settings(), f, indent=2)
            self.statusBar().showMessage(f"Saved settings: {os.path.basename(path)}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save settings", str(e))

    def _load_scene_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Scene Settings", "",
                                                        "JSON (*.json);;Text (*.txt);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_scene_settings(data)
            self.statusBar().showMessage(f"Loaded settings: {os.path.basename(path)}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load settings", str(e))

    # ----- Export actions -----
    def _save_png(self):
        if not HAVE_IMAGEIO:
            QtWidgets.QMessageBox.critical(self, "Missing dependency",
                                           "imageio is not installed.\n\nconda install -c conda-forge imageio")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "frame.png", "PNG Image (*.png)")
        if not path: return
        if not path.lower().endswith(".png"): path += ".png"
        pr = getattr(self.canvas, "pixel_ratio", 1.0)
        w_log, h_log = self.canvas.size
        img = self.canvas.render(size=(int(w_log * pr), int(h_log * pr)))[:, :, :3]
        imageio.imwrite(path, img)

    def _export_mp4(self):
        if not HAVE_IMAGEIO:
            QtWidgets.QMessageBox.critical(self, "Missing dependency",
                                           "imageio/imageio-ffmpeg is not installed.\n\nconda install -c conda-forge imageio imageio-ffmpeg ffmpeg")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MP4", "reconstruction.mp4", "MP4 Video (*.mp4)")
        if not path: return
        if not path.lower().endswith(".mp4"): path += ".mp4"
        self._do_export_simple(path, start=0, end=self.n - 1, step=1, fps=30)

    def _export_range(self):
        if not HAVE_IMAGEIO:
            QtWidgets.QMessageBox.critical(self, "Missing dependency",
                                           "imageio/imageio-ffmpeg is not installed.\n\nconda install -c conda-forge imageio imageio-ffmpeg ffmpeg")
            return
        dlg = ExportRangeDialog(self, self.n, self.i)
        dlg.start.setValue(self.mark_in);
        dlg.end.setValue(self.mark_out)
        if dlg.exec_() != QtWidgets.QDialog.Accepted: return
        start, end, step, fps = dlg.values()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MP4", "reconstruction.mp4", "MP4 Video (*.mp4)")
        if not path: return
        if not path.lower().endswith(".mp4"): path += ".mp4"
        self._do_export_simple(path, start, end, step, fps)

    def _cancel_export(self):
        self._cancel_flag = True

    def _blend_hud_onto_frame(self, frame_rgb: np.ndarray, idx: int) -> np.ndarray:
        h, w, _ = frame_rgb.shape
        qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB888)
        ptr = qimg.bits();
        ptr.setsize(h * w * 3);
        ptr[:] = frame_rgb.tobytes()
        painter = QtGui.QPainter(qimg)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        font = QtGui.QFont("Menlo, Courier New, monospace");
        font.setPointSize(28);
        painter.setFont(font)
        text = self._compose_hud_text(idx)
        if text:
            lines = text.split("\n");
            metrics = QtGui.QFontMetrics(font)
            pad_x, pad_y = 8, 6;
            line_h = metrics.height()
            box_w = max(metrics.horizontalAdvance(s) for s in lines) + 2 * pad_x
            box_h = line_h * len(lines) + 2 * pad_y
            rect = QtCore.QRect(10, 10, box_w, box_h)
            painter.fillRect(rect, QtGui.QColor(0, 0, 0, 140))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            x = rect.left() + pad_x;
            y = rect.top() + pad_y + metrics.ascent()
            for s in lines: painter.drawText(x, y, s); y += line_h
        painter.end()
        out = np.empty_like(frame_rgb);
        ptr = qimg.bits();
        ptr.setsize(h * w * 3)
        out[:] = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3)
        return out

    def _do_export_simple(self, path, start, end, step, fps):
        if self.timer.isActive(): self._toggle_play()
        cam = self.view.camera;
        restore_interactive = cam.interactive;
        cam.interactive = False

        rgba0 = self.canvas.render();
        H, W = int(rgba0.shape[0]), int(rgba0.shape[1])

        total = ((end - start) // step) + 1
        self._progress = QtWidgets.QProgressDialog("Exporting MP4…", "Cancel", 0, total, self)
        self._progress.setWindowTitle("Export");
        self._progress.setWindowModality(QtCore.Qt.WindowModal)
        self._progress.setAutoClose(True);
        self._progress.setAutoReset(True);
        self._progress.setMinimumDuration(0)
        self._progress.setValue(0)

        self._exporting = True;
        self._cancel_flag = False;
        self.act_cancel_export.setEnabled(True)
        include_hud = self.act_include_hud.isChecked()

        try:
            writer = imageio.get_writer(path, fps=fps, format="FFMPEG", codec="libx264", pixelformat="yuv420p")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not open writer:\n{e}")
            self._exporting = False;
            self._cancel_flag = False;
            self.act_cancel_export.setEnabled(False)
            cam.interactive = restore_interactive
            if self._progress: self._progress.close(); self._progress = None
            return

        error = None;
        written = 0;
        t0 = time.time()
        try:
            with writer:
                for k, idx in enumerate(range(start, end + 1, step), start=1):
                    if self._cancel_flag or (self._progress and self._progress.wasCanceled()):
                        raise KeyboardInterrupt
                    if self.neutral_pose:
                        self.root.transform.matrix = np.eye(4, dtype=np.float32)
                    else:
                        self.root.transform.matrix = self.transforms[idx]

                    frame = self.canvas.render(size=(W, H))[:, :, :3]
                    if include_hud: frame = self._blend_hud_onto_frame(frame, idx)
                    writer.append_data(frame);
                    written += 1
                    elapsed = time.time() - t0
                    ratio = written / float(total)
                    eta = (elapsed / ratio) - elapsed if ratio > 0 else 0.0
                    if self._progress:
                        self._progress.setValue(written)
                        self._progress.setLabelText(
                            f"Exporting MP4…  {written}/{total}  •  elapsed {self._fmt_hms(elapsed)}  •  ETA {self._fmt_hms(eta)}"
                        )
                        QtWidgets.QApplication.processEvents()
        except KeyboardInterrupt:
            error = "cancelled"
        except Exception as e:
            error = str(e)
        finally:
            self._exporting = False;
            self._cancel_flag = False;
            self.act_cancel_export.setEnabled(False)
            cam.interactive = restore_interactive
            if self._progress: self._progress.close(); self._progress = None

        if error == "cancelled":
            try:
                if os.path.exists(path) and written < total: os.remove(path)
            except Exception:
                pass
            QtWidgets.QMessageBox.information(self, "Export cancelled", "The export was cancelled.")
        elif error:
            QtWidgets.QMessageBox.critical(self, "Export failed", error)
        else:
            QtWidgets.QMessageBox.information(self, "Export complete", f"Saved:\n{path}")

    def _sync_bno_ui_from_state(self):
        self.chk_bno_enable.setChecked(bool(self.bno_origin_enabled))

        # block all
        for w in (
                self.bno_tx, self.bno_ty, self.bno_tz,
                self.bno_roll, self.bno_pitch, self.bno_yaw,
                self.bno_tx_slider, self.bno_ty_slider, self.bno_tz_slider,
                self.bno_roll_slider, self.bno_pitch_slider, self.bno_yaw_slider
        ):
            w.blockSignals(True)

        # set spins
        self.bno_tx.setValue(float(self.bno_origin_xyz[0]))
        self.bno_ty.setValue(float(self.bno_origin_xyz[1]))
        self.bno_tz.setValue(float(self.bno_origin_xyz[2]))
        self.bno_roll.setValue(float(self.bno_origin_rpy[0]))
        self.bno_pitch.setValue(float(self.bno_origin_rpy[1]))
        self.bno_yaw.setValue(float(self.bno_origin_rpy[2]))

        # set sliders using the same mapping used in bind_pair
        self.bno_tx_slider.setValue(int(round(self.bno_origin_xyz[0] * 100)))
        self.bno_ty_slider.setValue(int(round(self.bno_origin_xyz[1] * 100)))
        self.bno_tz_slider.setValue(int(round(self.bno_origin_xyz[2] * 100)))
        self.bno_roll_slider.setValue(int(round(self.bno_origin_rpy[0])))
        self.bno_pitch_slider.setValue(int(round(self.bno_origin_rpy[1])))
        self.bno_yaw_slider.setValue(int(round(self.bno_origin_rpy[2])))

        # unblock all
        for w in (
                self.bno_tx, self.bno_ty, self.bno_tz,
                self.bno_roll, self.bno_pitch, self.bno_yaw,
                self.bno_tx_slider, self.bno_ty_slider, self.bno_tz_slider,
                self.bno_roll_slider, self.bno_pitch_slider, self.bno_yaw_slider
        ):
            w.blockSignals(False)

    def _on_bno_origin_changed(self):
        self.bno_origin_xyz[:] = [self.bno_tx.value(), self.bno_ty.value(), self.bno_tz.value()]
        self.bno_origin_rpy[:] = [self.bno_roll.value(), self.bno_pitch.value(), self.bno_yaw.value()]
        self._apply_bno_origin_transform()

    def _bno_set_from_current(self):
        """Make current frame the new zero (roll0=pitch0=yaw0)."""
        if self.n == 0: return
        i = int(self.i)
        self.bno_origin_rpy[:] = [float(self.roll[i]), float(self.pitch[i]), float(self.yaw[i])]
        self._sync_bno_ui_from_state()
        self._apply_bno_origin_transform()

    def _bno_reset(self):
        self.bno_origin_xyz[:] = [0.0, 0.0, 0.0]
        self.bno_origin_rpy[:] = [0.0, 0.0, 0.0]
        self._sync_bno_ui_from_state()
        self._apply_bno_origin_transform()

    def _disable_wheel_on_controls(self):
        self._no_wheel = NoWheelFilter(self)

        widgets = []

        # View panel sliders/spins
        widgets += [
            self.cube_scale_slider, self.cube_scale_spin,
            self.model_scale_slider, self.model_scale_spin,
            self.cam_elev_slider, self.cam_elev_spin,
            self.cam_azim_slider, self.cam_azim_spin,
            self.cam_dist_slider, self.cam_dist_spin,
            self.cam_cx_slider, self.cam_cx_spin,
            self.cam_cy_slider, self.cam_cy_spin,
            self.cam_cz_slider, self.cam_cz_spin,
            self.bno_tx_slider, self.bno_tx,
            self.bno_ty_slider, self.bno_ty,
            self.bno_tz_slider, self.bno_tz,
            self.bno_roll_slider, self.bno_roll,
            self.bno_pitch_slider, self.bno_pitch,
            self.bno_yaw_slider, self.bno_yaw,
        ]

        # OBJ transform panel sliders/spins
        widgets += [
            self.scale_slider, self.scale_spin,
            self.tx_slider, self.tx_spin,
            self.ty_slider, self.ty_spin,
            self.tz_slider, self.tz_spin,
            self.roll_slider, self.roll_spin,
            self.pitch_slider, self.pitch_spin,
            self.yaw_slider, self.yaw_spin,
        ]

        # Eye vectors panel sliders/spins
        # (These are created in _build_eye_vectors_ui)
        if hasattr(self, "eye_pos_slider"):
            for idx in (0, 1):
                # Position (3 sliders + 3 spins)
                for j in (0, 1, 2):
                    widgets.append(self.eye_pos_slider[idx][j])
                    widgets.append(self.eye_pos_spin[idx][j])

                # Direction (yaw/pitch sliders + spins)
                widgets.append(self.eye_yaw_slider[idx])
                widgets.append(self.eye_yaw_spin[idx])
                widgets.append(self.eye_pitch_slider[idx])
                widgets.append(self.eye_pitch_spin[idx])

                # Length (slider + spin)
                widgets.append(self.eye_len_slider[idx])
                widgets.append(self.eye_len_spin[idx])

        for w in widgets:
            try:
                w.installEventFilter(self._no_wheel)
            except Exception:
                pass


# ===================== Main =====================
if __name__ == "__main__":
    appQt = QtWidgets.QApplication(sys.argv)

    win = Player(None)  # launch blank
    sys.exit(appQt.exec())
