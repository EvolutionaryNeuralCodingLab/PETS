"""Trace-return reversals, noise snippets, nictitating peri-traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.behavior_state import (
    label_by_time,
    read_behavior_state,
)
from eye_tracking_system_tools.analysis.event_cache import reload_traces
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    CATEGORY_PUPIL_PERIMETER,
    read_noise_epochs,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

RETURN_MS = 80.0
RETURN_FRAC = 0.5
NOISE_SNIPPET_S = 2.0
NICTITATING_HALF_MS = 150.0
DLC_LIKELIHOOD_THR = 0.5
DLC_MIN_FRAMES = 3


def _interp_angles(df: pd.DataFrame, t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if df is None or df.empty or "ms_axis" not in df.columns:
        return np.full(t_ms.shape, np.nan), np.full(t_ms.shape, np.nan)
    t = pd.to_numeric(df["ms_axis"], errors="coerce").to_numpy()
    phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy() if "k_phi" in df.columns else np.full(t.shape, np.nan)
    th = pd.to_numeric(df["k_theta"], errors="coerce").to_numpy() if "k_theta" in df.columns else np.full(t.shape, np.nan)
    order = np.argsort(t)
    t, phi, th = t[order], phi[order], th[order]
    return (
        np.interp(t_ms, t, phi, left=np.nan, right=np.nan),
        np.interp(t_ms, t, th, left=np.nan, right=np.nan),
    )


def event_returns_toward_start(
    row: pd.Series,
    eye_df: pd.DataFrame,
    *,
    return_ms: float = RETURN_MS,
    frac: float = RETURN_FRAC,
) -> bool:
    on = float(row["saccade_on_ms"])
    off = float(row["saccade_off_ms"])
    phi0, th0 = _interp_angles(eye_df, np.array([on]))
    phi1, th1 = _interp_angles(eye_df, np.array([off]))
    net = float(np.hypot(phi1[0] - phi0[0], th1[0] - th0[0]))
    if not np.isfinite(net) or net <= 0:
        return False
    if "ms_axis" not in eye_df.columns:
        return False
    t = pd.to_numeric(eye_df["ms_axis"], errors="coerce").to_numpy()
    win = (t > off) & (t <= off + return_ms)
    if not np.any(win):
        return False
    phi, th = _interp_angles(eye_df, t[win])
    dist = np.hypot(phi - phi0[0], th - th0[0])
    return bool(np.nanmin(dist) <= (1.0 - frac) * net)


def reversal_rates_by_state(
    tables: EventTables,
    *,
    return_ms: float = RETURN_MS,
    frac: float = RETURN_FRAC,
) -> pd.DataFrame:
    counts = {"active": [0, 0], "quiet": [0, 0]}
    examples: list[pd.Series] = []
    for bundle in tables.blocks:
        state = read_behavior_state(bundle.spec)
        for eye, ev, df in (
            ("L", bundle.l_saccades, bundle.left),
            ("R", bundle.r_saccades, bundle.right),
        ):
            if ev is None or ev.empty or df is None or df.empty:
                continue
            for _, row in ev.iterrows():
                lab = str(label_by_time(state, float(row["saccade_on_ms"])))
                if lab not in counts:
                    continue
                counts[lab][1] += 1
                if event_returns_toward_start(row, df, return_ms=return_ms, frac=frac):
                    counts[lab][0] += 1
                    if len(examples) < 6:
                        rec = row.copy()
                        rec["state"] = lab
                        rec["eye"] = eye
                        rec["block_key"] = bundle.spec.block_key
                        examples.append(rec)
    rows = []
    for lab, (n_ret, n_all) in counts.items():
        rows.append(
            {
                "state": lab,
                "n_reversal": n_ret,
                "n_events": n_all,
                "rate": (n_ret / n_all) if n_all else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["examples"] = examples
    return out


def _find_saccade_free_window(
    df: pd.DataFrame,
    onsets_ms: np.ndarray,
    state_df: pd.DataFrame | None,
    want_label: str,
    *,
    duration_s: float = NOISE_SNIPPET_S,
) -> tuple[float, float] | None:
    if df is None or df.empty or "ms_axis" not in df.columns:
        return None
    t = pd.to_numeric(df["ms_axis"], errors="coerce").dropna().to_numpy()
    if t.size < 10:
        return None
    dur_ms = duration_s * 1000.0
    t0, t1 = float(t.min()), float(t.max())
    step = dur_ms / 2.0
    start = t0
    while start + dur_ms <= t1:
        end = start + dur_ms
        mid = 0.5 * (start + end)
        lab = str(label_by_time(state_df, mid)) if state_df is not None else want_label
        if lab == want_label:
            hit = np.any((onsets_ms >= start) & (onsets_ms <= end)) if onsets_ms.size else False
            if not hit:
                return start / 1000.0, end / 1000.0
        start += step
    return None


def _runs_below(values: np.ndarray, *, thr: float, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(values.size)
    while i < n:
        if np.isfinite(values[i]) and values[i] < thr:
            j = i
            while j < n and np.isfinite(values[j]) and values[j] < thr:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _dlc_likelihood_drop_midpoints_ms(bundle, t_ms: np.ndarray, frame_ms: float) -> list[float]:
    """Midpoints of ≥3-frame pupil-likelihood drops below 0.5 (DLC fallback)."""
    try:
        from eye_tracking_system_tools.analysis.data_yield import resolve_block_dlc_csvs
        from eye_tracking_system_tools.preprocessing.dlc_csv_io import _likelihood_columns
    except Exception:
        return []
    paths = resolve_block_dlc_csvs(bundle.spec.block_path)
    csv_path = paths.get("left")
    if csv_path is None or not Path(csv_path).is_file():
        return []
    data = pd.read_csv(csv_path, header=1, low_memory=False)
    data = data.iloc[1:].apply(pd.to_numeric, errors="coerce")
    lik_cols = [c for c in _likelihood_columns(data) if "Pupil" in str(c)]
    if not lik_cols:
        lik_cols = _likelihood_columns(data)
    if not lik_cols:
        return []
    lik = data[lik_cols].mean(axis=1).to_numpy(dtype=float)
    mids: list[float] = []
    t0 = float(np.nanmin(t_ms)) if t_ms.size else 0.0
    for a, b in _runs_below(lik, thr=DLC_LIKELIHOOD_THR, min_len=DLC_MIN_FRAMES):
        mid_frame = 0.5 * (a + b)
        mids.append(t0 + mid_frame * frame_ms)
    return mids


def _nictitating_midpoints_ms(bundle, t_ms: np.ndarray, frame_ms: float) -> list[float]:
    """Prefer pupil_perimeter epochs; else DLC pupil-likelihood drops."""
    epochs = read_noise_epochs(bundle.spec.block_path, "left")
    mids: list[float] = []
    if epochs is not None and not epochs.empty:
        peri = epochs[epochs["category"] == CATEGORY_PUPIL_PERIMETER]
        t0 = float(np.nanmin(t_ms)) if t_ms.size else np.nan
        for _, ep in peri.iterrows():
            mid_frame = 0.5 * (float(ep["start_frame"]) + float(ep["end_frame"]))
            if np.isfinite(t0):
                mids.append(t0 + mid_frame * frame_ms)
    if mids:
        return mids
    return _dlc_likelihood_drop_midpoints_ms(bundle, t_ms, frame_ms)


def export_qc_reversals_noise_blinks(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
) -> dict[str, Path]:
    plot_bundle = begin_plot_bundle(
        out_dir,
        "qc_reversals_noise_blinks",
        kind="qc_reversals_noise_blinks",
        tables=tables,
        logic_key="qc_reversals_noise_blinks",
        params=dict(getattr(tables, "params", {}) or {}),
    )
    figures_dir, metadata_dir = plot_bundle.plots_dir, plot_bundle.metadata_dir
    written: dict[str, Path] = {}
    tables = reload_traces(tables)

    rates = reversal_rates_by_state(tables)
    rates.to_csv(metadata_dir / "reversal_rates.csv", index=False)
    fig, ax = plt.subplots(figsize=(2.0, 1.8), dpi=300)
    ax.bar(rates["state"], rates["rate"].fillna(0), color=["#D55E00", "#0072B2"][: len(rates)])
    for i, row in rates.iterrows():
        ax.text(i, 0.02, f"n={int(row.n_events)}", ha="center", fontsize=6)
    ax.set_ylabel("Return rate", fontsize=8)
    ax.set_ylim(0, max(0.2, float(np.nanmax(rates["rate"])) * 1.3 if rates["rate"].notna().any() else 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    p = figures_dir / "reversal_rate_by_state.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    examples = list(rates.attrs.get("examples") or [])
    if examples:
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
        for rec in examples:
            block = tables.block_dict.get(str(rec.get("block_key", "")))
            if block is None:
                continue
            df = block.left if rec.get("eye") == "L" else block.right
            off = float(rec["saccade_off_ms"])
            t = pd.to_numeric(df["ms_axis"], errors="coerce").to_numpy()
            phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy()
            m = (t >= off - 40) & (t <= off + RETURN_MS)
            if np.any(m):
                ax.plot((t[m] - off), phi[m] - np.nanmedian(phi[m]), lw=0.7, alpha=0.8)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Time from offset [ms]", fontsize=8)
        ax.set_ylabel("Δφ [deg]", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        p = figures_dir / "reversal_example_traces.pdf"
        fig.tight_layout()
        fig.savefig(p, format="pdf", bbox_inches="tight")
        show_and_close(fig, show)
        written[p.name] = p

    dphi_by_state: dict[str, list[float]] = {"active": [], "quiet": []}
    have_snippet = {"quiet": False, "active": False}
    for bundle in tables.blocks:
        state = read_behavior_state(bundle.spec)
        onsets = np.array([], dtype=float)
        for ev in (bundle.l_saccades, bundle.r_saccades):
            if ev is not None and not ev.empty:
                onsets = np.concatenate([onsets, ev["saccade_on_ms"].to_numpy(float)])
        for label in ("quiet", "active"):
            if have_snippet[label]:
                continue
            win = _find_saccade_free_window(bundle.left, onsets, state, label)
            if win is None:
                continue
            start_s, end_s = win
            t = bundle.left["ms_axis"].to_numpy(float) / 1000.0
            m = (t >= start_s) & (t <= end_s)
            fig, axes = plt.subplots(2, 1, figsize=(2.4, 1.8), dpi=300, sharex=True)
            t_rel = t[m] - start_s
            axes[0].plot(t_rel, bundle.left["k_phi"].to_numpy(float)[m], lw=0.7)
            axes[1].plot(t_rel, bundle.left["k_theta"].to_numpy(float)[m], lw=0.7)
            axes[0].set_ylabel("φ [deg]", fontsize=8)
            axes[1].set_ylabel("θ [deg]", fontsize=8)
            axes[1].set_xlabel("[s]", fontsize=8)
            for ax in axes:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            p = figures_dir / f"noise_trace_{label}.pdf"
            fig.tight_layout()
            fig.savefig(p, format="pdf", bbox_inches="tight")
            show_and_close(fig, show)
            written[p.name] = p
            have_snippet[label] = True

        if bundle.left is not None and not bundle.left.empty and "k_phi" in bundle.left.columns:
            phi = pd.to_numeric(bundle.left["k_phi"], errors="coerce").to_numpy()
            tms = pd.to_numeric(bundle.left["ms_axis"], errors="coerce").to_numpy()
            dphi = np.diff(phi)
            mid = 0.5 * (tms[1:] + tms[:-1])
            labs = np.asarray(label_by_time(state, mid), dtype=object)
            for lab in ("active", "quiet"):
                vals = dphi[labs == lab]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    dphi_by_state[lab].extend(vals.tolist())

    fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
    pooled = []
    for lab in ("quiet", "active"):
        pooled.extend(dphi_by_state[lab])
    pooled_arr = np.asarray(pooled, dtype=float)
    pooled_arr = pooled_arr[np.isfinite(pooled_arr)]
    hist_range = None
    if pooled_arr.size:
        lo, hi = np.nanpercentile(pooled_arr, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(pooled_arr)), float(np.nanmax(pooled_arr))
        if lo >= hi:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.05 * (hi - lo)
        hist_range = (lo - pad, hi + pad)
    for lab, color in (("quiet", "#0072B2"), ("active", "#D55E00")):
        vals = np.asarray(dphi_by_state[lab], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            kw = {"bins": 40, "density": True, "histtype": "step", "color": color, "label": lab}
            if hist_range is not None:
                kw["range"] = hist_range
            ax.hist(vals, **kw)
    ax.set_xlabel("Δφ / frame [deg]", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    p = figures_dir / "noise_dframe_hist.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    peri: list[np.ndarray] = []
    skipped = 0
    for bundle in tables.blocks:
        if bundle.left is None or bundle.left.empty:
            skipped += 1
            continue
        t = pd.to_numeric(bundle.left["ms_axis"], errors="coerce").to_numpy()
        phi = pd.to_numeric(bundle.left["k_phi"], errors="coerce").to_numpy()
        frame_ms = float(np.nanmedian(np.diff(t))) if t.size > 3 else 16.67
        mid_ms_list = _nictitating_midpoints_ms(bundle, t, frame_ms)
        if not mid_ms_list:
            skipped += 1
            continue
        for mid_ms in mid_ms_list:
            if not np.isfinite(mid_ms):
                continue
            rel = t - mid_ms
            m = (rel >= -NICTITATING_HALF_MS) & (rel <= NICTITATING_HALF_MS)
            if np.count_nonzero(m) < 5:
                continue
            grid = np.linspace(-NICTITATING_HALF_MS, NICTITATING_HALF_MS, 31)
            peri.append(np.interp(grid, rel[m], phi[m] - np.nanmedian(phi[m])))
    if peri:
        arr = np.vstack(peri)
        grid = np.linspace(-NICTITATING_HALF_MS, NICTITATING_HALF_MS, 31)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
        ax.plot(grid, mean, color="#0072B2", lw=1.2)
        ax.fill_between(grid, mean - sem, mean + sem, color="#0072B2", alpha=0.25)
        ax.set_xlabel("Time from midpoint [ms]", fontsize=8)
        ax.set_ylabel("Δφ [deg]", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        p = figures_dir / "nictitating_peri_trace.pdf"
        fig.tight_layout()
        fig.savefig(p, format="pdf", bbox_inches="tight")
        show_and_close(fig, show)
        written[p.name] = p
    else:
        (metadata_dir / "nictitating_skip.txt").write_text(
            f"No pupil_perimeter epochs in {skipped} blocks.\n", encoding="utf-8"
        )
    finish_plot_bundle(plot_bundle)
    return written
