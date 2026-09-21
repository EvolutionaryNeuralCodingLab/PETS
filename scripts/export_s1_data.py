#!/usr/bin/env python3
"""Build the PLOS numerical source-data workbook S1_Data.xlsx.

Reads frozen reproduction pickles/CSVs plus recovered S2/S4/S5 tables.
Does not recompute scientific analyses. Sheet names follow manuscript panel
letters (submitted PDFs), which differ from some repo folder letters.

Usage (from repo root, eye_repo_mac env):
    PYTHONPATH=.venv_pkgs python scripts/export_s1_data.py
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "src/eye_tracking_system_tools/figures/reproduction"
MAIN = REPRO / "main_figures"
SUPP = REPRO / "supplementary"
RECOVERED = ROOT / "publication/recovered"
DESKTOP_OUT = Path("/Users/nimi/Desktop/PLOS_tech_rev/S1_Data.xlsx")
REPO_OUT = ROOT / "publication/S1_Data.xlsx"

EXCEL_MAX_ROWS = 1_048_576
PATH_RE = re.compile(r"(/Volumes/[^\s,]+|/Users/[^\s,]+|[A-Za-z]:\\[^\s,]+|Z:\\[^\s,]+)")

VALIDATION: list[str] = []


def pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sanitize(val):
    if isinstance(val, str):
        return PATH_RE.sub("<lab_path>", val)
    return val


def note_fmt(wb):
    return wb.add_format({"italic": True, "text_wrap": True, "valign": "top", "font_size": 9})


def head_fmt(wb):
    return wb.add_format({"bold": True, "font_size": 10, "bottom": 1})


def write_notes(ws, lines, fmt, width=110):
    ws.set_column(0, 0, min(width, 60))
    for i, line in enumerate(lines):
        ws.write(i, 0, line, fmt)
        ws.set_row(i, 16)
    return len(lines) + 1


def write_df(ws, df: pd.DataFrame, start: int, hf, max_rows: int | None = None) -> int:
    if df is None or df.empty:
        ws.write(start, 0, "(empty table)")
        return start + 2
    cols = [str(c) for c in df.columns]
    for j, c in enumerate(cols):
        ws.write(start, j, c, hf)
        ws.set_column(j, j, max(12, min(28, len(c) + 2)))
    n = len(df) if max_rows is None else min(len(df), max_rows)
    values = df.iloc[:n].to_numpy()
    for i in range(n):
        row = values[i]
        for j, v in enumerate(row):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            if isinstance(v, (np.floating, float)):
                ws.write_number(start + 1 + i, j, float(v))
            elif isinstance(v, (np.integer, int)):
                ws.write_number(start + 1 + i, j, int(v))
            elif isinstance(v, (np.bool_, bool)):
                ws.write_boolean(start + 1 + i, j, bool(v))
            else:
                ws.write(start + 1 + i, j, sanitize(str(v)))
    return start + 1 + n + 2


def write_long_array(ws, header_row: int, headers: list[str], columns: list[np.ndarray], hf):
    """Write several 1-D arrays as columns, possibly wrapping each into parts."""
    for j, h in enumerate(headers):
        ws.write(header_row, j, h, hf)
        ws.set_column(j, j, max(14, min(24, len(h) + 2)))
    n = max((len(c) for c in columns), default=0)
    for i in range(n):
        for j, col in enumerate(columns):
            if i < len(col):
                v = col[i]
                if isinstance(v, (np.floating, float)) and np.isfinite(v):
                    ws.write_number(header_row + 1 + i, j, float(v))
                elif isinstance(v, (np.integer, int)):
                    ws.write_number(header_row + 1 + i, j, int(v))
    return header_row + 1 + n + 2


def wrap_array_columns(name: str, arr: np.ndarray, max_data_rows: int) -> tuple[list[str], list[np.ndarray]]:
    arr = np.asarray(arr).ravel()
    if len(arr) <= max_data_rows:
        return [name], [arr]
    headers, parts = [], []
    n_parts = int(np.ceil(len(arr) / max_data_rows))
    for k in range(n_parts):
        headers.append(f"{name}_part{k + 1}")
        parts.append(arr[k * max_data_rows : (k + 1) * max_data_rows])
    return headers, parts


def add_sheet(wb, name: str):
    return wb.add_worksheet(name[:31])


def main() -> int:
    nf_book_notes = None
    out_paths = [REPO_OUT]
    if DESKTOP_OUT.parent.is_dir():
        out_paths.append(DESKTOP_OUT)
    REPO_OUT.parent.mkdir(parents=True, exist_ok=True)

    wb = xlsxwriter.Workbook(str(REPO_OUT), {"constant_memory": False, "nan_inf_to_errors": True})
    nf = note_fmt(wb)
    hf = head_fmt(wb)
    nf_book_notes = nf

    # ------------------------------------------------------------------ README
    ws = add_sheet(wb, "README")
    readme = [
        "S1 Data. Numerical values underlying the editor-requested figure panels.",
        "Manuscript: An open-source, adaptable eye-tracking system enables studies of visual acquisition across diverse terrestrial vertebrates.",
        "Sheet names use published panel letters (submitted PDFs / figure legends).",
        "",
        "PANEL LETTER MAP (manuscript PDF vs some repo folders):",
        "  Fig3D = pupil-diameter histograms (repo Fig_3_e). n=4 animals for this analysis (manuscript n=5 is a typo).",
        "  Fig3E = per-animal z-scored pupil difference (repo Fig_3_f). Animals: PV_106, PV_143, PV_62, PV_126.",
        "  Fig3F = ISI distributions (repo Fig_3_d).",
        "  FigS8E = lizard vs mouse interocular coupling (repo S8j), not a still.",
        "  FigS11A = ISI by behavioral state (repo S11 folder 'b' ISI pickles).",
        "  FigS11B = epoch durations (repo S11 'a' / epoch_duration_bin_trials.pkl; n_active=174, n_quiet=191).",
        "  FigS12A = example back-and-forth traces; FigS12B = event table (repo letters were swapped).",
        "",
        "ROW MEANING: unless a sheet says otherwise, one data row is one observation (frame, saccade, pair, sample, or animal).",
        "Means and error bars can be reconstructed from those rows where individual observations are provided.",
        "",
        "KNOWN LIMITATIONS (observations not frozen; do not treat summaries as hidden replicates):",
        "  Fig2C/Fig2D: per-saccade aligned traces were never serialized. Sheets contain the published mean traces + n per bin (PV_106; bin n sum=6934; legend n=6954 includes 20 events outside 0–25°).",
        "  Fig2E: published OLS uses n=34876 points that were not pickled. Sheet contains the plotted per-animal bin means + exact OLS, plus the closest recovered event table (n=34336; slope 0.02199 vs published 0.02115). Do not substitute Fig2J (n=28146) or FigS13 (n=33294, different amplitude definition).",
        "  Fig2G: top panel is mean±SEM of animal-wise PDFs; bottom panel stores the five animal difference curves (the plotted replicates). Per-saccade amplitudes that regenerate those means were not frozen.",
        "  FigS3: individual L/R pairs were not frozen; sheet stores the published 99×99 normalized counts (n_still=11841, n_moving=6930, PV_62 excluded).",
        "  Fig3F/FigS11A: raw ISI lists were not frozen; sheets store the per-animal density curves that were plotted.",
        "  Fig1E vs FigS9: different jitter populations. Fig1E n=454300, median ~31 µm. S9 rigid lizard n=1473783, median ~42 µm.",
        "  S10 SNR: manuscript/PDF use SNR = threshold / median(D). Repo pickle captions used threshold/(2B). Both are tabulated. Turtle pickle has no stored threshold; manuscript reports 2.0°/frame.",
        "",
        "FigS9 Excel row limit: rigid (~1.47e6) and modular lizard (~1.03e6) samples are split across labelled _part columns. Concatenate part1, part2, … in column order to rebuild the vector.",
        "",
        "Text-only statistics (timing 78.1 ms, 3.7% corrective, 10.2% active time, Monte Carlo tests) are not in this workbook; see publication/TEXT_ONLY_STATISTICS.md.",
        "Units: degrees (deg), deg/ms, deg/frame, micrometers (µm), millimeters (mm), seconds (s), milliseconds (ms).",
    ]
    write_notes(ws, readme, nf, width=140)
    ws.set_column(0, 0, 140)

    def sheet(name, notes, *tables):
        w = add_sheet(wb, name)
        r = write_notes(w, notes, nf)
        for item in tables:
            if item is None:
                continue
            if isinstance(item, tuple) and item[0] == "arrays":
                _, headers, cols = item
                r = write_long_array(w, r, headers, cols, hf)
            else:
                title, df = item
                w.write(r, 0, title, hf)
                r = write_df(w, df, r + 1, hf)
        return w

    # ------------------------------------------------------------------ Fig1E
    dist = np.asarray(pkl(MAIN / "Fig_1_e/distances.pkl"), dtype=float)
    VALIDATION.append(f"Fig1E n={dist.size} median_um={np.nanmedian(dist):.3f}")
    sheet(
        "Fig1E",
        [
            "Fig 1E. Camera jitter displacements used for the published histogram.",
            "One row = one frame-to-frame displacement. Units: micrometers (µm).",
            "No animal/mount labels were stored. This pool is NOT the S9 rigid-lizard sample (see README).",
            f"n={dist.size}; median={np.nanmedian(dist):.3f} µm; p95={np.nanpercentile(dist,95):.3f} µm.",
        ],
        ("arrays", ["displacement_um"], [dist]),
    )

    # ------------------------------------------------------------------ Fig2B
    b = pkl(MAIN / "Fig_2_b/figure_2b.pickle")
    ex = pkl(MAIN / "Fig_2_b/saccade_examples.pickle")
    t = np.asarray(b["left_eye_data"]["ms_axis"], float) / 1000.0
    fig2b = pd.DataFrame(
        {
            "time_s": t,
            "phi_L_deg": np.asarray(b["left_eye_data"]["k_phi"], float),
            "theta_L_deg": np.asarray(b["left_eye_data"]["k_theta"], float),
            "pupil_L_mm": np.asarray(b["left_eye_data"]["pupil_diameter"], float),
            "phi_R_deg": np.asarray(b["right_eye_data"]["k_phi"], float),
            "theta_R_deg": np.asarray(b["right_eye_data"]["k_theta"], float),
            "pupil_R_mm": np.asarray(b["right_eye_data"]["pupil_diameter"], float),
        }
    )
    ex_rows = []
    for item in ex.get("saccade_examples", []):
        le, reye = item["left_eye_data"], item["right_eye_data"]
        tt = np.asarray(le["ms_axis"], float)
        for i in range(len(tt)):
            ex_rows.append(
                {
                    "example_label": item.get("label"),
                    "start_time_s": item.get("start_time"),
                    "end_time_s": item.get("end_time"),
                    "time_ms": float(tt[i]),
                    "phi_L_deg": float(np.asarray(le["k_phi"])[i]),
                    "theta_L_deg": float(np.asarray(le["k_theta"])[i]),
                    "phi_R_deg": float(np.asarray(reye["k_phi"])[i]),
                    "theta_R_deg": float(np.asarray(reye["k_theta"])[i]),
                }
            )
    sheet(
        "Fig2B",
        [
            "Fig 2B. Example binocular traces (illustration, not the n=5 population).",
            "Table 1: simultaneous φ/θ/pupil for the long trace (time in seconds).",
            "Table 2: four example saccade trajectories (divergent, conjugated, disjunct, monocular).",
            f"Long-trace window {b.get('start_time')}–{b.get('end_time')} s. Angles in degrees; pupil in mm.",
        ],
        ("Table 1. Long example trace", fig2b),
        ("Table 2. Example saccade trajectories", pd.DataFrame(ex_rows)),
    )

    # ------------------------------------------------------------------ Fig2C / Fig2D
    bundle_c = pkl(MAIN / "Fig_2_c/pos_vel_by_amp_bins_bundle.pkl")
    t_grid = np.asarray(bundle_c["t_grid"], float)
    series = bundle_c["animals"]["PV_106"]["series"]
    rows_c, rows_d, bin_n = [], [], []
    for ser in series:
        n_ev = int(ser["n_events"])
        lab = ser.get("raw_label") or ser.get("label")
        bin_n.append({"animal": "PV_106", "amplitude_bin": lab, "n_events": n_ev})
        vel = np.asarray(ser["vel_center"], float)
        pos = np.asarray(ser["pos_center"], float)
        for i, tt in enumerate(t_grid):
            rows_c.append({"animal": "PV_106", "amplitude_bin": lab, "n_events": n_ev, "time_ms": float(tt), "mean_speed_deg_per_s": float(vel[i])})
            rows_d.append({"animal": "PV_106", "amplitude_bin": lab, "n_events": n_ev, "time_ms": float(tt), "mean_displacement_deg": float(pos[i])})
    VALIDATION.append(f"Fig2C bin_n_sum={sum(x['n_events'] for x in bin_n)}")
    common_2cd = [
        "One animal (PV_106). Amplitude = net_angular_disp, 5° bins, aligned as in the published panel.",
        "Individual aligned saccade waveforms were not frozen. These are the published mean traces.",
        "Legend n=6954; stored bin n sum=6934 (events outside the plotted 0–25° bins).",
        "Units: time ms; Fig2C speed deg/s (replot); Fig2D displacement deg along movement axis.",
    ]
    sheet("Fig2C", ["Fig 2C. Mean saccade angular-speed profiles."] + common_2cd, ("Mean speed traces", pd.DataFrame(rows_c)), ("Events per bin", pd.DataFrame(bin_n)))
    sheet("Fig2D", ["Fig 2D. Mean angular-displacement traces from the same events as Fig 2C."] + common_2cd, ("Mean displacement traces", pd.DataFrame(rows_d)), ("Events per bin", pd.DataFrame(bin_n)))

    # ------------------------------------------------------------------ Fig2E
    e = pkl(MAIN / "Fig_2_e/amplitude_velocity_linear_fit_bundle.pkl")
    gf = e["global_fit"]
    VALIDATION.append(f"Fig2E n={gf['n']} slope={gf['slope']:.6f} intercept={gf['intercept']:.6f} R2={gf['R2']:.4f}")
    bin_rows = []
    for animal, df in e["per_animal_stats"].items():
        tmp = df.copy()
        tmp.insert(0, "animal", animal)
        bin_rows.append(tmp)
    bins_df = pd.concat(bin_rows, ignore_index=True)
    lin_df = e["linear_stats_df"].copy()
    fit_df = pd.DataFrame(
        [
            {
                "n_events_published": int(gf["n"]),
                "slope_per_ms": float(gf["slope"]),
                "intercept_deg_per_ms": float(gf["intercept"]),
                "R2": float(gf["R2"]),
                "amplitude_definition": e["params"].get("amp_col"),
            }
        ]
    )
    proxy_path = Path("/tmp/plos_source_recovery/fig2e_PROXY_event_cache_filtered.csv")
    proxy = pd.read_csv(proxy_path) if proxy_path.is_file() else pd.DataFrame()
    if not proxy.empty:
        VALIDATION.append(f"Fig2E proxy n={len(proxy)}")
    sheet(
        "Fig2E",
        [
            "Fig 2E. Amplitude vs peak angular speed. Plotted objects are per-animal binned mean lines; OLS uses n=34876 saccades.",
            f"Published OLS: slope={gf['slope']:.6f} ms^-1, intercept={gf['intercept']:.6f} deg/ms, R2={gf['R2']:.4f}, n=34876.",
            "The 34876 individual (amplitude, peak_velocity) points were not pickled.",
            "Table 3 is the closest recovered event-cache table (n=34336). It does NOT reproduce the published OLS; do not treat it as the frozen 2E sample.",
            "Do not use Fig2J (n=28146) or FigS13 (n=33294, onset-to-offset amplitude) as substitutes.",
            "Units: amplitude deg; peak velocity deg/ms.",
        ],
        ("Table 1. Published global OLS", fit_df),
        ("Table 2. Per-animal bin means (the plotted lines) and per-animal OLS", pd.concat([bins_df, lin_df], axis=0, ignore_index=True) if False else bins_df),
        ("Table 2b. Per-animal OLS on the published fit", lin_df),
        ("Table 3. Closest recovered per-saccade events (NOT n=34876)", proxy),
    )

    # ------------------------------------------------------------------ Fig2F
    f2 = pkl(MAIN / "Fig_2_f/figure_2f_nodowncast.pickle")
    fig2f = pd.DataFrame(
        {
            "left_peak_speed_deg_per_ms": np.asarray(f2["left_eye_speeds"], float),
            "right_peak_speed_deg_per_ms": np.asarray(f2["right_eye_speeds"], float),
            "weight": np.asarray(f2["weights"], float),
        }
    )
    VALIDATION.append(f"Fig2F n_pairs={len(fig2f)}")
    sheet(
        "Fig2F",
        [
            "Fig 2F. Inter-ocular peak-speed pairs for monocular, head-stationary saccades (equal-animal weights).",
            "One row = one L/R peak-speed pair used to build the published 2D histogram. Units: deg/ms.",
            "Animal IDs were not stored in the frozen pickle. n from the forward-facing reproduction bundle.",
        ],
        ("L/R peak-speed pairs", fig2f),
    )

    # ------------------------------------------------------------------ Fig2G
    gtop = pkl(MAIN / "Fig_2_g/averaged_saccade_amplitude_angle_data.pkl")
    gbot = pkl(MAIN / "Fig_2_g/saccade_amplitude_difference_all_animals_data.pkl")
    bins = np.asarray(gtop["bins"], float)
    centers = 0.5 * (bins[:-1] + bins[1:])
    top_df = pd.DataFrame(
        {
            "amplitude_bin_center_deg": centers,
            "concurrent_mean_pdf": np.asarray(gtop["synced_mean"], float),
            "concurrent_sem_pdf": np.asarray(gtop["synced_sem"], float),
            "monocular_mean_pdf": np.asarray(gtop["non_synced_mean"], float),
            "monocular_sem_pdf": np.asarray(gtop["non_synced_sem"], float),
        }
    )
    bot_bins = np.asarray(gbot.get("bins", bins), float)
    bot_centers = 0.5 * (bot_bins[:-1] + bot_bins[1:])
    bot_rows = []
    for animal, trace in gbot["animal_diff_traces"].items():
        tr = np.asarray(trace, float)
        n = min(len(bot_centers), len(tr))
        for i in range(n):
            bot_rows.append({"animal": animal, "amplitude_bin_center_deg": float(bot_centers[i]), "concurrent_minus_monocular_pdf": float(tr[i])})
    sheet(
        "Fig2G",
        [
            "Fig 2G. Amplitude PDFs for concurrent vs monocular saccades.",
            "Table 1: plotted mean±SEM (across animals) of animal-wise PDFs. One row = one amplitude bin.",
            "Table 2: per-animal difference curves (concurrent − monocular) — the five biological replicates in the bottom panel.",
            "Per-saccade amplitudes that generate Table 1 were not frozen (Fig2J events do not regenerate these means).",
            "Units: amplitude deg; PDF values are probability densities (independent normalization as in the panel).",
        ],
        ("Table 1. Mean ± SEM PDFs", top_df),
        ("Table 2. Per-animal difference curves", pd.DataFrame(bot_rows)),
    )

    # ------------------------------------------------------------------ Fig2I
    i2 = pkl(MAIN / "Fig_2_i/figure_2i.pickle")
    ang_rows, hist_rows = [], []
    for animal, eyes in i2["per_eye_histograms"].items():
        for eye, h in eyes.items():
            ang = np.asarray(h["rotated_angles"], float)
            raw = np.asarray(h["angles"], float)
            n = min(len(ang), len(raw))
            for k in range(n):
                ang_rows.append({"animal": str(animal), "eye": str(eye), "saccade_index": k, "rotated_angle_deg": float(ang[k]), "raw_angle_deg": float(raw[k])})
            cts = np.asarray(h["rotated_counts"], float)
            ctr = np.asarray(h["rotated_centers"], float)
            for k, c in enumerate(ctr):
                hist_rows.append({"animal": str(animal), "eye": str(eye), "bin_center_deg": float(c), "density": float(cts[k]), "n_saccades": int(h["n_saccades"]), "rotation_deg": float(h["rotation_angle"])})
    sheet(
        "Fig2I",
        [
            "Fig 2I. Polar saccade-direction distributions, rotated onto each eye’s dominant axis.",
            "Table 1: one row = one saccade (individual observations). Table 2: plotted polar densities.",
            "PV_57 is not in this pickle (four animals). Units: degrees. Density as stored (matches replot).",
        ],
        ("Table 1. Individual saccade directions", pd.DataFrame(ang_rows)),
        ("Table 2. Polar densities", pd.DataFrame(hist_rows)),
    )

    # ------------------------------------------------------------------ Fig2J
    j2 = pkl(MAIN / "Fig_2_j/saccade_angles_data.pkl")
    keep = [
        "animal", "block", "eye", "saccade_on_ms", "saccade_off_ms", "net_angular_disp",
        "magnitude_raw_angular", "overall_angle_deg", "peak_velocity", "head_movement", "length",
    ]
    syn = j2["synced_df"][[c for c in keep if c in j2["synced_df"].columns]].copy()
    syn["pairing"] = "concurrent"
    nsyn = j2["non_synced_df"][[c for c in keep if c in j2["non_synced_df"].columns]].copy()
    nsyn["pairing"] = "monocular"
    fig2j = pd.concat([syn, nsyn], ignore_index=True)
    VALIDATION.append(f"Fig2J n={len(fig2j)} synced={len(syn)} monocular={len(nsyn)}")
    sheet(
        "Fig2J",
        [
            "Fig 2J. Axial-bias source events (concurrent vs monocular).",
            "One row = one saccade. peak_velocity is deg/frame in this table unless noted in the pickle.",
            f"n_concurrent={len(syn)}, n_monocular={len(nsyn)}, total={len(fig2j)}. PV_57 head_movement is unused/NaN in this dump.",
            "This event set is NOT the Fig 2E n=34876 sample.",
        ],
        ("Saccade events", fig2j),
    )

    # ------------------------------------------------------------------ Fig3A–C
    for letter, fname, desc in [
        ("Fig3A", "Fig_3_a/figure_3a.pickle", "Quiet-epoch example (default PV_126 block_007, 210–240 s)."),
        ("Fig3B", "Fig_3_b/figure_3b.pickle", "Active-epoch example traces."),
        ("Fig3C", "Fig_3_c/figure_3c.pickle", "Longer vignette with pupil, saccade rate, head movement, and state."),
    ]:
        d = pkl(MAIN / fname)
        tms = np.asarray(d["left_eye_data"]["ms_axis"], float)
        df = pd.DataFrame(
            {
                "time_ms": tms,
                "time_s": tms / 1000.0,
                "phi_L_deg": np.asarray(d["left_eye_data"]["k_phi"], float),
                "theta_L_deg": np.asarray(d["left_eye_data"]["k_theta"], float),
                "pupil_L": np.asarray(d["left_eye_data"]["pupil_diameter"], float),
                "phi_R_deg": np.asarray(d["right_eye_data"]["k_phi"], float),
                "theta_R_deg": np.asarray(d["right_eye_data"]["k_theta"], float),
                "pupil_R": np.asarray(d["right_eye_data"]["pupil_diameter"], float),
            }
        )
        extra = []
        if d.get("behavior_state"):
            extra.append(("Behavior-state epochs", pd.DataFrame(d["behavior_state"])))
        sheet(letter, [f"{letter}. {desc}", "Example traces, not the population dataset. Angles in degrees.", f"Window {d.get('start_time')}–{d.get('end_time')} (figure time units)."], ("Traces", df), *extra)

    # ------------------------------------------------------------------ Fig3D pupil hist (repo 3e)
    e3 = pkl(MAIN / "Fig_3_e/combined_aggregated_data.pkl")
    q = np.asarray(e3["quiet"], float)
    a = np.asarray(e3["active"], float)
    VALIDATION.append(f"Fig3D pupil quiet={q.size} active={a.size}")
    nmax = max(q.size, a.size)
    q2 = np.full(nmax, np.nan)
    a2 = np.full(nmax, np.nan)
    q2[: q.size] = q
    a2[: a.size] = a
    sheet(
        "Fig3D",
        [
            "Fig 3D (manuscript/PDF). Pupil-diameter samples in quiet vs active (repo folder Fig_3_e).",
            "One value = one pupil sample (mm). Animal IDs were not stored in the frozen pooled lists.",
            "State-dependent pupil analysis used n=4 animals (PV_106, PV_143, PV_62, PV_126). Do not add PV_57.",
            f"n_quiet={q.size}, n_active={a.size}. Columns are independent samples (not paired rows).",
        ],
        ("arrays", ["pupil_quiet_mm", "pupil_active_mm"], [q2, a2]),
    )

    # ------------------------------------------------------------------ Fig3E zscore (repo 3f)
    f3 = pkl(MAIN / "Fig_3_f/animal_diff_data_zscore.pkl")
    edges = np.linspace(-3.0, 3.0, 16)
    centers = 0.5 * (edges[:-1] + edges[1:])
    zrows = []
    for animal, tr in f3.items():
        tr = np.asarray(tr, float)
        n = min(len(centers), len(tr))
        for i in range(n):
            zrows.append({"animal": animal, "pupil_zscore_bin_center": float(centers[i]), "active_minus_quiet_density": float(tr[i])})
    sheet(
        "Fig3E",
        [
            "Fig 3E (manuscript/PDF). Per-animal difference of z-scored pupil-diameter densities (repo Fig_3_f).",
            "One row = one animal × bin. The four animals are the biological replicates (n=4).",
            "x-axis rebuilt as 15 bins on [-3, 3] to match the export/replot script.",
        ],
        ("Per-animal z-score difference curves", pd.DataFrame(zrows)),
    )

    # ------------------------------------------------------------------ Fig3F ISI (repo 3d)
    isi_log = pkl(MAIN / "Fig_3_d/ISI_histogram_plotdata.pkl")
    isi_lin = pkl(MAIN / "Fig_3_d/ISI_hist_linear_plotdata.pkl")

    def isi_table(blob, scale):
        rows = []
        for animal, rec in blob.get("per_animal", {}).items():
            x = np.asarray(rec["x"], float)
            y = np.asarray(rec["y"], float)
            n = min(len(x), len(y))
            for i in range(n):
                rows.append({"animal": animal, "scale": scale, "isi_x": float(x[i]), "pdf_y": float(y[i])})
        comb = blob.get("combined", {})
        if comb:
            x = np.asarray(comb["x"], float)
            y = np.asarray(comb["y"], float)
            n = min(len(x), len(y))
            for i in range(n):
                rows.append({"animal": "all_mean", "scale": scale, "isi_x": float(x[i]), "pdf_y": float(y[i])})
        return pd.DataFrame(rows)

    sheet(
        "Fig3F",
        [
            "Fig 3F (manuscript/PDF). Inter-saccadic interval distributions (repo Fig_3_d).",
            "Raw ISI lists were not frozen. One row = one plotted density vertex for one animal (or the combined mean).",
            "Linear and log panels are both stored. ISI x-axis as in the pickle (ms on linear; log-x as stored).",
        ],
        ("Linear-scale densities", isi_table(isi_lin, "linear")),
        ("Log-scale densities", isi_table(isi_log, "log")),
    )

    # ------------------------------------------------------------------ S1
    sim = pd.read_csv(SUPP / "S1/metadata/ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv")
    if "Unnamed: 0" in sim.columns:
        sim = sim.drop(columns=["Unnamed: 0"])
    per_block = pd.read_csv(SUPP / "S1/metadata/cohort_component_error_per_block.csv")
    if "block_path" in per_block.columns:
        per_block["block_path"] = per_block["block_path"].map(lambda x: sanitize(str(x)))
    across = pd.read_csv(SUPP / "S1/metadata/cohort_component_error_across_animals.csv")
    sheet(
        "FigS1A",
        [
            "Fig S1A. Simulation of angular reconstruction (known φ/θ vs ellipse geometry).",
            "One row = one simulated pose. Angles in degrees.",
        ],
        ("Simulation table", sim),
    )
    sheet(
        "FigS1B",
        [
            "Fig S1B. Per-block Kerr residual errors (φ, θ) after centering on rest.",
            "One row = one animal/block/eye/axis summary used for the across-animal means in the caption.",
            "Lab paths in block_path were replaced with <lab_path>.",
            "Caption: φ 0.84±0.48°, θ 1.19±0.48° (mean ± SD across animals, n=5).",
        ],
        ("Per-block residuals", per_block),
        ("Across-animal summary", across),
    )

    # ------------------------------------------------------------------ S2
    s2 = pd.read_csv(RECOVERED / "figure_S2_per_animal.csv")
    s2s = json.loads((RECOVERED / "figure_S2_overall_stats.json").read_text())
    VALIDATION.append(f"FigS2 main={s2s.get('main_axis_mean_deg', s2['main_span_mean'].mean()):.3f}")
    sheet(
        "FigS2",
        [
            "Fig S2. Per-animal oculomotor span (main axis vs perpendicular).",
            "One row = one animal (mean of the two eyes). Units: degrees.",
            f"Overall mean±SD: main {s2['main_span_mean'].mean():.2f}±{s2['main_span_mean'].std(ddof=1):.2f}°; perp {s2['perp_span_mean'].mean():.2f}±{s2['perp_span_mean'].std(ddof=1):.2f}° (manuscript 35.0±7.3 / 26.1±6.4).",
            "Recovered from the Paper_Figures pickle that matches the published panel (not eye_movement_span.py).",
        ],
        ("Per-animal spans", s2),
    )

    # ------------------------------------------------------------------ S3
    s3 = pkl(SUPP / "S3/metadata/figure_S3.pickle")
    still = np.asarray(s3["still"]["norm_counts"], float)
    moving = np.asarray(s3["moving"]["norm_counts"], float)
    xe = np.asarray(s3["still"]["xedges"], float)
    ye = np.asarray(s3["still"]["yedges"], float)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    s3_rows = []
    for i, x in enumerate(xc):
        for j, y in enumerate(yc):
            s3_rows.append(
                {
                    "right_peak_speed_bin_center_deg_per_ms": float(x),
                    "left_peak_speed_bin_center_deg_per_ms": float(y),
                    "probability_head_still": float(still[j, i]) if still.shape[0] == len(yc) else float(still[i, j]),
                    "probability_head_moving": float(moving[j, i]) if moving.shape[0] == len(yc) else float(moving[i, j]),
                }
            )
    VALIDATION.append(f"FigS3 n_still={s3['n_still']} n_moving={s3['n_moving']}")
    sheet(
        "FigS3",
        [
            "Fig S3. Head-still vs head-moving interocular peak-speed histograms.",
            f"Individual L/R pairs were not frozen. n_still={s3['n_still']}, n_moving={s3['n_moving']}; exclude_animals={s3.get('exclude_animals')}.",
            "One row = one 2D histogram bin (normalized to sum to 1 within each panel). Units on axes: deg/ms.",
        ],
        ("Normalized 2D histogram bins", pd.DataFrame(s3_rows)),
    )

    # ------------------------------------------------------------------ S4
    s4 = pd.read_csv(RECOVERED / "figure_S4_per_animal.csv")
    VALIDATION.append(
        f"FigS4 corr={s4['frac_corr'].mean()*100:.1f} weak={s4['frac_weak'].mean()*100:.1f} anti={s4['frac_anti'].mean()*100:.1f}"
    )
    sheet(
        "FigS4",
        [
            "Fig S4. Time-weighted fractions of L–R pupil correlation states (rolling Pearson, 10 s window). n=5 animals including PV_57.",
            "One row = one animal (the plotted stacked-bar replicate). This is NOT the n=4 pupil-state analysis.",
            "Correlated: r>0.2; weak: |r|≤0.2; anti: r<-0.2. Caption 66.3±7.9 / 23.4±3.5 / 10.3±4.9 is the time-weighted overall fraction ± SD of the five animal fractions.",
        ],
        ("Per-animal fractions", s4),
    )

    # ------------------------------------------------------------------ S5
    s5 = pd.read_csv(RECOVERED / "figure_S5_literature_spans.csv")
    sheet(
        "FigS5",
        [
            "Fig S5. Comparative oculomotor spans. Pogona values are this-study S2 means (rounded). Other species are literature values used in the figure.",
            "One row = one species. Units: degrees. Sources as in Document S2 (human [78], mouse [34], cormorant [10], barn owl [87]).",
        ],
        ("Species spans", s5),
    )

    # ------------------------------------------------------------------ S8E
    s8 = pkl(SUPP / "S8/S8j/metadata/s8_2f_lizard_mouse_all.pkl")
    n8 = max(len(s8["lizard"]["left"]), len(s8["mouse"]["left"]))
    liz_L = np.full(n8, np.nan)
    liz_R = np.full(n8, np.nan)
    mou_L = np.full(n8, np.nan)
    mou_R = np.full(n8, np.nan)
    liz_L[: len(s8["lizard"]["left"])] = np.asarray(s8["lizard"]["left"], float)
    liz_R[: len(s8["lizard"]["right"])] = np.asarray(s8["lizard"]["right"], float)
    mou_L[: len(s8["mouse"]["left"])] = np.asarray(s8["mouse"]["left"], float)
    mou_R[: len(s8["mouse"]["right"])] = np.asarray(s8["mouse"]["right"], float)
    VALIDATION.append(f"FigS8E lizard_n={len(s8['lizard']['left'])} mouse_n={len(s8['mouse']['left'])}")
    sheet(
        "FigS8E",
        [
            "Fig S8E (manuscript/PDF). Interocular peak-speed coupling, lizard vs mouse (repo S8j).",
            "One row is not a paired lizard–mouse observation: lizard and mouse columns are independent samples (different n).",
            f"n_lizard={len(s8['lizard']['left'])}, n_mouse={len(s8['mouse']['left'])}. Units: deg/ms. Lizard threshold 0.8°/frame; mouse 3.23°/frame.",
            "Concatenate is not required; unused cells are blank because n differs.",
        ],
        (
            "arrays",
            ["lizard_left_deg_per_ms", "lizard_right_deg_per_ms", "mouse_left_deg_per_ms", "mouse_right_deg_per_ms"],
            [liz_L, liz_R, mou_L, mou_R],
        ),
    )

    # ------------------------------------------------------------------ S9
    s9 = pkl(SUPP / "S9/metadata/unified_jitter.pkl")
    jv = pd.read_csv(SUPP / "S9/metadata/jitter_values.csv")
    VALIDATION.append("FigS9 " + ", ".join(f"{k} n={len(v)} med={np.median(v):.2f}" for k, v in s9["pools"].items()))
    ws9 = add_sheet(wb, "FigS9")
    r = write_notes(
        ws9,
        [
            "Fig S9. Camera jitter samples at the imaged eye plane (µm).",
            "Table 1: epoch-level median/p95 (matches the caption). Table 2: individual samples.",
            "Rigid (~1.47e6) and modular lizard (~1.03e6) exceed Excel’s row limit, so each is split across _part columns.",
            "To rebuild a vector: concatenate part1, then part2, … top-to-bottom.",
            "Fig1E is a different sample (see README).",
        ],
        nf,
    )
    ws9.write(r, 0, "Table 1. Mount summaries", hf)
    r = write_df(ws9, jv, r + 1, hf)
    ws9.write(r, 0, "Table 2. Individual displacements (µm); wrap parts as needed", hf)
    r += 1
    max_data = EXCEL_MAX_ROWS - r - 5
    headers, cols = [], []
    for key, label in [("rigid", "rigid_lizard_um"), ("modular", "modular_lizard_um"), ("mouse", "mouse_um"), ("turtle", "turtle_um")]:
        h, p = wrap_array_columns(label, np.asarray(s9["pools"][key], float), max_data)
        headers.extend(h)
        cols.extend(p)
    write_long_array(ws9, r, headers, cols, hf)

    # ------------------------------------------------------------------ S10
    s10_map = {
        "FigS10A": ("S10a_lizard_rayleigh_noise_core.pkl", "lizard", 0.8, 7.05),
        "FigS10B": ("S10b_mouse_rayleigh_noise_core.pkl", "mouse", 3.23, 10.58),
        "FigS10C": ("S10c_turtle_rayleigh_noise_core.pkl", "turtle", 2.0, 12.28),
    }
    for sheet_name, (fn, species, ms_thr, ms_snr) in s10_map.items():
        d = pkl(SUPP / "S10/metadata" / fn)
        vec = np.asarray(d["d"], float)
        med = float(np.median(vec))
        bmed = float(d.get("B_med", med / np.sqrt(2 * np.log(2))))
        thr_pkl = d.get("threshold")
        meta = pd.DataFrame(
            [
                {
                    "species": species,
                    "animal": d.get("animal"),
                    "n": int(d.get("n", vec.size)),
                    "median_D_deg_per_frame": med,
                    "B_med": bmed,
                    "two_B": 2 * bmed,
                    "threshold_in_pickle": (None if thr_pkl in (None, "None") else thr_pkl),
                    "SNR_pickle_thr_over_2B": d.get("snr_thr_over_2B"),
                    "manuscript_threshold_deg_per_frame": ms_thr,
                    "manuscript_SNR_thr_over_medianD": ms_snr,
                    "KS_D": None,
                }
            ]
        )
        VALIDATION.append(f"{sheet_name} n={vec.size} medianD={med:.4f}")
        sheet(
            sheet_name,
            [
                f"{sheet_name}. Frame-to-frame angular displacement D=hypot(Δφ,Δθ) in quiet windows ({species}).",
                "Table 1: definition metadata. Manuscript/PDF SNR = threshold/median(D). Pickle SNR = threshold/(2B).",
                "Table 2: one row = one inter-frame D sample (°/frame) used for the histogram.",
                "Example traces in the left of the published panel are illustrations; numeric series were not frozen (window times are in S10 YAML).",
            ],
            ("Metadata", meta),
            ("D samples", pd.DataFrame({"D_deg_per_frame": vec})),
        )

    # ------------------------------------------------------------------ S11
    s11isi = pkl(SUPP / "S11/metadata/ISI_ISI_histogram_plotdata.pickle")
    s11lin = pkl(SUPP / "S11/metadata/ISI_ISI_hist_linear_10_300ms_plotdata.pickle")
    sheet(
        "FigS11A",
        [
            "Fig S11A (manuscript/PDF). ISI distributions split by state in the figure; frozen pickles store per-animal density curves.",
            "Raw ISI lists were not frozen. One row = one plotted density vertex.",
            "Same-epoch ISIs only (intervals that cross Active/Quiet are excluded), as in Document S2.",
        ],
        ("Log-scale densities", isi_table(s11isi, "log")),
        ("Linear-scale densities", isi_table(s11lin, "linear")),
    )
    dur = pkl(SUPP / "S11/metadata/epoch_duration_bin_trials.pkl")
    act = np.asarray(dur["active"], float)
    qui = np.asarray(dur["quiet"], float)
    VALIDATION.append(f"FigS11B n_active={act.size} n_quiet={qui.size} med_a={np.median(act):.1f} med_q={np.median(qui):.1f}")
    nmax = max(act.size, qui.size)
    a2 = np.full(nmax, np.nan)
    q2 = np.full(nmax, np.nan)
    a2[: act.size] = act
    q2[: qui.size] = qui
    sheet(
        "FigS11B",
        [
            "Fig S11B (manuscript/PDF). Sustained Active/Quiet epoch durations after bridging <3 s interruptions and retaining epochs ≥5 s (repo epoch_duration_bin_trials.pkl).",
            f"One column = one epoch duration (s). n_active={act.size} (PDF 174), n_quiet={qui.size} (PDF 191). Methods medians ~15 s / 59 s.",
            "Columns are independent (not paired). Animal IDs were not stored on these arrays.",
        ],
        ("arrays", ["active_duration_s", "quiet_duration_s"], [a2, q2]),
    )

    # ------------------------------------------------------------------ S12
    c3 = pkl(MAIN / "Fig_3_c/figure_3c.pickle")
    tms = np.asarray(c3["left_eye_data"]["ms_axis"], float)
    t0 = 297867.25
    lo, hi = t0 - 250.0, t0 + 550.0
    m = (tms >= lo) & (tms <= hi)
    if m.sum() < 10:
        m = (tms / 1000.0 >= 297.617) & (tms / 1000.0 <= 298.417)
    s12a = pd.DataFrame(
        {
            "time_ms": tms[m],
            "time_s": tms[m] / 1000.0,
            "phi_L_deg": np.asarray(c3["left_eye_data"]["k_phi"], float)[m],
            "theta_L_deg": np.asarray(c3["left_eye_data"]["k_theta"], float)[m],
            "phi_R_deg": np.asarray(c3["right_eye_data"]["k_phi"], float)[m],
            "theta_R_deg": np.asarray(c3["right_eye_data"]["k_theta"], float)[m],
        }
    )
    sheet(
        "FigS12A",
        [
            "Fig S12A (manuscript/PDF). Example back-and-forth sequence, event_id PV_126_007_297867_R.",
            "Traces extracted from the Fig 3C vignette pickle (window 200–415 s) around primary onset 297.867 s.",
            "This is an illustration. One row = one time sample. Angles in degrees.",
        ],
        ("Example traces", s12a),
    )
    s12b = pd.read_csv(SUPP / "S12/metadata/large_saccades.csv")
    if "lizmov_path" in s12b.columns:
        s12b["lizmov_path"] = s12b["lizmov_path"].map(lambda x: sanitize(str(x)))
    n_hit = int(s12b["has_back_and_forth"].sum()) if "has_back_and_forth" in s12b.columns else 0
    VALIDATION.append(f"FigS12B n={len(s12b)} hits={n_hit} pct={100*n_hit/len(s12b):.2f}")
    sheet(
        "FigS12B",
        [
            "Fig S12B (manuscript/PDF). Rate of post-saccadic back-and-forth among large head-moving saccades.",
            "One row = one unique large saccade during annotated head movement. has_back_and_forth is the plotted classification.",
            f"n={len(s12b)}, n_hit={n_hit}, percent={100 * n_hit / max(len(s12b),1):.1f} (manuscript 21.3% = 639/2997).",
            "Lab paths sanitized.",
        ],
        ("Large head-moving saccades", s12b),
    )

    # ------------------------------------------------------------------ S13
    s13 = pkl(SUPP / "S13/metadata/s13_preonset_2e.pkl")
    sl = s13["slope_summary"]
    VALIDATION.append(f"FigS13 n_all={sl['n_events_all']} t={sl['paired_t']:.3f} p={sl['paired_p']:.4f}")
    rows_abc = []
    for cls_name, key in [("all", "all"), ("concurrent", "concurrent"), ("monocular", "monocular")]:
        pas = s13["classes"][key]["per_animal_stats"]
        for animal, df in pas.items():
            tmp = df.copy()
            tmp.insert(0, "animal", animal)
            tmp.insert(0, "saccade_class", cls_name)
            rows_abc.append(tmp)
    abc = pd.concat(rows_abc, ignore_index=True)
    pooled = pd.DataFrame(
        {
            "amplitude_deg": np.asarray(s13["pooled"]["amp"], float),
            "peak_velocity_deg_per_ms": np.asarray(s13["pooled"]["vel"], float),
        }
    )
    e_df = pd.DataFrame(
        {
            "animal": sl["animals"],
            "concurrent_slope": sl["concurrent_slopes"],
            "monocular_slope": sl["monocular_slopes"],
        }
    )
    stats = pd.DataFrame(
        [
            {
                "amp_definition": sl["amp_definition"],
                "n_events_all": sl["n_events_all"],
                "n_concurrent": sl["n_events_concurrent"],
                "n_monocular": sl["n_events_monocular"],
                "paired_t": sl["paired_t"],
                "paired_p": sl["paired_p"],
                "df": sl["df"],
                "concurrent_mean_slope": sl["concurrent_mean_slope"],
                "monocular_mean_slope": sl["monocular_mean_slope"],
            }
        ]
    )
    note_s13 = [
        "Fig S13. Main-sequence with S13 amplitude (onset→offset; length-1 events use pre-onset).",
        "This is a different event definition from Fig 2E (n=33294 vs 34876).",
        "A–C: per-animal binned mean peak velocity (the plotted colored lines).",
        "D: pooled individual saccades (animal ID not stored on the pooled vectors).",
        "E: per-animal OLS slopes (the plotted paired points) + paired t-test.",
        "Units: amplitude deg; velocity deg/ms; slope (deg/ms)/deg.",
    ]
    sheet("FigS13A", ["Fig S13A. All saccades, per-animal binned means."] + note_s13, ("Per-animal bin means (class=all)", abc[abc["saccade_class"] == "all"]))
    sheet("FigS13B", ["Fig S13B. Concurrent saccades, per-animal binned means."] + note_s13, ("Per-animal bin means (class=concurrent)", abc[abc["saccade_class"] == "concurrent"]))
    sheet("FigS13C", ["Fig S13C. Monocular saccades, per-animal binned means."] + note_s13, ("Per-animal bin means (class=monocular)", abc[abc["saccade_class"] == "monocular"]))
    sheet("FigS13D", ["Fig S13D. Pooled individual saccades (n=33294)."] + note_s13, ("Pooled observations", pooled))
    sheet("FigS13E", ["Fig S13E. Per-animal concurrent vs monocular OLS slopes."] + note_s13, ("Per-animal slopes", e_df), ("Paired t-test", stats))

    wb.close()

    report = ROOT / "publication/VALIDATION_REPORT.txt"
    report.write_text("\n".join(VALIDATION) + "\n")
    print("Wrote", REPO_OUT, "size_MB", round(REPO_OUT.stat().st_size / 1e6, 2))
    print("\n".join(VALIDATION))
    if DESKTOP_OUT.parent.is_dir():
        import shutil

        shutil.copy2(REPO_OUT, DESKTOP_OUT)
        print("Copied", DESKTOP_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
