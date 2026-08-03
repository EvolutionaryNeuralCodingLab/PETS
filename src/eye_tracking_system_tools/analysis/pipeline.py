"""Orchestrate analyzed-block → event tables → figure exports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Collection

import pandas as pd

from eye_tracking_system_tools.analysis.binocular import (
    combine_synced_dataframes,
    find_synced_saccades_ms,
)
from eye_tracking_system_tools.analysis.block_registry import BlockSpec, load_registry
from eye_tracking_system_tools.analysis.export_meta import load_params_yaml, write_pickle_with_meta
from eye_tracking_system_tools.analysis.eye_trace_io import (
    LoadedBlockEyes,
    csv_choices_meta,
    load_block_eyes,
)
from eye_tracking_system_tools.analysis.head_labels import label_saccades_head_movement
from eye_tracking_system_tools.analysis.saccade_events import (
    create_saccade_events_with_direction_segmentation_robust,
)


@dataclass
class BlockBundle:
    spec: BlockSpec
    left: pd.DataFrame
    right: pd.DataFrame
    left_csv_meta: dict
    right_csv_meta: dict
    l_saccades: pd.DataFrame
    r_saccades: pd.DataFrame
    all_saccades: pd.DataFrame


@dataclass
class EventTables:
    blocks: list[BlockBundle]
    all_saccades: pd.DataFrame
    synced: pd.DataFrame
    non_synced: pd.DataFrame
    csv_meta: list[dict] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def block_dict(self) -> dict[str, BlockBundle]:
        return {b.spec.block_key: b for b in self.blocks}


def _saccade_params(params: dict) -> dict:
    s = params.get("saccade", {})
    return {
        "speed_threshold": float(s.get("speed_threshold_deg_per_frame", 0.8)),
        "directional_delta_threshold_deg": float(
            s.get("directional_delta_threshold_deg", 90.0)
        ),
        "min_subsaccade_samples": int(s.get("min_subsaccade_samples", 2)),
        "min_net_disp": float(s.get("min_net_disp_deg", 0.5)),
        "speed_profile": bool(s.get("speed_profile", True)),
    }


def _row_block_key(animal: Any, block: Any) -> str:
    """Rebuild ``BlockSpec.block_key`` from event-table ``animal`` / ``block`` columns."""
    animal_s = str(animal)
    block_s = str(block)
    # block_num is stored zero-padded (e.g. "001"); tolerate int or "1"/"block_001".
    digits = "".join(c for c in block_s if c.isdigit())
    block_num = digits.zfill(3) if digits else block_s
    return f"{animal_s}_block_{block_num}"


def _mask_by_block_keys(df: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    if "animal" not in df.columns or "block" not in df.columns:
        return df.iloc[0:0].copy()
    mask = [
        _row_block_key(a, b) in keys
        for a, b in zip(df["animal"].tolist(), df["block"].tolist())
    ]
    return df.loc[mask].reset_index(drop=True)


def filter_event_tables(
    tables: EventTables,
    *,
    block_keys: Collection[str] | None = None,
    animals: Collection[str] | None = None,
) -> EventTables:
    """
    Subset ``EventTables`` by block key and/or animal.

    Binocular pairs are formed within a block, so dropping whole blocks never
    splits a ``Main`` group. ``params`` are left untouched (use
    ``dataclasses.replace`` / :func:`with_params` for overrides).
    """
    if block_keys is None and animals is None:
        return tables

    animal_set = {str(a) for a in animals} if animals is not None else None
    key_set = {str(k) for k in block_keys} if block_keys is not None else None

    def _keep_key(key: str, animal: str) -> bool:
        if key_set is not None and key not in key_set:
            return False
        if animal_set is not None and animal not in animal_set:
            return False
        return True

    bundles = [
        b for b in tables.blocks
        if _keep_key(b.spec.block_key, b.spec.animal)
    ]

    if bundles and len(tables.csv_meta) == len(tables.blocks):
        csv_meta = [
            m for b, m in zip(tables.blocks, tables.csv_meta)
            if _keep_key(b.spec.block_key, b.spec.animal)
        ]
    else:
        csv_meta = list(tables.csv_meta)

    if bundles:
        all_saccades = pd.concat(
            [b.all_saccades for b in bundles], ignore_index=True
        )
    else:
        # Event-only tables (e.g. from a frozen pickle): filter by columns.
        if tables.all_saccades.empty:
            all_saccades = tables.all_saccades.copy()
        else:
            keep = [
                _keep_key(_row_block_key(a, b), str(a))
                for a, b in zip(
                    tables.all_saccades["animal"].tolist(),
                    tables.all_saccades["block"].tolist(),
                )
            ]
            all_saccades = tables.all_saccades.loc[keep].reset_index(drop=True)

    # Derive the final key set for synced / non_synced masking.
    if bundles:
        final_keys = {b.spec.block_key for b in bundles}
    elif not all_saccades.empty:
        final_keys = {
            _row_block_key(a, b)
            for a, b in zip(
                all_saccades["animal"].tolist(),
                all_saccades["block"].tolist(),
            )
        }
    else:
        final_keys = key_set or set()

    synced = _mask_by_block_keys(tables.synced, final_keys)
    non_synced = _mask_by_block_keys(tables.non_synced, final_keys)

    return EventTables(
        blocks=bundles,
        all_saccades=all_saccades,
        synced=synced,
        non_synced=non_synced,
        csv_meta=csv_meta,
        params=tables.params,
    )


# ---------------------------------------------------------------------------
# Row-level saccade filters (head movement, concurrent/monocular, queries)
# ---------------------------------------------------------------------------

_EVENT_KINDS = frozenset({"all", "concurrent", "monocular", "synced", "non_synced"})
_HEAD_TRUE = frozenset({True, "with", "true", "yes", "1"})
_HEAD_FALSE = frozenset({False, "without", "false", "no", "0"})
_HEAD_LABELED = frozenset({"labeled", "notna", "notnull"})
_PAIR_COLS = ("Main", "Sub")


@dataclass(frozen=True)
class SaccadeFilter:
    """Row-level event filter applied after block/animal selection.

    Parameters
    ----------
    event_kind
        ``"all"`` keep both partitions; ``"concurrent"`` / ``"synced"`` keep
        binocular pairs only; ``"monocular"`` / ``"non_synced"`` keep unpaired.
    head_movement
        ``None`` / ``"any"`` — no filter; ``True`` / ``"with"`` — head-coupled;
        ``False`` / ``"without"`` — head-stationary; ``"labeled"`` — drop NaN.
    query
        Optional pandas ``DataFrame.query`` expression (after presets / equals).
    column_equals
        Column → required value (``True`` / ``False`` / scalar / ``None`` for NA).
    """

    event_kind: str = "all"
    head_movement: bool | str | None = None
    query: str | None = None
    column_equals: dict[str, Any] | None = None

    def normalized_kind(self) -> str:
        kind = str(self.event_kind or "all").strip().lower()
        if kind in {"synced", "binocular", "concurrent"}:
            return "concurrent"
        if kind in {"non_synced", "nonsynced", "monocular", "uniocular"}:
            return "monocular"
        if kind not in _EVENT_KINDS and kind != "all":
            raise ValueError(
                f"Unknown event_kind {self.event_kind!r}; "
                "expected all / concurrent / monocular"
            )
        return "all" if kind == "all" else kind

    def normalized_head(self) -> bool | str | None:
        hm = self.head_movement
        if hm is None or (isinstance(hm, str) and hm.strip().lower() in {"", "any", "all"}):
            return None
        if isinstance(hm, str):
            key = hm.strip().lower()
        else:
            key = hm
        if key in _HEAD_TRUE:
            return True
        if key in _HEAD_FALSE:
            return False
        if isinstance(key, str) and key in _HEAD_LABELED:
            return "labeled"
        raise ValueError(
            f"Unknown head_movement {self.head_movement!r}; "
            "expected any / with / without / labeled"
        )

    def is_active(self) -> bool:
        if self.normalized_kind() != "all":
            return True
        if self.normalized_head() is not None:
            return True
        if self.query and str(self.query).strip():
            return True
        if self.column_equals:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event_kind": self.normalized_kind(),
            "head_movement": self.normalized_head(),
            "query": (str(self.query).strip() or None) if self.query else None,
            "column_equals": dict(self.column_equals) if self.column_equals else None,
        }
        return {k: v for k, v in out.items() if v is not None and v != {} and v != "all"}

    @classmethod
    def from_mapping(cls, data: SaccadeFilter | dict[str, Any] | None) -> SaccadeFilter | None:
        if data is None:
            return None
        if isinstance(data, SaccadeFilter):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"saccade_filter must be dict or SaccadeFilter, got {type(data)!r}")
        if not data:
            return None
        return cls(
            event_kind=str(data.get("event_kind", "all")),
            head_movement=data.get("head_movement"),
            query=data.get("query"),
            column_equals=data.get("column_equals") or data.get("columns"),
        )

    def describe(self) -> str:
        parts: list[str] = []
        kind = self.normalized_kind()
        if kind != "all":
            parts.append(kind)
        hm = self.normalized_head()
        if hm is True:
            parts.append("with head")
        elif hm is False:
            parts.append("without head")
        elif hm == "labeled":
            parts.append("head labeled")
        for col, val in (self.column_equals or {}).items():
            parts.append(f"{col}={val!r}")
        q = str(self.query).strip() if self.query else ""
        if q:
            parts.append(f"query:{q}")
        return ", ".join(parts) if parts else "none"


def _empty_like(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df.iloc[0:0].copy()


def _drop_pair_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _PAIR_COLS if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _apply_row_predicates(df: pd.DataFrame, filt: SaccadeFilter) -> pd.DataFrame:
    """Apply head_movement / column_equals / query; leave event_kind to the caller."""
    if df is None or df.empty:
        return _empty_like(df)

    out = df
    hm = filt.normalized_head()
    if hm is not None:
        if "head_movement" not in out.columns:
            raise KeyError(
                "head_movement filter requested but column is missing from event table"
            )
        col = out["head_movement"]
        if hm is True:
            out = out.loc[col == True]  # noqa: E712
        elif hm is False:
            out = out.loc[col == False]  # noqa: E712
        else:  # labeled
            out = out.loc[col.notna()]

    for col_name, want in (filt.column_equals or {}).items():
        if col_name not in out.columns:
            raise KeyError(f"column_equals: {col_name!r} not in event table")
        series = out[col_name]
        if want is None or (isinstance(want, float) and pd.isna(want)):
            out = out.loc[series.isna()]
        elif want is True or want is False:
            out = out.loc[series == want]
        else:
            out = out.loc[series == want]

    q = str(filt.query).strip() if filt.query else ""
    if q:
        out = out.query(q, engine="python")

    return out.reset_index(drop=True)


def _reslice_block_events(
    bundles: list[BlockBundle],
    all_saccades: pd.DataFrame,
) -> list[BlockBundle]:
    """Replace per-block event frames from the filtered ``all_saccades`` pool."""
    if not bundles:
        return []
    by_key: dict[str, pd.DataFrame] = {}
    if not all_saccades.empty and {"animal", "block"}.issubset(all_saccades.columns):
        for (animal, block), g in all_saccades.groupby(["animal", "block"], sort=False):
            by_key[_row_block_key(animal, block)] = g.reset_index(drop=True)

    out: list[BlockBundle] = []
    for b in bundles:
        g = by_key.get(b.spec.block_key)
        if g is None or g.empty:
            empty = _empty_like(b.all_saccades if b.all_saccades is not None else pd.DataFrame())
            out.append(
                replace(b, l_saccades=empty.copy(), r_saccades=empty.copy(), all_saccades=empty)
            )
            continue
        if "eye" in g.columns:
            l_ev = g.loc[g["eye"].astype(str) == "L"].reset_index(drop=True)
            r_ev = g.loc[g["eye"].astype(str) == "R"].reset_index(drop=True)
        else:
            l_ev = g.copy()
            r_ev = g.copy()
        out.append(replace(b, l_saccades=l_ev, r_saccades=r_ev, all_saccades=g))
    return out


def apply_saccade_filter(
    tables: EventTables,
    filt: SaccadeFilter | dict[str, Any] | None,
) -> EventTables:
    """
    Filter saccade rows by kind (concurrent / monocular), head-movement flag,
    column equality, and/or a pandas query.

    Block list / traces / ``params`` are preserved; per-block event frames are
    re-sliced from the filtered ``all_saccades`` pool.
    """
    parsed = SaccadeFilter.from_mapping(filt)
    if parsed is None or not parsed.is_active():
        return tables

    kind = parsed.normalized_kind()
    synced = _apply_row_predicates(tables.synced, parsed)
    non_synced = _apply_row_predicates(tables.non_synced, parsed)
    all_saccades = _apply_row_predicates(tables.all_saccades, parsed)

    if kind == "concurrent":
        synced = synced.reset_index(drop=True)
        non_synced = _empty_like(tables.non_synced)
        all_saccades = _drop_pair_cols(synced).reset_index(drop=True)
    elif kind == "monocular":
        non_synced = non_synced.reset_index(drop=True)
        synced = _empty_like(tables.synced)
        all_saccades = non_synced.copy()
    else:
        # Keep partition filters in sync with the all_saccades row filter.
        synced = synced.reset_index(drop=True)
        non_synced = non_synced.reset_index(drop=True)
        all_saccades = all_saccades.reset_index(drop=True)

    bundles = _reslice_block_events(tables.blocks, all_saccades)
    return EventTables(
        blocks=bundles,
        all_saccades=all_saccades,
        synced=synced,
        non_synced=non_synced,
        csv_meta=list(tables.csv_meta),
        params=tables.params,
    )


def boolish_event_columns(df: pd.DataFrame, *, exclude: Collection[str] = ()) -> list[str]:
    """Column names that look like True/False/NaN flags (for GUI truth filters)."""
    skip = {str(c) for c in exclude}
    allowed = {True, False, 0, 1, 0.0, 1.0}
    found: list[str] = []
    if df is None or df.empty:
        return found
    for col in df.columns:
        name = str(col)
        if name in skip or name in _PAIR_COLS:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            found.append(name)
            continue
        sample = series.dropna().head(64)
        if sample.empty:
            continue
        vals: set[Any] = set()
        ok = True
        for v in sample:
            # Skip profile / nested columns (speed profiles, etc.).
            if isinstance(v, (list, tuple, dict, set)):
                ok = False
                break
            ndim = getattr(v, "ndim", None)
            if ndim is not None and int(ndim) > 0:
                ok = False
                break
            try:
                # Normalize numpy scalar bools/ints into Python hashes.
                if hasattr(v, "item") and not isinstance(v, (bytes, str)):
                    try:
                        v = v.item()
                    except (ValueError, AttributeError):
                        ok = False
                        break
                vals.add(v)
            except TypeError:
                ok = False
                break
        if ok and vals and vals <= allowed:
            found.append(name)
    return found


def with_params(tables: EventTables, overrides: dict[str, Any] | None) -> EventTables:
    """Shallow-copy ``tables`` with deep-merged ``params`` overrides."""
    if not overrides:
        return tables
    merged = deep_merge_params(tables.params, overrides)
    return replace(tables, params=merged)


def deep_merge_params(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a deep copy of ``base``."""
    out = deepcopy(base) if base else {}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_params(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def drop_traces(tables: EventTables) -> EventTables:
    """Return a copy whose ``BlockBundle.left/right`` frames are empty (cache-friendly)."""
    slim = []
    for b in tables.blocks:
        slim.append(
            replace(
                b,
                left=pd.DataFrame(),
                right=pd.DataFrame(),
            )
        )
    return replace(tables, blocks=slim)


def build_event_tables(
    specs: list[BlockSpec],
    *,
    params: dict[str, Any] | None = None,
    keep_traces: bool = True,
    prefer_finalized: bool = True,
) -> EventTables:
    """Detect saccades per eye, label metadata, pair concurrent vs monocular.

    When ``keep_traces`` is False the per-frame eye DataFrames are dropped after
    detection (events / synced tables are retained). Prefer this for caches;
    Fig 2f needs traces and should reload them for its selected blocks only.

    When ``prefer_finalized`` is True (default), blocks with
    ``analysis/saccades/`` outputs from the preprocessing Saccades tab are
    loaded instead of re-detecting. A warning is printed if saved params
    differ from the run YAML.
    """
    from eye_tracking_system_tools.analysis.saccade_export import (
        block_bundle_from_finalized,
        finalized_params_match,
        has_finalized_saccades,
        read_finalized_saccades,
    )

    params = params or {}
    sp = _saccade_params(params)
    sync_diff_ms = float(params.get("binocular", {}).get("sync_diff_ms", 34.0))

    bundles: list[BlockBundle] = []
    synced_parts: list[pd.DataFrame] = []
    non_synced_parts: list[pd.DataFrame] = []
    csv_meta: list[dict] = []

    empty_df = pd.DataFrame()

    for spec in specs:
        if prefer_finalized and has_finalized_saccades(spec.block_path):
            if not finalized_params_match(spec.block_path, params):
                print(
                    f"[{spec.block_key}] WARNING: finalized saccade params differ "
                    f"from run YAML — using on-disk events anyway"
                )
            finalized = read_finalized_saccades(spec.block_path)
            bundle = block_bundle_from_finalized(
                spec, finalized, keep_traces=keep_traces
            )
            synced = finalized.synced
            non_synced = finalized.non_synced
            synced_parts.append(synced)
            non_synced_parts.append(non_synced)
            meta = {
                "left_csv": bundle.left_csv_meta.get("path"),
                "left_rule": bundle.left_csv_meta.get("rule"),
                "right_csv": bundle.right_csv_meta.get("path"),
                "right_rule": bundle.right_csv_meta.get("rule"),
                "source": f"finalized:{finalized.source}",
            }
            csv_meta.append(meta)
            bundles.append(bundle)
            n_pairs = (
                0
                if synced.empty or "Main" not in synced.columns
                else int(synced["Main"].nunique())
            )
            print(
                f"[{spec.block_key}] finalized saccades "
                f"n={len(bundle.all_saccades)} synced_pairs≈{n_pairs} "
                f"non_synced={len(non_synced)} (from {finalized.source})"
            )
            continue

        loaded: LoadedBlockEyes = load_block_eyes(spec)
        left_df, l_ev = create_saccade_events_with_direction_segmentation_robust(
            loaded.left, **sp
        )
        right_df, r_ev = create_saccade_events_with_direction_segmentation_robust(
            loaded.right, **sp
        )

        for ev, eye in ((l_ev, "L"), (r_ev, "R")):
            if ev.empty:
                continue
            ev["eye"] = eye
            ev["block"] = spec.block_num
            ev["animal"] = spec.animal

        l_ev = label_saccades_head_movement(l_ev, spec)
        r_ev = label_saccades_head_movement(r_ev, spec)
        all_ev = pd.concat([l_ev, r_ev], ignore_index=True)

        synced, non_synced = find_synced_saccades_ms(
            all_ev, sync_diff_ms=sync_diff_ms
        )
        synced_parts.append(synced)
        non_synced_parts.append(non_synced)

        meta = csv_choices_meta(loaded)
        csv_meta.append(meta)
        bundles.append(
            BlockBundle(
                spec=spec,
                left=left_df if keep_traces else empty_df,
                right=right_df if keep_traces else empty_df,
                left_csv_meta={
                    "path": meta["left_csv"],
                    "rule": meta["left_rule"],
                },
                right_csv_meta={
                    "path": meta["right_csv"],
                    "rule": meta["right_rule"],
                },
                l_saccades=l_ev,
                r_saccades=r_ev,
                all_saccades=all_ev,
            )
        )
        print(
            f"[{spec.block_key}] saccades L={len(l_ev)} R={len(r_ev)} "
            f"synced_pairs≈{0 if synced.empty else synced['Main'].nunique()} "
            f"non_synced={len(non_synced)}"
        )

    all_saccades = (
        pd.concat([b.all_saccades for b in bundles], ignore_index=True)
        if bundles
        else pd.DataFrame()
    )
    synced_all = combine_synced_dataframes(synced_parts)
    non_synced_all = (
        pd.concat(non_synced_parts, ignore_index=True)
        if non_synced_parts
        else pd.DataFrame()
    )

    return EventTables(
        blocks=bundles,
        all_saccades=all_saccades,
        synced=synced_all,
        non_synced=non_synced_all,
        csv_meta=csv_meta,
        params=params,
    )


def run_figure_exports(
    tables: EventTables,
    out_dir: Path,
    *,
    figures: list[str] | None = None,
    include_archived_2f: bool = False,
) -> dict[str, Path]:
    """Export requested figure pickles/PDFs under ``out_dir/{figures,metadata}/``."""
    from eye_tracking_system_tools.analysis import (
        figures_2c_2e,
        figures_2f_2h_2i,
        figures_2g_2j,
    )
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    out_dir = Path(out_dir)
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    wanted = set(figures or ["2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j"])
    written: dict[str, Path] = {}

    # Persist event tables once for reuse / inspection.
    events_pkl = metadata_dir / "event_tables.pkl"
    write_pickle_with_meta(
        {
            "all_saccades": tables.all_saccades,
            "synced": tables.synced,
            "non_synced": tables.non_synced,
        },
        events_pkl,
        meta={
            "csv_choices": tables.csv_meta,
            "params": tables.params,
            "n_all": int(len(tables.all_saccades)),
            "n_synced_rows": int(len(tables.synced)),
            "n_non_synced": int(len(tables.non_synced)),
        },
        entrypoint="eye_tracking_system_tools.analysis.pipeline.build_event_tables",
    )
    written["event_tables"] = events_pkl

    if wanted & {"2c", "2d"}:
        p = figures_2c_2e.export_pos_vel_bundle(tables, out_dir)
        written["2c_2d"] = p
    if "2e" in wanted:
        written["2e"] = figures_2c_2e.export_amplitude_velocity_fit(tables, out_dir)
    if "2f" in wanted:
        if include_archived_2f or not tables.blocks:
            written["2f"] = figures_2f_2h_2i.export_archived_figure_2f(out_dir)
        else:
            written["2f"] = figures_2f_2h_2i.export_figure_2f(tables, out_dir)
    if "2h" in wanted:
        written["2h"] = figures_2f_2h_2i.export_figure_2h(tables, out_dir)
    if "2i" in wanted:
        written["2i"] = figures_2f_2h_2i.export_figure_2i(tables, out_dir)
    if "2g" in wanted:
        top, bot = figures_2g_2j.export_figure_2g(tables, out_dir)
        written["2g_top"] = top
        written["2g_bot"] = bot
    if "2j" in wanted:
        written["2j"] = figures_2g_2j.export_figure_2j(tables, out_dir)

    written["_figures_dir"] = figures_dir
    written["_metadata_dir"] = metadata_dir
    return written


def event_tables_from_saccade_angles_pickle(
    pickle_path: Path | str,
    *,
    params: dict[str, Any] | None = None,
) -> EventTables:
    """
    Build ``EventTables`` from a Fig-2j-style pickle
    (``{"synced_df", "non_synced_df"}``), e.g. the paper reproduction file.

    Use this for paper-faithful replot of 2c–2e / 2g / 2h / 2i / 2j when you
    want the exact published event set. Fig 2f is emitted from the archived
    ``figure_2f_nodowncast.pickle`` in event-pickle mode.
    """
    import pickle

    pickle_path = Path(pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "synced_df" not in data or "non_synced_df" not in data:
        raise ValueError(
            f"{pickle_path}: expected dict with synced_df and non_synced_df"
        )
    synced = data["synced_df"].copy()
    nons = data["non_synced_df"].copy()
    all_saccades = pd.concat([synced, nons], ignore_index=True)
    return EventTables(
        blocks=[],
        all_saccades=all_saccades,
        synced=synced,
        non_synced=nons,
        csv_meta=[{"source_pickle": str(pickle_path.resolve())}],
        params=params or {},
    )


def run_from_registry(
    registry_path: Path,
    params_path: Path,
    out_dir: Path,
    *,
    figures: list[str] | None = None,
) -> dict[str, Path]:
    specs = load_registry(registry_path)
    params = load_params_yaml(params_path)
    tables = build_event_tables(specs, params=params)
    return run_figure_exports(tables, out_dir, figures=figures)


def run_from_event_pickle(
    event_pickle: Path,
    params_path: Path,
    out_dir: Path,
    *,
    figures: list[str] | None = None,
) -> dict[str, Path]:
    """Export figures from a frozen event pickle (paper event set)."""
    params = load_params_yaml(params_path)
    tables = event_tables_from_saccade_angles_pickle(event_pickle, params=params)
    wanted = list(figures) if figures is not None else [
        "2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j"
    ]
    return run_figure_exports(
        tables,
        out_dir,
        figures=wanted,
        include_archived_2f=True,
    )
