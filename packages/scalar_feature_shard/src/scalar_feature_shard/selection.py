"""Dense-long scalar feature-selection facade."""

import polars as pl

from ._impl.candidates import build_candidates_from_stats
from ._impl.dense_long import ScalarDenseLongDataset, load_dense_long_manifest
from ._impl.incremental import select_features_incremental
from .config import SelectionConfig
from .exceptions import ManifestFormatError
from .models import SelectionCandidate, SelectionOptions, SelectionResult


def _resolve_selection_options(
    options: SelectionOptions | SelectionConfig | None,
    *,
    y_col,
    y_r2_threshold,
    min_non_null_y,
    ff_r2_threshold,
    min_non_null_pair,
    top_m,
    initial_cap,
    max_step,
    batch_size,
    max_gap,
    max_candidates,
    mask_fastpath_min_group,
    mask_fastpath_min_pairs,
):
    """Merge object-style selection options with explicit keyword overrides."""

    base = options or SelectionOptions()
    if isinstance(base, SelectionConfig):
        base = SelectionOptions(
            y_col="y",
            y_r2_threshold=base.y_r2_threshold,
            min_non_null_y=base.min_non_null_y,
            ff_r2_threshold=base.ff_r2_threshold,
            min_non_null_pair=base.min_non_null_pair,
            top_m=base.top_m,
            initial_cap=base.initial_cap,
            max_step=base.max_step,
            batch_size=base.batch_size,
            max_gap=base.max_gap,
            max_candidates=base.max_candidates,
            mask_fastpath_min_group=base.mask_fastpath_min_group,
            mask_fastpath_min_pairs=base.mask_fastpath_min_pairs,
        )
    return SelectionOptions(
        y_col=str(base.y_col if y_col is None else y_col),
        y_r2_threshold=float(base.y_r2_threshold if y_r2_threshold is None else y_r2_threshold),
        min_non_null_y=int(base.min_non_null_y if min_non_null_y is None else min_non_null_y),
        ff_r2_threshold=float(base.ff_r2_threshold if ff_r2_threshold is None else ff_r2_threshold),
        min_non_null_pair=int(base.min_non_null_pair if min_non_null_pair is None else min_non_null_pair),
        top_m=int(base.top_m if top_m is None else top_m),
        initial_cap=int(base.initial_cap if initial_cap is None else initial_cap),
        max_step=int(base.max_step if max_step is None else max_step),
        batch_size=int(base.batch_size if batch_size is None else batch_size),
        max_gap=int(base.max_gap if max_gap is None else max_gap),
        max_candidates=int(base.max_candidates if max_candidates is None else max_candidates),
        mask_fastpath_min_group=int(
            base.mask_fastpath_min_group if mask_fastpath_min_group is None else mask_fastpath_min_group
        ),
        mask_fastpath_min_pairs=int(
            base.mask_fastpath_min_pairs if mask_fastpath_min_pairs is None else mask_fastpath_min_pairs
        ),
    )


def _load_feature_keys_by_id(manifest):
    key_col = str(getattr(manifest, "feature_key_col", "") or "")
    if not key_col:
        return None
    df = pl.read_parquet(manifest.feature_meta_path, columns=[key_col])
    if key_col not in df.columns:
        return None
    return tuple(None if value is None else str(value) for value in df[key_col].to_list())


def select_features(
    manifest_path,
    *,
    options: SelectionOptions | SelectionConfig | None = None,
    y_col: str | None = None,
    y_r2_threshold: float | None = None,
    min_non_null_y: int | None = None,
    ff_r2_threshold: float | None = None,
    min_non_null_pair: int | None = None,
    top_m: int | None = None,
    initial_cap: int | None = None,
    max_step: int | None = None,
    batch_size: int | None = None,
    max_gap: int | None = None,
    max_candidates: int | None = None,
    mask_fastpath_min_group: int | None = None,
    mask_fastpath_min_pairs: int | None = None,
    include_candidates: bool = False,
):
    """Run scalar feature selection against a dense-long shard manifest."""

    resolved = _resolve_selection_options(
        options,
        y_col=y_col,
        y_r2_threshold=y_r2_threshold,
        min_non_null_y=min_non_null_y,
        ff_r2_threshold=ff_r2_threshold,
        min_non_null_pair=min_non_null_pair,
        top_m=top_m,
        initial_cap=initial_cap,
        max_step=max_step,
        batch_size=batch_size,
        max_gap=max_gap,
        max_candidates=max_candidates,
        mask_fastpath_min_group=mask_fastpath_min_group,
        mask_fastpath_min_pairs=mask_fastpath_min_pairs,
    )
    try:
        manifest = load_dense_long_manifest(str(manifest_path))
    except Exception as exc:
        raise ManifestFormatError(f"failed to load dense-long scalar manifest: {manifest_path}") from exc

    stats_path = (manifest.selection_stats or {}).get(resolved.y_col)
    if not stats_path:
        raise ValueError(f"dense-long selection stats not found for y column: {resolved.y_col}")

    candidates = build_candidates_from_stats(
        stats_path,
        min_non_null_y=resolved.min_non_null_y,
        y_r2_threshold=resolved.y_r2_threshold,
        max_candidates=resolved.max_candidates,
    )
    with ScalarDenseLongDataset(str(manifest_path)) as reader:
        selected = select_features_incremental(
            candidates,
            reader,
            SelectionConfig(
                y_r2_threshold=resolved.y_r2_threshold,
                min_non_null_y=resolved.min_non_null_y,
                ff_r2_threshold=resolved.ff_r2_threshold,
                min_non_null_pair=resolved.min_non_null_pair,
                top_m=resolved.top_m,
                initial_cap=resolved.initial_cap,
                max_step=resolved.max_step,
                batch_size=resolved.batch_size,
                max_gap=resolved.max_gap,
                max_candidates=resolved.max_candidates,
                mask_fastpath_min_group=resolved.mask_fastpath_min_group,
                mask_fastpath_min_pairs=resolved.mask_fastpath_min_pairs,
            ),
        )

    feature_keys_by_id = _load_feature_keys_by_id(manifest)
    selected_feature_ids = tuple(int(candidate.feature_id) for candidate in selected)
    selected_feature_keys = tuple(
        None if feature_keys_by_id is None else feature_keys_by_id[int(candidate.feature_id)]
        for candidate in selected
    )
    public_candidates = ()
    if include_candidates:
        public_candidates = tuple(
            SelectionCandidate(
                feature_id=int(candidate.feature_id),
                feature_key=None if feature_keys_by_id is None else feature_keys_by_id[int(candidate.feature_id)],
                r2_y=float(candidate.r2_y),
                n_valid_y=int(candidate.n_valid_y),
            )
            for candidate in candidates
        )
    return SelectionResult(
        y_col=resolved.y_col,
        selected_feature_ids=selected_feature_ids,
        selected_feature_keys=selected_feature_keys,
        candidates=public_candidates,
        candidate_count=int(len(candidates)),
        selected_count=int(len(selected_feature_ids)),
        used_precomputed_stats=True,
    )


def run_selection(manifest_path, **kwargs) -> SelectionResult:
    """Alias for `select_features(...)`."""

    return select_features(manifest_path, **kwargs)
