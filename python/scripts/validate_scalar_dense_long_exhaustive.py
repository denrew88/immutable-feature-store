from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base.parent / path).resolve()


def _fail(label: str, message: str):
    raise AssertionError(f"{label}: {message}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auto_stage_manifest(final_manifest_path: Path) -> Path | None:
    candidates = [
        final_manifest_path.parent / "sample_major_manifest.json",
        final_manifest_path.parent.parent / "sample_major_manifest.json",
        final_manifest_path.parent.parent / "stage" / "sample_major_manifest.json",
        final_manifest_path.parent.parent / "scalar_stage" / "sample_major_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_raw_sample(path: Path, sample_id: int | None, sample_id_col: str, feature_id_col: str, value_col: str) -> pl.DataFrame:
    schema = pl.read_parquet_schema(path)
    if sample_id_col in schema:
        df = pl.read_parquet(path, columns=[sample_id_col, feature_id_col, value_col])
    else:
        if sample_id is None:
            raise ValueError(f"raw sample file has no {sample_id_col} column and no sample_ids sidecar: {path}")
        df = pl.read_parquet(path, columns=[feature_id_col, value_col]).with_columns(
            pl.lit(int(sample_id), dtype=pl.Int64).alias(sample_id_col)
        )
    return df.select(
        [
            pl.col(sample_id_col).cast(pl.Int64).alias("sample_id"),
            pl.col(feature_id_col).cast(pl.Int32).alias("feature_id"),
            pl.col(value_col).cast(pl.Float64).alias("value"),
        ]
    )


def _raw_reference_for_part(
    stage_manifest: dict[str, Any],
    stage_manifest_path: Path,
    *,
    first_feature_id: int,
    last_feature_id: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = int(last_feature_id) - int(first_feature_id) + 1
    expected_values = np.full((feature_count, int(n_samples)), np.nan, dtype=np.float64)
    expected_mask = np.zeros((feature_count, int(n_samples)), dtype=np.uint8)
    sample_id_col = str(stage_manifest.get("sample_id_col", "sample_id"))
    feature_id_col = str(stage_manifest.get("feature_id_col", "feature_id"))
    value_col = str(stage_manifest.get("value_col", "value"))
    sample_paths = [_resolve(stage_manifest_path, str(value)) for value in stage_manifest.get("sample_paths", [])]
    sample_ids_raw = stage_manifest.get("sample_ids")
    sample_ids = None if sample_ids_raw is None else [int(value) for value in sample_ids_raw]
    if sample_ids is not None and len(sample_ids) != len(sample_paths):
        raise ValueError("stage sample_ids length does not match sample_paths length")

    frames = []
    for idx, raw_path in enumerate(sample_paths):
        sample_id = None if sample_ids is None else sample_ids[idx]
        df = _read_raw_sample(raw_path, sample_id, sample_id_col, feature_id_col, value_col)
        df = df.filter((pl.col("feature_id") >= int(first_feature_id)) & (pl.col("feature_id") <= int(last_feature_id)))
        if df.height:
            frames.append(df)
    if not frames:
        return expected_values, expected_mask

    raw = pl.concat(frames, how="vertical")
    raw = raw.filter(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
    duplicate_count = raw.group_by(["sample_id", "feature_id"]).len().filter(pl.col("len") > 1).height
    if duplicate_count:
        raise AssertionError(f"raw reference contains duplicate sample/feature rows: {duplicate_count}")

    sample_ids_np = raw["sample_id"].to_numpy().astype(np.int64, copy=False)
    feature_ids_np = raw["feature_id"].to_numpy().astype(np.int32, copy=False)
    values_np = raw["value"].to_numpy().astype(np.float64, copy=False)
    if sample_ids_np.size:
        if int(sample_ids_np.min()) < 0 or int(sample_ids_np.max()) >= int(n_samples):
            raise AssertionError("raw reference sample_id is outside final shard sample range")
    offsets = feature_ids_np.astype(np.int64, copy=False) - int(first_feature_id)
    expected_values[offsets, sample_ids_np] = values_np
    expected_mask[offsets, sample_ids_np] = 1
    return expected_values, expected_mask


def _r2_one_vs_y(values: np.ndarray, mask: np.ndarray, y: np.ndarray, y_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overlap = (mask.astype(bool, copy=False) & y_valid.reshape(1, -1))
    n = overlap.sum(axis=1).astype(np.int32, copy=False)
    r2 = np.zeros(values.shape[0], dtype=np.float64)
    for idx in range(values.shape[0]):
        if int(n[idx]) < 2:
            continue
        m = overlap[idx]
        xv = values[idx, m]
        yv = y[m]
        sx = float(xv.sum())
        sy = float(yv.sum())
        sx2 = float((xv * xv).sum())
        sy2 = float((yv * yv).sum())
        sxy = float((xv * yv).sum())
        count = float(n[idx])
        var_x = sx2 - sx * sx / count
        var_y = sy2 - sy * sy / count
        if var_x <= 0.0 or var_y <= 0.0:
            continue
        cov = sxy - sx * sy / count
        value = cov / np.sqrt(var_x * var_y)
        r2[idx] = float(value * value)
    return r2, n


def _assert_array_equal(label: str, part_id: int, name: str, actual: np.ndarray, expected: np.ndarray):
    if np.array_equal(actual, expected):
        return
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
        actual_flat = actual.reshape(-1)
        expected_flat = expected.reshape(-1)
        equal = (actual_flat == expected_flat) | (np.isnan(actual_flat) & np.isnan(expected_flat))
        if bool(np.all(equal)):
            return
        mismatch = int(np.flatnonzero(~equal)[0])
        raise AssertionError(
            f"{label}: part={part_id} {name} mismatch at flat row {mismatch}: "
            f"actual={actual_flat[mismatch]!r} expected={expected_flat[mismatch]!r}"
        )
    mismatch = int(np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))[0])
    raise AssertionError(f"{label}: part={part_id} {name} mismatch at flat row {mismatch}")


def _validate_locator(label: str, final_manifest_path: Path, manifest: dict[str, Any]):
    locator_path = _resolve(final_manifest_path, str(manifest["feature_locator_path"]))
    locator = pl.read_parquet(locator_path).sort("feature_id")
    n_features = int(manifest["n_features"])
    n_samples = int(manifest["n_samples"])
    if locator.height != n_features:
        _fail(label, f"feature_locator row count mismatch: {locator.height} != {n_features}")
    part_by_feature: dict[int, tuple[int, int]] = {}
    for part in manifest.get("parts", []):
        part_id = int(part["part_id"])
        first = int(part["first_feature_id"])
        last = int(part["last_feature_id"])
        for feature_id in range(first, last + 1):
            part_by_feature[feature_id] = (part_id, feature_id - first)
    for row in locator.iter_rows(named=True):
        feature_id = int(row["feature_id"])
        expected_part_id, expected_offset = part_by_feature[feature_id]
        if int(row["part_id"]) != expected_part_id:
            _fail(label, f"locator part_id mismatch for feature_id={feature_id}")
        if int(row["offset_in_part"]) != expected_offset:
            _fail(label, f"locator offset_in_part mismatch for feature_id={feature_id}")
        if int(row["first_row_in_part"]) != expected_offset * n_samples:
            _fail(label, f"locator first_row_in_part mismatch for feature_id={feature_id}")


def _validate_selection_stats(
    label: str,
    final_manifest_path: Path,
    manifest: dict[str, Any],
    part_values: dict[int, np.ndarray],
    part_masks: dict[int, np.ndarray],
    r2_atol: float,
):
    stats_map = manifest.get("selection_stats") or {}
    if not stats_map:
        return
    sample_meta = pl.read_parquet(_resolve(final_manifest_path, str(manifest["sample_meta_path"])))
    for y_col, rel_path in stats_map.items():
        if y_col not in sample_meta.columns:
            _fail(label, f"selection stats y column not found in sample_meta: {y_col}")
        y = sample_meta[str(y_col)].to_numpy().astype(np.float64, copy=False)
        y_valid = ~np.isnan(y)
        stats = pl.read_parquet(_resolve(final_manifest_path, str(rel_path))).sort("feature_id")
        if stats.height != int(manifest["n_features"]):
            _fail(label, f"{y_col} stats row count mismatch: {stats.height} != {manifest['n_features']}")
        for part in manifest.get("parts", []):
            part_id = int(part["part_id"])
            first = int(part["first_feature_id"])
            last = int(part["last_feature_id"])
            expected_r2, expected_n = _r2_one_vs_y(part_values[part_id], part_masks[part_id], y, y_valid)
            actual = stats.filter((pl.col("feature_id") >= first) & (pl.col("feature_id") <= last)).sort("feature_id")
            actual_n = actual["n_y_overlap"].to_numpy().astype(np.int32, copy=False)
            actual_r2 = actual["r2y"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
            _assert_array_equal(label, part_id, f"{y_col}.n_y_overlap", actual_n, expected_n)
            if not np.allclose(actual_r2, expected_r2, rtol=0.0, atol=float(r2_atol), equal_nan=False):
                mismatch = int(np.flatnonzero(np.abs(actual_r2 - expected_r2) > float(r2_atol))[0])
                feature_id = first + mismatch
                raise AssertionError(
                    f"{label}: {y_col}.r2y mismatch feature_id={feature_id} "
                    f"actual={actual_r2[mismatch]} expected={expected_r2[mismatch]}"
                )


def validate_manifest(
    final_manifest_path: Path,
    *,
    sample_major_manifest_path: Path | None = None,
    label: str = "",
    progress_every: int = 250,
    r2_atol: float = 1e-10,
):
    """Validate a dense-long scalar shard against actual raw sample parquet input.

    This intentionally does not re-create values from a synthetic formula. The
    final dense rows are compared against committed raw sample parquet files
    referenced by scalar-sample-major-v1.
    """

    started = time.perf_counter()
    final_manifest_path = final_manifest_path.resolve()
    manifest = _load_json(final_manifest_path)
    label = label or final_manifest_path.parent.name
    stage_path = sample_major_manifest_path.resolve() if sample_major_manifest_path else _auto_stage_manifest(final_manifest_path)
    if stage_path is None:
        _fail(label, "sample_major_manifest.json was not found; pass --sample-major-manifest for direct value validation")
    stage_manifest = _load_json(stage_path)
    if str(stage_manifest.get("format")) != "scalar-sample-major-v1":
        _fail(label, f"unsupported stage manifest format: {stage_manifest.get('format')}")

    n_samples = int(manifest["n_samples"])
    n_features = int(manifest["n_features"])
    parts = list(manifest.get("parts", []))
    previous_last = -1
    observed_total_rows = 0
    part_values: dict[int, np.ndarray] = {}
    part_masks: dict[int, np.ndarray] = {}

    _validate_locator(label, final_manifest_path, manifest)

    for idx, part in enumerate(parts):
        part_id = int(part["part_id"])
        first = int(part["first_feature_id"])
        last = int(part["last_feature_id"])
        feature_count = int(part["feature_count"])
        manifest_row_count = int(part["row_count"])
        expected_feature_count = last - first + 1
        expected_rows = expected_feature_count * n_samples
        if part_id != idx:
            _fail(label, f"part_id sequence mismatch: {part_id} != {idx}")
        if first != previous_last + 1:
            _fail(label, f"feature range gap/overlap at part={part_id}: first={first}, previous_last={previous_last}")
        if feature_count != expected_feature_count:
            _fail(label, f"feature_count mismatch at part={part_id}: {feature_count} != {expected_feature_count}")
        if manifest_row_count != expected_rows:
            _fail(label, f"manifest row_count mismatch at part={part_id}: {manifest_row_count} != {expected_rows}")
        previous_last = last

        path = _resolve(final_manifest_path, str(part["path"]))
        pf = pq.ParquetFile(path)
        if int(pf.metadata.num_rows) != expected_rows:
            _fail(label, f"parquet row_count mismatch at part={part_id}: {pf.metadata.num_rows} != {expected_rows}")

        final_df = pl.read_parquet(path, columns=["feature_id", "sample_id", "mask", "value"])
        if final_df.height != expected_rows:
            _fail(label, f"read row_count mismatch at part={part_id}: {final_df.height} != {expected_rows}")
        expected_feature_ids = np.repeat(np.arange(first, last + 1, dtype=np.int32), n_samples)
        expected_sample_ids = np.tile(np.arange(n_samples, dtype=np.int64), expected_feature_count)
        actual_feature_ids = final_df["feature_id"].to_numpy().astype(np.int32, copy=False)
        actual_sample_ids = final_df["sample_id"].to_numpy().astype(np.int64, copy=False)
        _assert_array_equal(label, part_id, "feature_id order", actual_feature_ids, expected_feature_ids)
        _assert_array_equal(label, part_id, "sample_id order", actual_sample_ids, expected_sample_ids)

        expected_values, expected_mask = _raw_reference_for_part(
            stage_manifest,
            stage_path,
            first_feature_id=first,
            last_feature_id=last,
            n_samples=n_samples,
        )
        actual_mask = final_df["mask"].to_numpy().astype(np.uint8, copy=False).reshape(expected_feature_count, n_samples)
        actual_values = final_df["value"].to_numpy().astype(np.float64, copy=False).reshape(expected_feature_count, n_samples)
        _assert_array_equal(label, part_id, "mask", actual_mask, expected_mask)
        _assert_array_equal(label, part_id, "value", actual_values, expected_values)
        part_values[part_id] = actual_values
        part_masks[part_id] = actual_mask
        observed_total_rows += expected_rows

        if progress_every > 0 and ((idx + 1) % progress_every == 0 or idx + 1 == len(parts)):
            elapsed = time.perf_counter() - started
            print(f"{label}: checked parts {idx + 1}/{len(parts)} rows={observed_total_rows} elapsed_sec={elapsed:.1f}", flush=True)

    if previous_last != n_features - 1:
        _fail(label, f"final feature coverage mismatch: last={previous_last}, expected={n_features - 1}")
    if observed_total_rows != n_samples * n_features:
        _fail(label, f"total row count mismatch: {observed_total_rows} != {n_samples * n_features}")

    _validate_selection_stats(label, final_manifest_path, manifest, part_values, part_masks, float(r2_atol))
    elapsed = time.perf_counter() - started
    print(
        f"{label}: direct validation passed rows={observed_total_rows} parts={len(parts)} "
        f"stage_manifest={stage_path} elapsed_sec={elapsed:.3f}",
        flush=True,
    )


def main(argv=None):
    ap = ArgumentParser()
    ap.add_argument("manifest_paths", nargs="+")
    ap.add_argument("--sample-major-manifest", default="")
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument("--r2-atol", type=float, default=1e-10)
    args = ap.parse_args(argv)
    sample_major = Path(args.sample_major_manifest).resolve() if args.sample_major_manifest else None
    if sample_major is not None and len(args.manifest_paths) != 1:
        raise SystemExit("--sample-major-manifest can only be used with one final manifest path")
    for raw_path in args.manifest_paths:
        validate_manifest(
            Path(raw_path),
            sample_major_manifest_path=sample_major,
            progress_every=int(args.progress_every),
            r2_atol=float(args.r2_atol),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
