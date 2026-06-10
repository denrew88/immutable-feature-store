from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq


POINT_KEY_COLS = ["sample_id", "feature_id", "point_idx"]
TRACE_INDEX_COLS = ["sample_id", "feature_id", "trace_len"]


def _fail(label: str, message: str):
    raise AssertionError(f"[{label}] {message}")


def _resolve_against(base_path: Path, stored_path: str) -> Path:
    path = Path(str(stored_path))
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "array-sample-parquet":
        _fail("manifest", f"unexpected format: {manifest.get('format')!r}")
    if int(manifest.get("version", 0)) != 1:
        _fail("manifest", f"unexpected version: {manifest.get('version')!r}")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        _fail("raw-log", f"missing raw commit log: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                _fail("raw-log", f"invalid JSON at line {line_no}: {exc}")
            out.append(dict(item))
    return out


def _latest_raw_records(stage_dir: Path, n_samples: int) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for item in _read_jsonl(stage_dir / "raw_samples.jsonl"):
        try:
            sample_id = int(item["sample_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if sample_id < 0 or sample_id >= int(n_samples):
            _fail("raw-log", f"sample_id out of range in raw log: {sample_id}")
        point_path = (stage_dir / str(item.get("path") or "")).resolve()
        trace_index_path = (stage_dir / str(item.get("trace_index_path") or "")).resolve()
        if not point_path.exists():
            _fail("raw-log", f"missing raw point parquet for sample_id={sample_id}: {point_path}")
        if not trace_index_path.exists():
            _fail("raw-log", f"missing raw trace-index parquet for sample_id={sample_id}: {trace_index_path}")
        record = dict(item)
        record["_point_path"] = str(point_path)
        record["_trace_index_path"] = str(trace_index_path)
        records[sample_id] = record
    return records


def _parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def _read_many(paths: list[Path], columns: list[str]) -> pl.DataFrame:
    paths = [path for path in paths if path.exists() and _parquet_row_count(path) > 0]
    if not paths:
        return pl.DataFrame({name: [] for name in columns}).select(columns)
    return pl.scan_parquet([str(path) for path in paths], glob=False).select(columns).collect()


def _sort_frame(df: pl.DataFrame, keys: list[str], columns: list[str]) -> pl.DataFrame:
    if df.height <= 1:
        return df.select(columns)
    return df.select(columns).sort(keys)


def _assert_series_equal(label: str, name: str, actual: pl.Series, expected: pl.Series):
    if len(actual) != len(expected):
        _fail(label, f"{name} length mismatch: actual={len(actual)} expected={len(expected)}")
    if len(actual) == 0:
        return

    if actual.dtype in {pl.Float32, pl.Float64} or expected.dtype in {pl.Float32, pl.Float64}:
        actual_values = actual.to_numpy().astype(np.float64, copy=False)
        expected_values = expected.to_numpy().astype(np.float64, copy=False)
        equal = (actual_values == expected_values) | (np.isnan(actual_values) & np.isnan(expected_values))
        if bool(np.all(equal)):
            return
        mismatch = int(np.flatnonzero(~equal)[0])
        _fail(
            label,
            f"{name} mismatch at row={mismatch}: actual={actual_values[mismatch]!r} expected={expected_values[mismatch]!r}",
        )

    actual_values = actual.to_list()
    expected_values = expected.to_list()
    for idx, (actual_value, expected_value) in enumerate(zip(actual_values, expected_values)):
        if actual_value != expected_value:
            _fail(
                label,
                f"{name} mismatch at row={idx}: actual={actual_value!r} expected={expected_value!r}",
            )


def _assert_frame_equal(label: str, actual: pl.DataFrame, expected: pl.DataFrame, columns: list[str]):
    if actual.height != expected.height:
        _fail(label, f"row_count mismatch: actual={actual.height} expected={expected.height}")
    for name in columns:
        _assert_series_equal(label, name, actual[name], expected[name])


def _assert_sorted(label: str, df: pl.DataFrame, keys: list[str]):
    if df.height <= 1:
        return
    sorted_keys = df.select(keys).sort(keys)
    _assert_frame_equal(label, df.select(keys), sorted_keys, keys)


def _assert_unique_pairs(label: str, df: pl.DataFrame, keys: list[str]):
    if df.height <= 1:
        return
    unique_count = int(df.select(keys).unique().height)
    if unique_count != int(df.height):
        _fail(label, f"duplicate trace key rows detected for keys={keys}")


def _validate_raw_records(label: str, records: dict[int, dict[str, Any]], n_samples: int, require_all: bool):
    if require_all and len(records) != int(n_samples):
        pending = [idx for idx in range(int(n_samples)) if idx not in records]
        _fail(label, f"raw stage is not complete: completed={len(records)} pending={pending[:20]}")

    for sample_id, record in sorted(records.items()):
        point_path = Path(record["_point_path"])
        trace_index_path = Path(record["_trace_index_path"])
        point_rows = _parquet_row_count(point_path)
        trace_rows = _parquet_row_count(trace_index_path)
        if int(record.get("row_count", point_rows)) != point_rows:
            _fail(label, f"raw row_count mismatch for sample_id={sample_id}: log={record.get('row_count')} parquet={point_rows}")
        if int(record.get("trace_count", trace_rows)) != trace_rows:
            _fail(label, f"raw trace_count mismatch for sample_id={sample_id}: log={record.get('trace_count')} parquet={trace_rows}")

        point_df = pl.read_parquet(point_path)
        trace_index_df = pl.read_parquet(trace_index_path)
        _assert_sorted(f"{label}:raw-points:{sample_id}", point_df, POINT_KEY_COLS)
        _assert_sorted(f"{label}:raw-trace-index:{sample_id}", trace_index_df, ["sample_id", "feature_id"])
        _assert_unique_pairs(f"{label}:raw-trace-index:{sample_id}", trace_index_df, ["sample_id", "feature_id"])
        if point_df.height and set(int(value) for value in point_df["sample_id"].to_list()) != {sample_id}:
            _fail(label, f"raw point file contains another sample_id: {point_path}")
        if trace_index_df.height and set(int(value) for value in trace_index_df["sample_id"].to_list()) != {sample_id}:
            _fail(label, f"raw trace-index file contains another sample_id: {trace_index_path}")
        trace_len_sum = int(trace_index_df.select(pl.col("trace_len").sum()).item() or 0)
        if trace_len_sum != int(point_df.height):
            _fail(label, f"raw trace_len sum mismatch for sample_id={sample_id}: trace_len_sum={trace_len_sum} point_rows={point_df.height}")


def _validate_part_metadata(label: str, manifest_path: Path, manifest: dict[str, Any]):
    for part in manifest.get("parts") or []:
        part_id = int(part["part_id"])
        point_path = _resolve_against(manifest_path, part["path"])
        trace_index_path = _resolve_against(manifest_path, part["trace_index_path"])
        if not point_path.exists():
            _fail(label, f"missing final point part: {point_path}")
        if not trace_index_path.exists():
            _fail(label, f"missing final trace-index part: {trace_index_path}")
        if int(part["row_count"]) != _parquet_row_count(point_path):
            _fail(label, f"final part row_count mismatch at part_id={part_id}")
        if int(part["trace_count"]) != _parquet_row_count(trace_index_path):
            _fail(label, f"final trace_count mismatch at part_id={part_id}")
        if int(part["byte_size"]) != int(os.path.getsize(point_path)):
            _fail(label, f"final byte_size mismatch at part_id={part_id}")
        if int(part.get("trace_index_byte_size", 0)) != int(os.path.getsize(trace_index_path)):
            _fail(label, f"final trace_index_byte_size mismatch at part_id={part_id}")

        point_df = pl.read_parquet(point_path)
        trace_index_df = pl.read_parquet(trace_index_path)
        _assert_sorted(f"{label}:final-points:{part_id}", point_df, POINT_KEY_COLS)
        _assert_sorted(f"{label}:final-trace-index:{part_id}", trace_index_df, ["sample_id", "feature_id"])
        _assert_unique_pairs(f"{label}:final-trace-index:{part_id}", trace_index_df, ["sample_id", "feature_id"])
        trace_len_sum = int(trace_index_df.select(pl.col("trace_len").sum()).item() or 0)
        if trace_len_sum != int(point_df.height):
            _fail(label, f"final trace_len sum mismatch at part_id={part_id}: trace_len_sum={trace_len_sum} point_rows={point_df.height}")


def _validate_meta_counts(label: str, manifest_path: Path, manifest: dict[str, Any]):
    sample_meta_path = _resolve_against(manifest_path, manifest["sample_meta_path"])
    feature_meta_path = _resolve_against(manifest_path, manifest["feature_meta_path"])
    if pl.read_parquet(sample_meta_path).height != int(manifest["n_samples"]):
        _fail(label, "sample_meta row count does not match manifest n_samples")
    if pl.read_parquet(feature_meta_path).height != int(manifest["n_features"]):
        _fail(label, "feature_meta row count does not match manifest n_features")


def validate_manifest(
    manifest_path: str | Path,
    *,
    stage_dir: str | Path | None = None,
    require_all_raw_samples: bool = True,
    label: str = "array-sample-parquet",
):
    final_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(final_manifest_path)
    stage_root = Path(stage_dir).expanduser().resolve() if stage_dir is not None else final_manifest_path.parent
    point_columns = POINT_KEY_COLS + [str(item["name"]) for item in manifest.get("point_schema") or []]

    _validate_meta_counts(label, final_manifest_path, manifest)
    records = _latest_raw_records(stage_root, int(manifest["n_samples"]))
    _validate_raw_records(label, records, int(manifest["n_samples"]), bool(require_all_raw_samples))
    _validate_part_metadata(label, final_manifest_path, manifest)

    raw_point_paths = [Path(record["_point_path"]) for _, record in sorted(records.items())]
    raw_trace_index_paths = [Path(record["_trace_index_path"]) for _, record in sorted(records.items())]
    final_point_paths = [_resolve_against(final_manifest_path, part["path"]) for part in manifest.get("parts") or []]
    final_trace_index_paths = [_resolve_against(final_manifest_path, part["trace_index_path"]) for part in manifest.get("parts") or []]

    expected_points = _sort_frame(_read_many(raw_point_paths, point_columns), POINT_KEY_COLS, point_columns)
    actual_points = _sort_frame(_read_many(final_point_paths, point_columns), POINT_KEY_COLS, point_columns)
    _assert_frame_equal(f"{label}:points", actual_points, expected_points, point_columns)

    expected_trace_index = _sort_frame(
        _read_many(raw_trace_index_paths, TRACE_INDEX_COLS),
        ["sample_id", "feature_id"],
        TRACE_INDEX_COLS,
    )
    actual_trace_index = _sort_frame(
        _read_many(final_trace_index_paths, TRACE_INDEX_COLS),
        ["sample_id", "feature_id"],
        TRACE_INDEX_COLS,
    )
    _assert_frame_equal(f"{label}:trace-index", actual_trace_index, expected_trace_index, TRACE_INDEX_COLS)

    print(
        f"{label} validation passed: samples={manifest['n_samples']} features={manifest['n_features']} "
        f"traces={actual_trace_index.height} point_rows={actual_points.height}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--allow-incomplete-raw", action="store_true")
    parser.add_argument("--label", default="array-sample-parquet")
    args = parser.parse_args()
    validate_manifest(
        args.manifest_path,
        stage_dir=args.stage_dir,
        require_all_raw_samples=not bool(args.allow_incomplete_raw),
        label=args.label,
    )


if __name__ == "__main__":
    main()
