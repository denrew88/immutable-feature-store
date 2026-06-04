from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _fail(label: str, part_id: int, message: str):
    raise AssertionError(f"{label} part={part_id}: {message}")


def _expected_missing_count(n_samples: int, n_features: int) -> int:
    total = 0
    for sample_id in range(n_samples):
        residue = (-sample_id) % 5
        total += (n_features + 4 - residue) // 5
    return total


def validate_manifest(manifest_path: Path, *, label: str = "", progress_every: int = 250):
    """생성 규칙이 `(sample_id + feature_id) % 5 != 0`인 synthetic dense-long shard를 전수 검사한다."""

    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    label = label or manifest_path.parent.parent.name
    n_samples = int(data["n_samples"])
    n_features = int(data["n_features"])
    expected_total_rows = n_samples * n_features
    observed_total_rows = 0
    observed_present_rows = 0
    observed_missing_rows = 0
    parts = data["parts"]
    previous_last = -1

    for idx, part in enumerate(parts):
        part_id = int(part["part_id"])
        first = int(part["first_feature_id"])
        last = int(part["last_feature_id"])
        feature_count = int(part["feature_count"])
        row_count = int(part["row_count"])
        expected_feature_count = last - first + 1
        expected_rows = expected_feature_count * n_samples
        if part_id != idx:
            _fail(label, part_id, f"part_id sequence mismatch: expected {idx}")
        if first != previous_last + 1:
            _fail(label, part_id, f"feature range gap/overlap: first={first}, previous_last={previous_last}")
        if feature_count != expected_feature_count:
            _fail(label, part_id, f"feature_count mismatch: {feature_count} != {expected_feature_count}")
        if row_count != expected_rows:
            _fail(label, part_id, f"manifest row_count mismatch: {row_count} != {expected_rows}")
        previous_last = last

        path = _resolve(manifest_path, part["path"])
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows != expected_rows:
            _fail(label, part_id, f"parquet row_count mismatch: {pf.metadata.num_rows} != {expected_rows}")

        expected_feature_ids = np.repeat(np.arange(first, last + 1, dtype=np.int32), n_samples)
        expected_sample_ids = np.tile(np.arange(n_samples, dtype=np.int64), expected_feature_count)
        expected_mask = ((expected_feature_ids.astype(np.int64) + expected_sample_ids) % 5 != 0).astype(np.uint8)
        expected_values = np.where(
            expected_mask.astype(bool),
            expected_sample_ids.astype(np.float64) * 1000.0 + expected_feature_ids.astype(np.float64),
            0.0,
        )

        offset = 0
        for batch in pf.iter_batches(columns=["feature_id", "sample_id", "mask", "value"], batch_size=262144):
            size = batch.num_rows
            fids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            sids = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            masks = batch.column(2).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
            values = batch.column(3).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            span = slice(offset, offset + size)
            if not np.array_equal(fids, expected_feature_ids[span]):
                mismatch = int(np.flatnonzero(fids != expected_feature_ids[span])[0])
                _fail(label, part_id, f"feature_id mismatch at row {offset + mismatch}")
            if not np.array_equal(sids, expected_sample_ids[span]):
                mismatch = int(np.flatnonzero(sids != expected_sample_ids[span])[0])
                _fail(label, part_id, f"sample_id mismatch at row {offset + mismatch}")
            if not np.array_equal(masks, expected_mask[span]):
                mismatch = int(np.flatnonzero(masks != expected_mask[span])[0])
                _fail(label, part_id, f"mask mismatch at row {offset + mismatch}")
            if not np.array_equal(values, expected_values[span]):
                mismatch = int(np.flatnonzero(values != expected_values[span])[0])
                _fail(label, part_id, f"value mismatch at row {offset + mismatch}")
            present = int(masks.sum())
            observed_present_rows += present
            observed_missing_rows += int(size - present)
            offset += size

        if offset != expected_rows:
            _fail(label, part_id, f"iterated row count mismatch: {offset} != {expected_rows}")
        observed_total_rows += offset
        if progress_every > 0 and ((idx + 1) % progress_every == 0 or idx + 1 == len(parts)):
            elapsed = time.perf_counter() - started
            print(f"{label}: checked parts {idx + 1}/{len(parts)} rows={observed_total_rows} elapsed_sec={elapsed:.1f}", flush=True)

    if previous_last != n_features - 1:
        raise AssertionError(f"{label}: final feature coverage mismatch: last={previous_last}, expected={n_features - 1}")
    if observed_total_rows != expected_total_rows:
        raise AssertionError(f"{label}: total row count mismatch: {observed_total_rows} != {expected_total_rows}")
    expected_missing = _expected_missing_count(n_samples, n_features)
    expected_present = expected_total_rows - expected_missing
    if observed_missing_rows != expected_missing:
        raise AssertionError(f"{label}: missing row count mismatch: {observed_missing_rows} != {expected_missing}")
    if observed_present_rows != expected_present:
        raise AssertionError(f"{label}: present row count mismatch: {observed_present_rows} != {expected_present}")

    elapsed = time.perf_counter() - started
    print(
        f"{label}: exhaustive validation passed rows={observed_total_rows} "
        f"present={observed_present_rows} missing={observed_missing_rows} parts={len(parts)} elapsed_sec={elapsed:.3f}",
        flush=True,
    )


def main(argv=None):
    ap = ArgumentParser()
    ap.add_argument("manifest_paths", nargs="+")
    ap.add_argument("--progress-every", type=int, default=250)
    args = ap.parse_args(argv)
    for raw_path in args.manifest_paths:
        validate_manifest(Path(raw_path), progress_every=int(args.progress_every))


if __name__ == "__main__":
    main(sys.argv[1:])
