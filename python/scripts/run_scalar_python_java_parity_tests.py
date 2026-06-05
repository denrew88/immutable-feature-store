from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "packages" / "scalar_feature_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scalar_feature_shard import build_dense_long_shards_from_sample_major_manifest
from validate_scalar_dense_long_exhaustive import validate_manifest


STATS_Y_COLS = ("y", "y_alt", "y_const")


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _java_exe(name: str) -> str:
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    default = Path("C:/Program Files/Java/jdk-1.8/bin") / name
    if default.exists():
        return str(default)
    return name


def _classpath(repo_root: Path) -> str:
    return os.pathsep.join([str(repo_root / "java" / "out"), str(repo_root / "java" / "lib" / "*")])


def _run(cmd: list[str], *, cwd: Path):
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            + "cmd: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed


def _compile_java(repo_root: Path):
    _run(
        [
            _java_exe("javac.exe" if os.name == "nt" else "javac"),
            "-encoding",
            "UTF-8",
            "-cp",
            str(repo_root / "java" / "lib" / "*"),
            "-d",
            str(repo_root / "java" / "out"),
            "@" + str(repo_root / "java" / "sources.txt"),
        ],
        cwd=repo_root,
    )


def _make_input(root: Path, *, n_samples: int, n_features: int, density: float) -> Path:
    rng = np.random.default_rng(20260605)
    stage_dir = root / "stage"
    raw_dir = stage_dir / "raw_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = np.arange(n_samples, dtype=np.int64)
    y = rng.normal(loc=0.0, scale=3.0, size=n_samples).astype(np.float64)
    y_alt = rng.normal(loc=1.0, scale=2.0, size=n_samples).astype(np.float64)
    y_const = np.full(n_samples, 7.0, dtype=np.float64)
    y[::11] = np.nan
    y_alt[::13] = np.nan
    sample_meta = pl.DataFrame(
        {
            "sample_id": pl.Series("sample_id", sample_ids, dtype=pl.Int64),
            "sample_key": [f"sample_{sample_id:06d}" for sample_id in range(n_samples)],
            "y": pl.Series("y", y, dtype=pl.Float64),
            "y_alt": pl.Series("y_alt", y_alt, dtype=pl.Float64),
            "y_const": pl.Series("y_const", y_const, dtype=pl.Float64),
        }
    )
    sample_meta_path = root / "sample_meta.parquet"
    sample_meta.write_parquet(sample_meta_path, compression="zstd")

    feature_ids = np.arange(n_features, dtype=np.int32)
    feature_meta = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
            "feature_key": [f"feature_{feature_id:06d}" for feature_id in range(n_features)],
        }
    )
    feature_meta_path = root / "feature_meta.parquet"
    feature_meta.write_parquet(feature_meta_path, compression="zstd")

    raw_paths: list[str] = []
    invalid_rows = 0
    present_rows = 0
    for sample_id in range(n_samples):
        present = rng.random(n_features) < float(density)
        if not bool(present.any()):
            present[int(rng.integers(0, n_features))] = True
        feature_id_values = np.flatnonzero(present).astype(np.int32, copy=False)
        values = rng.normal(loc=sample_id * 0.05, scale=5.0, size=feature_id_values.shape[0]).astype(np.float64)
        null_mask = ((feature_id_values % 29) == 0) & ((sample_id % 5) == 0)
        nan_mask = ((feature_id_values % 31) == 0) & ((sample_id % 7) == 0)
        value_list: list[float | None] = []
        for idx, value in enumerate(values):
            if bool(null_mask[idx]):
                value_list.append(None)
                invalid_rows += 1
            elif bool(nan_mask[idx]):
                value_list.append(float("nan"))
                invalid_rows += 1
            else:
                value_list.append(float(value))
                present_rows += 1
        path = raw_dir / f"sample_{sample_id:06d}.parquet"
        pl.DataFrame(
            {
                "sample_id": pl.Series("sample_id", np.full(feature_id_values.shape[0], sample_id, dtype=np.int64), dtype=pl.Int64),
                "feature_id": pl.Series("feature_id", feature_id_values, dtype=pl.Int32),
                "value": pl.Series("value", value_list, dtype=pl.Float64),
            }
        ).write_parquet(path, compression="zstd")
        raw_paths.append(str(path.relative_to(stage_dir)).replace("\\", "/"))

    if invalid_rows <= 0:
        raise AssertionError("parity fixture did not generate null/NaN raw rows")
    if present_rows <= 0:
        raise AssertionError("parity fixture did not generate finite raw rows")

    manifest_path = stage_dir / "sample_major_manifest.json"
    _write_json(
        manifest_path,
        {
            "format": "scalar-sample-major-v1",
            "version": 1,
            "sample_meta_path": "../sample_meta.parquet",
            "feature_meta_path": "../feature_meta.parquet",
            "sample_paths": raw_paths,
            "sample_ids": [int(sample_id) for sample_id in range(n_samples)],
            "sample_id_col": "sample_id",
            "feature_id_col": "feature_id",
            "value_col": "value",
        },
    )
    return manifest_path


def _build_python(stage_manifest: Path, out_dir: Path) -> Path:
    manifest_path = build_dense_long_shards_from_sample_major_manifest(
        str(stage_manifest),
        str(out_dir),
        target_part_bytes=1,
        stats_y_cols=list(STATS_Y_COLS),
        row_group_features=128,
        input_batch_files=7,
        compression="zstd",
    )
    return Path(manifest_path)


def _build_java(repo_root: Path, stage_manifest: Path, out_dir: Path) -> Path:
    _run(
        [
            _java_exe("java.exe" if os.name == "nt" else "java"),
            "-cp",
            _classpath(repo_root),
            "scripts.BuildShardsMain",
            "--sample-major-manifest",
            str(stage_manifest),
            "--out-dir",
            str(out_dir),
            "--target-shard-mb",
            "0",
            "--stats-y-col",
            "y",
            "--stats-y-col",
            "y_alt",
            "--stats-y-col",
            "y_const",
        ],
        cwd=repo_root,
    )
    return out_dir / "scalar_shard_manifest.json"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parts(manifest_path: Path) -> pl.DataFrame:
    manifest = _load_manifest(manifest_path)
    frames = []
    for part in manifest["parts"]:
        part_path = _resolve(manifest_path, str(part["path"]))
        frames.append(pl.read_parquet(part_path, columns=["feature_id", "sample_id", "mask", "value"]))
    if not frames:
        return pl.DataFrame(schema={"feature_id": pl.Int32, "sample_id": pl.Int64, "mask": pl.UInt8, "value": pl.Float64})
    return pl.concat(frames, how="vertical").sort(["feature_id", "sample_id"])


def _assert_float_equal(label: str, left: np.ndarray, right: np.ndarray, *, atol: float = 0.0):
    if atol == 0.0:
        equal = (left == right) | (np.isnan(left) & np.isnan(right))
    else:
        equal = np.isclose(left, right, rtol=0.0, atol=float(atol), equal_nan=True)
    if bool(np.all(equal)):
        return
    idx = int(np.flatnonzero(~equal)[0])
    raise AssertionError(f"{label} mismatch at row {idx}: left={left[idx]!r} right={right[idx]!r}")


def _assert_frame_equal(label: str, left: pl.DataFrame, right: pl.DataFrame, *, r2_atol: float = 0.0):
    if left.columns != right.columns:
        raise AssertionError(f"{label} columns differ: {left.columns} != {right.columns}")
    if left.height != right.height:
        raise AssertionError(f"{label} row count differs: {left.height} != {right.height}")
    for col in left.columns:
        left_values = left[col].to_numpy()
        right_values = right[col].to_numpy()
        if left[col].dtype.is_float():
            _assert_float_equal(label + "." + col, left_values.astype(np.float64), right_values.astype(np.float64), atol=r2_atol)
        elif not np.array_equal(left_values, right_values):
            idx = int(np.flatnonzero(left_values != right_values)[0])
            raise AssertionError(f"{label}.{col} mismatch at row {idx}: left={left_values[idx]!r} right={right_values[idx]!r}")


def _compare_outputs(py_manifest_path: Path, java_manifest_path: Path):
    py_parts = _load_parts(py_manifest_path)
    java_parts = _load_parts(java_manifest_path)
    _assert_frame_equal("parts", py_parts, java_parts)

    py_manifest = _load_manifest(py_manifest_path)
    java_manifest = _load_manifest(java_manifest_path)
    py_locator = pl.read_parquet(_resolve(py_manifest_path, str(py_manifest["feature_locator_path"]))).sort("feature_id")
    java_locator = pl.read_parquet(_resolve(java_manifest_path, str(java_manifest["feature_locator_path"]))).sort("feature_id")
    _assert_frame_equal("feature_locator", py_locator, java_locator)

    for y_col in STATS_Y_COLS:
        py_stats = pl.read_parquet(_resolve(py_manifest_path, str(py_manifest["selection_stats"][y_col]))).sort("feature_id")
        java_stats = pl.read_parquet(_resolve(java_manifest_path, str(java_manifest["selection_stats"][y_col]))).sort("feature_id")
        _assert_frame_equal(f"selection_stats.{y_col}", py_stats, java_stats, r2_atol=1e-10)


def main(argv=None):
    ap = ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--n-features", type=int, default=320)
    ap.add_argument("--density", type=float, default=0.62)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--skip-java-compile", action="store_true")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    root = Path(args.out_dir).resolve() if args.out_dir else repo_root / "data" / "tmp_scalar_python_java_parity_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    stage_manifest = _make_input(root, n_samples=int(args.n_samples), n_features=int(args.n_features), density=float(args.density))
    input_sec = time.perf_counter() - started

    if not args.skip_java_compile:
        compile_started = time.perf_counter()
        _compile_java(repo_root)
        compile_sec = time.perf_counter() - compile_started
    else:
        compile_sec = 0.0

    py_started = time.perf_counter()
    py_manifest = _build_python(stage_manifest, root / "python_shard")
    py_sec = time.perf_counter() - py_started

    java_started = time.perf_counter()
    java_manifest = _build_java(repo_root, stage_manifest, root / "java_shard")
    java_sec = time.perf_counter() - java_started

    validate_manifest(py_manifest, sample_major_manifest_path=stage_manifest, label="python parity shard", progress_every=0)
    validate_manifest(java_manifest, sample_major_manifest_path=stage_manifest, label="java parity shard", progress_every=0)
    _compare_outputs(py_manifest, java_manifest)

    total_sec = time.perf_counter() - started
    print(
        "scalar python/java parity tests passed "
        f"n_samples={int(args.n_samples)} n_features={int(args.n_features)} "
        f"density={float(args.density):.3f} input_sec={input_sec:.3f} "
        f"compile_sec={compile_sec:.3f} python_build_sec={py_sec:.3f} "
        f"java_build_sec={java_sec:.3f} total_sec={total_sec:.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
