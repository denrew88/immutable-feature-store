# `scalar_feature_shard`

Standalone dense-id reader, writer, and selector for scalar feature shards.

This package wraps the scalar shard format used in this repository behind a small public API:

- `open_shard(...)`
- `ScalarDatasetBuilder`
- `build_shard(...)`
- `select_features(...)`
- `write_sample_meta(...)`
- `write_feature_meta(...)`

## Install

Build a wheel locally:

```powershell
cd packages\scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## Metadata helpers

```python
from scalar_feature_shard import write_feature_meta, write_sample_meta

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "y": 1.0, "y_alt": 1.5},
        {"sample_key": "sample_000001", "y": 2.0, "y_alt": 2.5},
    ],
    "sample_meta.parquet",
)

feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    "feature_meta.parquet",
)
```

## Direct ingestion

```python
from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder

builder = ScalarDatasetBuilder(
    out_dir="scalar_shards",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=BuildOptions(
        target_shard_mb=32,
        stats_y_cols=("y", "y_alt"),
    ),
)

builder.write_sample(0, {"feature_a": 1.23, "feature_b": 4.56})
builder.write_sample(1, {"feature_a": 7.89})
manifest_path = builder.build_shards()
```

The direct-ingestion builder is intentionally sample-scoped:

- `write_sample(sample_id, values)` writes one complete sample and flushes it immediately.
- `open_sample(sample_id)` returns a sample-scoped context for incremental writes.
- samples may be written in any order
- the same `sample_id` may only be written once

```python
with ScalarDatasetBuilder(
    out_dir="scalar_shards",
    sample_meta_path=sample_meta_path,
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
) as builder:
    with builder.open_sample(0) as sample:
        sample.write_value("feature_x", 1.23)
        sample.write_value("feature_y", 4.56)

    with builder.open_sample(1) as sample:
        sample.write_values({"feature_y": 7.89, "feature_z": 0.12})

    builder.finish_sample_major()
    builder.update_feature_meta(
        [
            {"feature_key": "feature_x", "group": "alpha"},
            {"feature_key": "feature_y", "group": "beta"},
            {"feature_key": "feature_z", "group": "gamma"},
        ],
        require_all=True,
    )
    manifest_path = builder.build_shards(keep_sample_major=True)
```

## Build from sample-major files

If sample-major parquet files already exist, use the thin wrapper:

```python
from scalar_feature_shard import BuildOptions, build_shard

manifest_path = build_shard(
    sample_meta_path,
    "scalar_shards",
    feature_meta_path=feature_meta_path,
    options=BuildOptions(target_shard_mb=32, stats_y_cols=("y", "y_alt")),
)
```

## Read values

```python
from scalar_feature_shard import open_shard

with open_shard(manifest_path) as ds:
    value = ds.get_value(feature_id=0, sample_id=0)
    batch = ds.get_values(feature_id=0, sample_ids=[0, 1, 3])
    keyed = ds.get_value_by_key(feature_key="feature_a", sample_key="sample_000000")
    result = ds.get_many(feature_ids=[0, 1], sample_ids=[0, 1, 3])
```

Each `ScalarValue` contains:

- `feature_id`
- `sample_id`
- `present`
- `value`
- optional `feature_key`
- optional `sample_key`

## Run selection

```python
from scalar_feature_shard import SelectionOptions, select_features

result = select_features(
    manifest_path,
    options=SelectionOptions(
        y_col="y_alt",
        top_m=32,
        min_non_null_y=20,
        min_non_null_pair=20,
    ),
    include_candidates=True,
)

print(result.selected_feature_ids)
print(result.selected_feature_keys)
```

`select_features(...)` uses:

- `selection_stats/<y>.parquet` fast-path when present
- otherwise shard scan fallback

## Public API

- `BuildOptions`
- `SelectionOptions`
- `ScalarDatasetBuilder`
- `open_shard(...)`
- `build_shard(...)`
- `select_features(...)`
- `run_selection(...)`
- `write_sample_meta(...)`
- `write_feature_meta(...)`
