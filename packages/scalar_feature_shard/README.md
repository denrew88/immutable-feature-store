# scalar-feature-shard

scalar feature shard를 만들고 읽고 selection까지 돌릴 수 있는 독립 Python 패키지입니다.

## 개요

이 패키지는 dense id 기반 scalar shard를 다룹니다.

핵심 규칙:

- `sample_id`는 `sample_meta.parquet`의 dense row index
- `feature_id`는 `feature_meta.parquet`의 dense row index

external key는 metadata에 둡니다.

- `sample_key`
- `feature_key`

최종 artifact는 standalone 폴더 하나로 이동할 수 있다.

```text
scalar_shard_dataset/
  shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  selection_stats/
  feature_shards/
```

## public API

- reader
  - `open_shard(...)`
  - `ScalarShardDataset`
- writer
  - `ScalarDatasetBuilder`
  - `build_shard(...)`
- selection
  - `select_features(...)`
  - `run_selection(...)`
- metadata helper
  - `write_sample_meta(...)`
  - `write_feature_meta(...)`

## 빌드

```bash
cd packages/scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## direct-ingestion builder

known-feature mode:

```python
from scalar_feature_shard import (
    BuildOptions,
    ScalarDatasetBuilder,
    write_feature_meta,
    write_sample_meta,
)

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "y": 1.0, "y_alt": 1.5},
        {"sample_key": "sample_000001", "y": 2.0, "y_alt": 2.5},
    ],
    ".../sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    ".../feature_meta.parquet",
)

builder = ScalarDatasetBuilder(
    out_dir=".../scalar_shards",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y", "y_alt")),
)

builder.write_sample(0, {"feature_a": 1.23, "feature_b": 4.56})
builder.write_sample(1, {"feature_a": 7.89})
manifest_path = builder.build_shards()
```

discovered-feature mode:

```python
from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder

with ScalarDatasetBuilder(
    out_dir=".../scalar_shards",
    sample_meta_path=".../sample_meta.parquet",
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

## sample-major stage

public builder는 intermediate sample-major stage를 명시적으로 노출한다.

- `finish_sample_major()`
- `build_shards(keep_sample_major=False)`

현재 intermediate 표현은 file-per-sample이 아니라 bundle 기반입니다.

```text
sample_major_stage/
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  sample_bundles/
    bundle_000000.parquet
    bundle_000001.parquet
    ...
```

기본값은 `keep_sample_major=False`라서 최종 shard를 만든 뒤 intermediate stage를 지웁니다.

## 조회

```python
from scalar_feature_shard import open_shard

with open_shard(".../shard_manifest.json") as ds:
    value = ds.get_value(feature_id=123, sample_id=10)
    batch = ds.get_values(feature_id=123, sample_ids=[10, 11, 12])
```

key 기반 조회도 가능하다.

```python
from scalar_feature_shard import open_shard

with open_shard(".../shard_manifest.json") as ds:
    value = ds.get_value_by_key("feature_000123", "sample_000010")
```

## selection

```python
from scalar_feature_shard import select_features

result = select_features(
    manifest_path=".../shard_manifest.json",
    y_col="y",
    top_m=100,
)
```

manifest에 해당 `y_col`의 `selection_stats/<y>.parquet`가 있으면 fast path를 쓰고, 없으면 shard를 다시 읽는 fallback을 탑니다.
