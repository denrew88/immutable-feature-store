# scalar-feature-shard

dense-id 기반 scalar shard를 만들고 읽고 selection까지 수행하는 Python 패키지입니다.

## 포함 API

- reader
  - `open_shard(...)`
  - `ScalarShardDataset`
- builder
  - `ScalarDatasetBuilder`
  - `build_shard(...)`
- selection
  - `select_features(...)`
  - `run_selection(...)`
- metadata helper
  - `write_sample_meta(...)`
  - `write_feature_meta(...)`

## 핵심 규칙

- `sample_id == sample_meta.parquet` row index
- `feature_id == feature_meta.parquet` row index
- builder는 **resumable session** 모델을 사용합니다.
- 자동 chunking은 내부 구현입니다.
- resume는 `status().next_expected_sample_id`를 기준으로 합니다.

## 빌드

```bash
cd packages/scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## builder session

scalar는 public write 단위를 sample 하나로 고정합니다.

- `ScalarDatasetBuilder.open_session(...)`
- `status()`
- `write_sample(sample_id, values)`
- `finish_stage()`
- `build_shards(...)`

예:

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

session = ScalarDatasetBuilder.open_session(
    out_dir=".../scalar_shards",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y", "y_alt")),
)

st = session.status()
for sample_id in range(st.next_expected_sample_id, 2):
    if sample_id == 0:
        session.write_sample(sample_id, {"feature_a": 1.23, "feature_b": 4.56})
    else:
        session.write_sample(sample_id, {"feature_a": 7.89})

session.finish_stage()
manifest_path = session.build_shards(keep_sample_major=False)
```

중요:

- `open_sample()`, `write_value()` 같은 per-value public path는 비활성화되어 있습니다.
- 같은 sample을 다시 쓰지 말고, 항상 `status().next_expected_sample_id`부터 이어서 넣습니다.
- `finish_sample_major()`는 legacy alias로 남아 있지만 새 코드는 `finish_stage()`를 권장합니다.

## reader

```python
from scalar_feature_shard import open_shard

with open_shard(".../shard_manifest.json") as ds:
    value = ds.get_value(feature_id=123, sample_id=10)
    batch = ds.get_values(feature_id=123, sample_ids=[10, 11, 12])
    by_key = ds.get_value_by_key("feature_000123", "sample_000010")
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

## 참고

- 포맷 상세: [../../docs/scalar_parquet_shard_format.md](../../docs/scalar_parquet_shard_format.md)
- core 사용법: [../../python/README.md](../../python/README.md)
