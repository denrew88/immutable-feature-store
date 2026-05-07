# Python 구현 안내

이 디렉터리는 array/scalar shard의 Python 구현, 테스트 스크립트, 그리고 예제 서버 코드를 담고 있습니다.

최근 builder는 array/scalar 모두 **resumable session** 모델로 정리되어 있습니다.

- 자동 chunking은 내부 구현입니다.
- committed bundle parquet 하나가 durable checkpoint 하나입니다.
- resume는 `status().next_expected_sample_id`를 기준으로 합니다.

## 주요 진입점

- scalar
  - `fs.scalar`
  - `scripts/build_shards.py`
  - `scripts/run_selection.py`
- array
  - `fs.array`
  - `scripts/build_array_shards.py`
  - `scripts/build_array_binary_shards.py`
- 테스트
  - `scripts/run_scalar_builder_tests.py`
  - `scripts/run_scalar_package_tests.py`
  - `scripts/run_array_binary_storage_tests.py`
  - `scripts/run_array_binary_package_tests.py`

## 실행 위치

보통 `python/` 아래에서 실행합니다.

```powershell
cd python
```

## Scalar 사용법

### builder session

scalar는 public write 단위를 sample 하나로 고정합니다.

- `ScalarDatasetBuilder.open_session(...)`
- `status()`
- `write_sample(sample_id, values)`
- `finish_stage()`
- `build_shards(...)`

예:

```python
from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder, write_feature_meta, write_sample_meta

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "y": 1.0, "y_alt": 1.5},
        {"sample_key": "sample_000001", "y": 2.0, "y_alt": 2.5},
    ],
    "..\\data\\scalar_sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    "..\\data\\scalar_feature_meta.parquet",
)

session = ScalarDatasetBuilder.open_session(
    out_dir="..\\data\\scalar_shards",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=ScalarShardBuildOptions(
        target_shard_mb=32,
        stats_y_cols=("y", "y_alt"),
    ),
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

- `open_sample()` public path는 비활성화되어 있습니다.
- 같은 sample을 다시 쓰지 않고, 항상 `next_expected_sample_id`부터 이어서 넣습니다.
- `finish_sample_major()`는 legacy alias로 남아 있지만 새 코드는 `finish_stage()`를 권장합니다.

### reader / selection

```python
from fs.scalar import open_shard, select_features

with open_shard("..\\data\\scalar_shards\\shard_manifest.json") as ds:
    value = ds.get_value(feature_id=0, sample_id=0)
    batch = ds.get_values(feature_id=0, sample_ids=[0, 1])

result = select_features(
    manifest_path="..\\data\\scalar_shards\\shard_manifest.json",
    y_col="y",
    top_m=100,
)
```

## Array 사용법

### builder session

array는 sample context 안에서 trace를 추가합니다.

- `ArrayDatasetBuilder.open_session(...)`
- `status()`
- `sample(sample_id=...)`
- sample context 안에서 `add_trace(...)`
- `finish_stage()`
- `build_shards(...)`

예:

```python
from fs.array import (
    ArrayDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
    write_feature_meta,
    write_sample_meta,
)
from fs.config import ArrayBinaryBuildOptions

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "split": "train"},
        {"sample_key": "sample_000001", "split": "test"},
    ],
    "..\\data\\array_sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
    ],
    "..\\data\\array_feature_meta.parquet",
)

session = ArrayDatasetBuilder.open_session(
    out_dir="..\\data\\array_shards",
    sample_meta_path=sample_meta_path,
    point_schema=[
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ],
    feature_meta_path=feature_meta_path,
    build_options=ArrayBinaryBuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
)

st = session.status()
for sample_id in range(st.next_expected_sample_id, 2):
    with session.sample(sample_id=sample_id) as sample:
        sample.add_trace(
            feature_key="feature_a",
            columns={
                "phase": [10, 11, 12],
                "state_code": ["OK", "OK", "WARN"],
            },
        )

session.finish_stage()
manifest_path = session.build_shards(cleanup_bundles=False)
```

중요:

- array는 sample 경계에서만 checkpoint가 생깁니다.
- top-level `add_trace(...)`는 같은 sample 안에서만 연속 호출해야 합니다.
- `finish_bundles()`는 legacy alias로 남아 있지만 새 코드는 `finish_stage()`를 권장합니다.

### reader

```python
from fs.array import open_shard

with open_shard("..\\data\\array_shards\\array_binary_shard_manifest.json") as ds:
    trace = ds.get_trace(feature_id=0, sample_id=0)
    traces = ds.get_traces(feature_id=0, sample_ids=[0, 1])
```

## 테스트

```powershell
python -m scripts.run_tests --seed 0
python -m scripts.run_scalar_api_tests
python -m scripts.run_selection_api_tests
python -m scripts.run_array_storage_tests
python -m scripts.run_array_binary_storage_tests
python -m scripts.run_array_api_tests
python -m scripts.run_scalar_builder_tests
python -m scripts.run_scalar_package_tests
python -m scripts.run_array_binary_package_tests
```

## 참고

- array 포맷: [../docs/array_binary_shard_format_v3.md](../docs/array_binary_shard_format_v3.md)
- scalar 포맷: [../docs/scalar_parquet_shard_format.md](../docs/scalar_parquet_shard_format.md)
- Python wheel/package 사용법:
  - [../packages/array_binary_shard/README.md](../packages/array_binary_shard/README.md)
  - [../packages/scalar_feature_shard/README.md](../packages/scalar_feature_shard/README.md)
