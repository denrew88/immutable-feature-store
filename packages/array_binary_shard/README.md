# array-binary-shard

dense-id 기반 array binary shard v3를 위한 Python 패키지입니다.

## 포함 API

- reader
  - `open_shard(...)`
  - `BinaryShardDataset`
- builder
  - `ArrayDatasetBuilder`
  - `build_shard(...)`
- schema helper
  - `PointColumnSpec`
  - `StorageType`
  - `LogicalType`
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
cd packages/array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## bundle manifest에서 최종 shard 만들기

```python
from array_binary_shard import BuildOptions, build_shard

manifest_path = build_shard(
    source=".../array_bundle_manifest.json",
    out_dir=".../array_binary_shards",
    options=BuildOptions(
        samples_per_block=16,
        target_shard_mb=32,
        codec="none",
        sample_key_col="sample_key",
        feature_key_col="feature_key",
    ),
)
```

## builder session

array는 sample context 안에서 trace를 추가합니다.

- `ArrayDatasetBuilder.open_session(...)`
- `status()`
- `sample(sample_id=...)`
- sample context 안에서 `add_trace(...)`
- `finish_stage()`
- `build_shards(...)`

`sample(...)`가 필요한 이유:

- array 입력은 sample 하나 안에 trace 여러 개가 들어가는 형태가 자연스럽습니다.
- builder는 이 sample 경계를 기준으로 trace 묶음을 닫고, 그 뒤에만 checkpoint commit을 할 수 있습니다.
- 그래서 sample context는 단순 편의 API가 아니라 resume-safe array ingestion의 기본 경계입니다.

예:

```python
from array_binary_shard import (
    ArrayDatasetBuilder,
    BuildOptions,
    LogicalType,
    PointColumnSpec,
    StorageType,
    write_feature_meta,
    write_sample_meta,
)

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "split": "train"},
        {"sample_key": "sample_000001", "split": "test"},
    ],
    ".../sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
    ],
    ".../feature_meta.parquet",
)

session = ArrayDatasetBuilder.open_session(
    out_dir=".../array_binary_shards",
    sample_meta_path=sample_meta_path,
    point_schema=[
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ],
    feature_meta_path=feature_meta_path,
    build_options=BuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
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

- array checkpoint는 sample 경계에서만 생성됩니다.
- top-level `add_trace(...)`는 같은 sample 안에서만 연속 호출해야 합니다.
- 일반적으로는 `with session.sample(...):` 안에서 trace를 모두 넣는 경로를 권장합니다.
- `finish_bundles()`는 legacy alias로 남아 있지만 새 코드는 `finish_stage()`를 권장합니다.

## reader

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    trace = ds.get_trace(feature_id=123, sample_id=1001)
    by_key = ds.get_trace_by_key("feature_000123", "sample_001001")
```

## 참고

- 포맷 상세: [../../docs/array_binary_shard_format_v3.md](../../docs/array_binary_shard_format_v3.md)
- core 사용법: [../../python/README.md](../../python/README.md)
