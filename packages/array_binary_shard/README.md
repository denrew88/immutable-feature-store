# array-binary-shard

dense-id 기반 array binary shard 포맷을 위한 독립 Python 패키지입니다.

## 개요

이 패키지는
[docs/array_binary_shard_format_v3.md](../../docs/array_binary_shard_format_v3.md)
에 정의된 v3 포맷을 구현한다.

핵심 규칙:

- `sample_id`는 `sample_meta.parquet`의 dense row index
- `feature_id`는 `feature_meta.parquet`의 dense row index

선택적으로 외부 식별자를 metadata에 둘 수 있다.

- `sample_key`
- `feature_key`

즉 이 패키지는 다음 둘 다 지원한다.

- dense id 기반 빠른 조회
- external key 기반 편의 조회

manifest마다 point schema는 고정이지만, manifest가 다르면 schema는 달라도 된다.
v3에서는 `time`, `value`가 더 이상 필수 컬럼이 아닙니다.

categorical point column은 shard 내부에 integer code로 저장되고, 필요하면 sidecar dictionary JSON을 통해 원래 문자열로 되돌릴 수 있다.

최종 산출물은 self-contained dataset 디렉터리이다. manifest는 다음 경로를 relative path로 저장한다.

- `sample_meta.parquet`
- `feature_meta.parquet`
- `array_binary_feature_shards/`

즉 출력 폴더 전체를 artifact 하나처럼 이동할 수 있다.

## public API

- reader
  - `open_shard(...)`
  - `BinaryShardDataset`
- writer
  - `build_shard(...)`
  - `ArrayDatasetBuilder`
- schema enum
  - `StorageType`
  - `LogicalType`
  - `PointColumnSpec`

## 빌드

```bash
cd packages/array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## bundle에서 shard 만들기

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

public API에서는 spill 옵션을 노출하지 않는다. 내부적으로는 append-only spill 경로를 사용한다.

## trace에서 바로 만들기

사용자 입력에서는 sample-major parquet를 직접 조립하기보다 `ArrayDatasetBuilder`를 쓰는 편이 낫습니다.

known-feature mode:

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
        {"feature_key": "feature_b", "group": "beta"},
    ],
    ".../feature_meta.parquet",
)

builder = ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    sample_meta_path=sample_meta_path,
    point_schema=[
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ],
    feature_meta_path=feature_meta_path,
    build_options=BuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
)

builder.add_trace(
    sample_id=0,
    feature_key="feature_a",
    columns={
        "phase": [10, 11, 12],
        "state_code": ["OK", "OK", "WARN"],
    },
)
manifest_path = builder.build_shards()
```

discovered-feature mode:

```python
from array_binary_shard import ArrayDatasetBuilder, LogicalType, StorageType

with ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    sample_meta_path=".../sample_meta.parquet",
    point_schema=[
        {"name": "phase", "storage_type": StorageType.INT32, "logical_type": LogicalType.INTEGER},
        {"name": "state_code", "storage_type": StorageType.UINT32, "logical_type": LogicalType.CATEGORICAL},
    ],
) as builder:
    with builder.sample(sample_id=0) as sample:
        sample.add_trace(
            feature_key="feature_a",
            columns={
                "phase": [10, 11, 12],
                "state_code": ["OK", "OK", "WARN"],
            },
        )
```

categorical point column은 문자열 label로 넣어도 되고, builder가 자동으로 integer code를 부여한 뒤 dictionary JSON을 같이 만든다.

temporal point column도 지원한다.

- `logical_type="timestamp_ns"`: `datetime64[ns]`
- `logical_type="timedelta_ns"`: `timedelta64[ns]`

저장 자체는 raw nanosecond `int64`로 하고, public reader가 NumPy temporal array로 복원한다.

## intermediate bundle stage

builder는 intermediate bundle stage를 명시적으로 노출한다.

```python
builder = ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    bundle_out_dir=".../array_bundle_stage",
    sample_meta_path=".../sample_meta.parquet",
    point_schema=[
        {"name": "phase", "storage_type": "int32", "logical_type": "integer"},
    ],
    feature_keys=["feature_a"],
)

builder.add_trace(
    sample_id=0,
    feature_key="feature_a",
    columns={"phase": [10, 11, 12]},
)

bundle_manifest_path = builder.finish_bundles()
manifest_path = builder.build_shards(cleanup_bundles=False)
```

의미:

- `finish_bundles()`
  - intermediate bundle artifact를 확정
- `build_shards()`
  - 최종 binary shard artifact 생성
- `cleanup_bundles=True`
  - shard 생성 후 bundle stage 삭제

discovered-feature mode에서 자동 생성되는 `feature_meta.parquet`는 기본적으로 `feature_id`, `feature_key`만 갖는다. richer metadata가 필요하면 known-feature mode를 쓰거나, bundle stage 이후 `update_feature_meta(...)`로 보강하면 된다.

## 결과 디렉터리

```text
array_binary_shards/
  array_binary_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  array_binary_feature_shards/
    shard_0000.blocks.idx
    shard_0000.blocks.bin
    ...
```

## dense id로 읽기

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    schema = ds.point_schema
    trace = ds.get_trace(feature_id=123, sample_id=1001)
    batch = ds.get_traces(feature_id=123, sample_ids=[1001, 1007, 1015])
```

## external key로 읽기

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    trace = ds.get_trace_by_key(
        feature_key="feature_000123",
        sample_key="sample_001001",
    )
```

## categorical decode

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    dictionaries = ds.categorical_dictionaries()
    trace = ds.get_trace(
        feature_id=123,
        sample_id=1001,
        decode_categorical=True,
    )
```

## 참고

- dense id가 가장 빠른 경로입니다.
- key 기반 조회는 metadata dictionary를 lazy load한 뒤 같은 dense-id 경로를 탑니다.
- 권장 기본 codec은 `none`입니다.
- canonical 응답 형태는 `trace.columns`입니다.
