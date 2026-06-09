# array-binary-shard

`array-binary-shard`는 dense-id 기반 array custom binary shard v3를 만들고 읽는 Python 패키지입니다. 조회 속도는 빠르지만 포맷 유지보수 비용이 있으므로, viewer/debugging 목적이면 `array-sample-parquet`를 먼저 고려하십시오.

## Public API

- `ArrayDatasetBuilder`: sample context 기반 resumable builder입니다.
- `build_shard(...)`: bundle manifest에서 최종 custom binary shard를 만듭니다.
- `open_shard(...)`, `BinaryShardDataset`: reader API입니다.
- `BuildOptions`: direct builder와 `build_shard(...)`에서 쓰는 public build 설정입니다.
- `PointColumnSpec`, `StorageType`, `LogicalType`: point schema 정의 helper입니다.
- `write_sample_meta(...)`, `write_feature_meta(...)`: dense metadata parquet 작성 helper입니다.

## ID 규칙

- `sample_id == sample_meta.parquet` row index
- `feature_id == feature_meta.parquet` row index
- key 조회가 필요하면 metadata에 `sample_key`, `feature_key` 같은 external key column을 둡니다.
- builder는 resumable session 모델을 사용합니다.

## Build

```bash
cd packages/array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## Config Guide

처음에는 아래 설정만 넣으면 됩니다.

```python
BuildOptions(samples_per_block=16, target_shard_mb=32, codec="none")
```

| option | 기본값 | 설명 |
| --- | --- | --- |
| `samples_per_block` | 16 | feature 하나를 sample 축으로 자르는 block 크기입니다. 작을수록 부분 조회 낭비가 줄지만 index overhead가 늘어납니다. |
| `target_shard_mb` | 32 | shard 하나의 목표 크기입니다. shard가 너무 많으면 키우고, 한 shard가 너무 크면 줄입니다. |
| `n_shards` | `None` | shard 개수를 직접 고정할 때만 넣습니다. 보통 자동 분할을 둡니다. |
| `codec` | `"none"` | payload codec입니다. 현재는 유지보수성과 속도 때문에 `"none"`을 권장합니다. |
| `zstd_level` | 3 | `codec="zstd"`일 때만 쓰는 압축 level입니다. |
| `sample_key_col` | `"sample_key"` | sample metadata의 key column 이름이 다를 때만 바꿉니다. |
| `feature_key_col` | `"feature_key"` | feature metadata의 key column 이름이 다를 때만 바꿉니다. |

`ArrayBundleConfig`와 `ArrayShardConfig`는 bundle-to-shard 변환을 직접 호출할 때 쓰는 low-level 설정입니다. direct builder를 쓰는 일반 경로에서는 `BuildOptions`만 넣으면 됩니다.

## Bundle Manifest에서 Shard 만들기

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

## Builder Session

array 입력은 sample 하나 안에 여러 feature trace가 들어가므로 sample context 안에서 trace를 추가합니다.

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

주의할 점:

- checkpoint는 sample 경계에서만 생성됩니다.
- 같은 sample의 trace는 `with session.sample(...):` 안에서 모두 넣는 경로를 권장합니다.
- `finish_bundles()`는 legacy alias입니다. 새 코드는 `finish_stage()`를 사용하십시오.

## File Replace Retry

custom binary builder는 동시 worker writer용 표준 경로가 아니므로 `.lock` 기반 병렬 commit 모델은 제공하지 않습니다. 다만 Windows에서 IDE, 백신, 인덱서가 JSON 또는 bundle parquet 파일 핸들을 짧게 잡는 경우를 대비해 state JSON, dictionary JSON, bundle parquet commit은 unique tmp 파일과 짧은 replace retry를 사용합니다.

## Reader

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    trace = ds.get_trace(feature_id=123, sample_id=1001)
    by_key = ds.get_trace_by_key("feature_000123", "sample_001001")
```

## Reference

- 포맷 상세: [../../docs/array_binary_shard_format_v3.md](../../docs/array_binary_shard_format_v3.md)
- core 사용법: [../../python/README.md](../../python/README.md)
