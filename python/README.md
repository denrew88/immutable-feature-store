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

`sample(...)`가 필요한 이유:

- array는 sample 하나 안에 여러 feature trace가 들어가는 구조입니다.
- builder는 sample 경계를 알아야 다른 sample trace가 섞이는 실수를 막을 수 있습니다.
- 자동 checkpoint commit도 sample이 완전히 끝난 뒤에만 안전하게 할 수 있습니다.
- 그래서 array 쪽 sample context는 단순 편의 문법이 아니라, resume-safe ingestion 경계를 표시하는 public API입니다.

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
- 여러 trace를 한 sample로 묶어 쓰는 것이 기본 경로이므로, 보통은 `with session.sample(...):` 형태를 권장합니다.
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

## Array Sample Parquet

`array_sample_parquet` 패키지는 기존 custom binary shard와 별개인 viewer/debugging용 sample-major Parquet 포맷입니다.

- 최종 `sample_parts/*.parquet`의 행 하나는 trace point 하나입니다.
- point part는 `sample_id`, `feature_id`, `point_idx`, point primitive columns로 구성됩니다.
- present trace 목록과 길이는 `trace_index_parts/*.parquet`에 `(sample_id, feature_id, trace_len)`으로 따로 저장됩니다.
- raw builder는 sample 하나를 `raw_samples/sample_*.parquet`와 `raw_trace_index/sample_*.parquet`로 commit하므로 sample 순서와 무관하게 재개할 수 있습니다.
- part 크기는 sample 개수가 아니라 `target_part_bytes`, `max_part_rows`, `max_part_samples` 기준으로 자동 조절합니다.
- categorical point column은 별도 dictionary sidecar 없이 string column으로 저장하고 Parquet writer의 dictionary/RLE encoding에 맡깁니다.
- 권장 API 서버 endpoint는 `/array-sample-parquet/schema`, `/array-sample-parquet/traces`입니다.

```python
from array_sample_parquet import ArraySampleParquetRawDatasetBuilder, open_array_sample_parquet

with ArraySampleParquetRawDatasetBuilder.open_session(
    out_dir="..\\data\\array_sample_parquet",
    sample_meta_path="..\\data\\array_sample_meta.parquet",
    point_schema=point_schema,
    feature_meta_path="..\\data\\array_feature_meta.parquet",
) as session:
    for sample_id in session.pending_sample_ids():
        with session.sample(sample_id=sample_id, skip_if_completed=True) as sample:
            sample.add_trace(feature_key="feature_a", columns=columns)
    manifest_path = session.compact()

reader = open_array_sample_parquet(manifest_path)
payload = reader.get_traces_json(
    sample_keys=["sample_000001"],
    feature_keys=["feature_a"],
    decode_categorical=True,
    layout="nested",
)
```

추가 테스트:

```powershell
python -m scripts.run_array_sample_parquet_tests
```

추가 문서와 패키지:

- [../docs/array_sample_parquet_format_v1.md](../docs/array_sample_parquet_format_v1.md)
- [../packages/array_sample_parquet/README.md](../packages/array_sample_parquet/README.md)

## 조회 API 서버

패키지 기준 조회 서버는 `python/scripts/serve_feature_query_api.py`입니다.

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 endpoint:

- `POST /array-sample-parquet/schema`
- `POST /array-sample-parquet/traces`
- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

scalar endpoint는 blob shard와 dense-long shard를 모두 지원합니다. 요청에서 `feature_ids`와 `feature_keys`, `sample_ids`와 `sample_keys`는 각각 둘 중 하나만 지정해야 합니다.

Python 패키지 예제:

- [../packages/array_sample_parquet/examples/build_array_sample_parquet_example.py](../packages/array_sample_parquet/examples/build_array_sample_parquet_example.py)
- [../packages/scalar_feature_shard/examples/build_scalar_dense_long_example.py](../packages/scalar_feature_shard/examples/build_scalar_dense_long_example.py)

## Synthetic Value API

`python/scripts/serve_synthetic_value_api.py`는 shard reader가 아니라 ingestion source 예제입니다.
sample metadata와 feature metadata는 이미 parquet로 존재한다고 가정하고, Java builder가 필요한 값만 HTTP로 요청합니다.

실행:

```powershell
python python\scripts\serve_synthetic_value_api.py --host 127.0.0.1 --port 8010
```

Scalar endpoint:

- `POST /scalar/values`
- 요청은 `sample_meta_path`, `feature_meta_path`, `sample_id` 또는 `sample_key`, `feature_ids` 또는 `feature_keys` 중 하나를 받습니다.
- 응답은 `values: [{feature_id, feature_key, present, value}]` 형태입니다.

Array endpoint:

- `POST /array/traces`
- 요청 규칙은 scalar와 같고, 응답은 `traces: [{feature_id, feature_key, present, trace_len, columns}]` 형태입니다.
- 기본 point schema는 `time: float64`, `value: float64`, `ch_step: string categorical`입니다.

이 서버는 deterministic synthetic 값을 만들기 때문에 같은 metadata, seed, sample/feature id 조합이면 항상 같은 값을 반환합니다.
