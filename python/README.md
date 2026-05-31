# Python 구현 안내

이 디렉터리는 core Python 구현, 테스트 스크립트, API 서버 예제를 담고 있습니다.

현재 scalar 최종 shard는 dense-long 포맷만 지원합니다. array binary의 `open_shard(...)` API는 그대로 유지됩니다.

## 주요 진입점

- `fs.array`: array binary shard core wrapper
- `fs.array_sample_parquet`: array sample-major Parquet wrapper
- `fs.scalar`: scalar dense-long wrapper
- `scripts/serve_feature_query_api.py`: 권장 조회 API 서버

## Scalar Dense-Long

순차 builder는 sample을 `status().next_expected_sample_id`부터 순서대로 받습니다.

```python
from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder, open_dense_long_shard, write_feature_meta, write_sample_meta

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "y": 1.0},
        {"sample_key": "sample_000001", "y": 2.0},
    ],
    "..\\data\\scalar_sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a"},
        {"feature_key": "feature_b"},
    ],
    "..\\data\\scalar_feature_meta.parquet",
)

session = ScalarDatasetBuilder.open_session(
    out_dir="..\\data\\scalar_dense_long",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=ScalarShardBuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
)

for sample_id in range(session.status().next_expected_sample_id, 2):
    session.write_sample(sample_id, {"feature_a": 1.23, "feature_b": None})

manifest_path = session.build_shards()

with open_dense_long_shard(manifest_path) as ds:
    values, valid = ds.load_feature_by_key("feature_a")
```

random-order 생성이 필요하면 package API의 `ScalarRawDatasetBuilder`를 사용합니다. sample별 raw parquet를 먼저 만들고, 마지막에 dense-long shard로 묶습니다.

## Array Binary

array binary shard는 custom binary serving 포맷입니다.

```python
from fs.array import open_shard

with open_shard("..\\data\\array_shards\\array_binary_shard_manifest.json") as ds:
    traces = ds.load_feature_samples_by_ids(0, [0, 1])
```

## Array Sample Parquet

array sample parquet는 viewer/debugging용 sample-major long Parquet 포맷입니다. `packages/array_sample_parquet`의 public API를 권장합니다.

## API Server

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

## Tests

```powershell
python python\scripts\run_scalar_package_tests.py
python python\scripts\run_scalar_api_tests.py
python python\scripts\run_selection_api_tests.py
```

array 테스트는 기존 `run_array_*` 스크립트를 사용합니다.
