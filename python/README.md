# Python 구현 안내

이 디렉터리에는 core Python wrapper, 테스트 스크립트, API 서버 예제가 들어 있습니다.

주요 진입점:

- `fs.array`: array binary shard wrapper
- `fs.array_sample_parquet`: array sample-major Parquet wrapper
- `fs.scalar`: scalar dense-long wrapper
- `scripts/serve_feature_query_api.py`: 권장 조회 API 서버

## Config Quick Start

처음에는 아래 설정만 넣으면 됩니다.

```python
# scalar dense-long
ScalarShardBuildOptions(target_shard_mb=32, stats_y_cols=("y",))

# array sample parquet
ArraySampleParquetBuildOptions(target_part_bytes=128 * 1024 * 1024, compression="zstd")

# array custom binary
ArrayBinaryBuildOptions(samples_per_block=16, target_shard_mb=32, codec="none")
```

- scalar의 `target_shard_mb`는 dense-long part 목표 크기이고, `stats_y_cols`는 selection stats를 만들 target column입니다.
- array sample parquet의 `target_part_bytes`는 final parquet part 목표 크기입니다.
- array custom binary의 `samples_per_block`은 feature 하나를 sample 축으로 자르는 block 크기입니다.
- `sample_key_col`, `feature_key_col`은 metadata column 이름이 기본값과 다를 때만 바꿉니다.

## Scalar Dense-Long

scalar는 `ScalarDatasetBuilder` 하나를 표준 builder로 사용합니다. sample별 raw parquet를 먼저 commit하고, 마지막에 dense-long shard로 materialize합니다. sample id 순서는 강제하지 않습니다.

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

with ScalarDatasetBuilder.open_session(
    out_dir="..\\data\\scalar_stage",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=ScalarShardBuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
) as session:
    for sample_id in session.pending_sample_ids():
        session.write_sample(sample_id, {"feature_a": 1.23, "feature_b": None}, skip_if_completed=True)
    manifest_path = session.build_shards(require_all=True)

with open_dense_long_shard(manifest_path) as ds:
    values, valid = ds.load_feature_by_key("feature_a")
```

중간에 중단되면 같은 `out_dir`로 session을 다시 열고 `pending_sample_ids()`만 이어서 처리하면 됩니다.

## Array Binary

array binary shard는 custom binary serving 포맷입니다.

```python
from fs.array import open_shard

with open_shard("..\\data\\array_shards\\array_binary_shard_manifest.json") as ds:
    traces = ds.load_feature_samples_by_ids(0, [0, 1])
```

## Array Sample Parquet

array sample parquet는 viewer/debugging용 sample-major long Parquet 포맷입니다. 표준 builder는 scalar와 마찬가지로 sample별 raw parquet를 먼저 만들고 `finish()` 또는 `compact()`로 final part를 만듭니다.

```python
from fs.array_sample_parquet import ArraySampleParquetDatasetBuilder

with ArraySampleParquetDatasetBuilder.open_session(
    out_dir="..\\data\\array_sample_parquet",
    sample_meta_path="..\\data\\sample_meta.parquet",
    point_schema=point_schema,
    feature_meta_path="..\\data\\feature_meta.parquet",
) as builder:
    for sample_id in builder.pending_sample_ids():
        with builder.sample(sample_id=sample_id, skip_if_completed=True) as sample:
            if not sample.skipped:
                sample.add_trace(feature_key="feature_a", columns={"time": [0.0], "value": [1.0]})
    manifest_path = builder.finish()
```

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
python python\scripts\run_array_sample_parquet_tests.py
python python\scripts\run_builder_session_tests.py
```

## Script Inventory

유지하는 스크립트는 현재 포맷 기준으로 직접 실행할 일이 있는 것만 남겼습니다.

- `build_shards.py`: scalar sample metadata 또는 `scalar-sample-major-v1` manifest에서 dense-long shard를 만듭니다.
- `build_array_binary_shards.py`: array bundle manifest에서 custom binary shard를 만듭니다.
- `generate_synth.py`, `generate_array_synth.py`: synthetic scalar/array 입력 데이터를 만듭니다.
- `serve_feature_query_api.py`: 권장 scalar dense-long 및 array sample parquet 조회 서버입니다.
- `serve_array_api.py`: 기존 custom binary array shard 조회가 필요한 경우에만 사용하는 서버입니다.
- `run_*_tests.py`: 현재 유지 중인 package, API, session 검증 스크립트입니다.
- `benchmark_scalar_builder.py`, `benchmark_array_sample_parquet_vs_binary.py`: 최근 포맷 기준 성능 비교용 스크립트입니다.
