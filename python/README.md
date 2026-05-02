# Python 구현 안내

이 디렉터리는 array/scalar shard의 기준 Python 구현과 서버, 테스트 스크립트를 담고 있다.

## 주요 진입점

- scalar shard 빌드
  - `scripts/build_shards.py`
- scalar selection
  - `scripts/run_selection.py`
- scalar synthetic 생성
  - `scripts/generate_synth.py`
- array shard 빌드
  - `scripts/build_array_shards.py`
  - `scripts/build_array_binary_shards.py`
- array synthetic 생성
  - `scripts/generate_array_synth.py`
- FastAPI 서버
  - `scripts/serve_array_api.py`
- 테스트
  - `scripts/run_tests.py`
  - `scripts/run_scalar_api_tests.py`
  - `scripts/run_selection_api_tests.py`
  - `scripts/run_array_storage_tests.py`
  - `scripts/run_array_binary_storage_tests.py`
  - `scripts/run_array_api_tests.py`
  - `scripts/run_scalar_builder_tests.py`
  - `scripts/run_scalar_package_tests.py`
  - `scripts/run_array_binary_package_tests.py`

## 실행 위치

패키지 import가 맞도록 보통 `python/`에서 실행한다.

```powershell
cd python
```

## Scalar 파이프라인

### 기본 예시

```powershell
python -m scripts.generate_synth --out-dir ..\data\synth_py\samples --sample-meta ..\data\synth_py\sample_meta.parquet --feature-meta ..\data\synth_py\feature_meta.parquet
python -m scripts.build_shards --sample-meta ..\data\synth_py\sample_meta.parquet --feature-meta ..\data\synth_py\feature_meta.parquet --out-dir ..\data\synth_py\shards --target-shard-mb 32 --stats-y-col y --stats-y-col y_alt
python -m scripts.run_selection --manifest ..\data\synth_py\shards\shard_manifest.json --top-m 30
python -m scripts.locate_feature --manifest ..\data\synth_py\shards\shard_manifest.json --feature-id 12
```

### direct-ingestion builder

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

builder = ScalarDatasetBuilder(
    out_dir="..\\data\\scalar_shards",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=ScalarShardBuildOptions(
        target_shard_mb=32,
        stats_y_cols=("y", "y_alt"),
    ),
)

builder.write_sample(0, {"feature_a": 1.23, "feature_b": 4.56})
builder.write_sample(1, {"feature_a": 7.89})
manifest_path = builder.build_shards()
```

discovered-feature mode:

```python
from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder

with ScalarDatasetBuilder(
    out_dir="..\\data\\scalar_shards",
    sample_meta_path="..\\data\\scalar_sample_meta.parquet",
    build_options=ScalarShardBuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
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

주의:

- primary API는 sample 단위입니다.
- 같은 `sample_id`를 두 번 쓰면 에러입니다.
- intermediate sample-major stage는 bundle 기반입니다.

## Array 파이프라인

### bundle/shard 예시

```powershell
python -m scripts.generate_array_synth --bundle-out-dir ..\data\array_bundles --sample-meta ..\data\array_sample_meta.parquet --shard-out-dir ..\data\array_shards --n-samples 256 --n-features 64 --seed 7 --target-shard-mb 256 --samples-per-block 8 --row-group-size 64
python -m scripts.build_array_shards --bundle-manifest ..\data\array_bundle_manifest.json --out-dir ..\data\array_shards --target-shard-mb 256 --samples-per-block 8 --row-group-size 64
python -m scripts.build_array_binary_shards --bundle-manifest ..\data\array_bundle_manifest.json --out-dir ..\data\array_binary_shards --target-shard-mb 32 --samples-per-block 16 --codec none
```

### direct-ingestion builder

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
    "..\\data\\sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    "..\\data\\feature_meta.parquet",
)

with ArrayDatasetBuilder(
    out_dir="..\\data\\array_binary_shards",
    sample_meta_path=sample_meta_path,
    point_schema=[
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ],
    feature_meta_path=feature_meta_path,
    build_options=ArrayBinaryBuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
) as builder:
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

known-feature mode와 discovered-feature mode를 모두 지원한다.

## 서버

서버 실행:

```powershell
python -m scripts.serve_array_api
```

주요 엔드포인트:

- `POST /array-schema`
- `POST /array-feature`
- `POST /scalar-feature`
- `POST /run-selection`

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
