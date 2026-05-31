# immutable-feature-store

오프라인으로 생성하고 빠르게 조회하는 feature store 저장소입니다.

이 저장소는 세 가지 저장 경로를 다룹니다.

- **array binary shard**
  - 샘플마다 길이가 다른 trace를 feature-major custom binary로 저장
  - manifest가 point schema를 정의
  - 빠른 serving과 random access에 사용
- **array sample parquet**
  - 샘플마다 길이가 다른 trace를 sample-major Parquet으로 저장
  - 일반 Parquet 도구로 열어보기 쉬운 viewer/debugging용 포맷
- **scalar shard**
  - `feature x sample` 형태의 dense 값 저장
  - 빠른 조회와 feature selection에 사용

포함 내용:

- 포맷 문서
- Python 기준 구현
- Python wheel 패키지
- Java 코드와 Java jar 패키지

## 폴더 구조

```text
docs/
  array_binary_shard_format_v3.md
  array_sample_parquet_format_v1.md
  scalar_parquet_shard_format.md

python/
  fs/
    array/
    array_sample_parquet/
    scalar/
    feature_selection/
  scripts/
  README.md

java/
  src/
  README.md

packages/
  array_binary_shard/
  array_sample_parquet/
  scalar_feature_shard/
  array_binary_shard_java/
  array_sample_parquet_java/
  scalar_feature_shard_java/
```

## 주요 구성

### Array

- 문서
  - [docs/array_binary_shard_format_v3.md](docs/array_binary_shard_format_v3.md)
- Python core
  - [python/fs/array](python/fs/array)
- Python wheel
  - [packages/array_binary_shard](packages/array_binary_shard)
- Java jar
  - [packages/array_binary_shard_java](packages/array_binary_shard_java)

Array binary shard v3 특징:

- dense `sample_id` / `feature_id`
- manifest 정의 point schema
- categorical dictionary sidecar
- direct-ingestion builder
- `sample_key` / `feature_key` 기반 조회

### Array Sample Parquet

- 문서
  - [docs/array_sample_parquet_format_v1.md](docs/array_sample_parquet_format_v1.md)
- Python core
  - [python/fs/array_sample_parquet](python/fs/array_sample_parquet)
- Python wheel
  - [packages/array_sample_parquet](packages/array_sample_parquet)
- Java jar
  - [packages/array_sample_parquet_java](packages/array_sample_parquet_java)

- row 하나가 point 하나입니다.
- `sample_parts/`는 point rows를 저장합니다.
- `trace_index_parts/`는 present trace와 empty trace index를 저장합니다.
- sample-major layout이며 물리 순서는 `(sample_id, feature_id, point_idx)`입니다.
- raw builder도 `raw_samples/` point rows와 `raw_trace_index/` present trace index를 분리해 씁니다.
- categorical column은 별도 dictionary sidecar 없이 string으로 저장합니다.
- `target_part_bytes` / `targetPartBytes` 기반 자동 part 크기 조절, `raw_state.json`과 `raw_samples.jsonl` 기반 sample 단위 resume, `sample_key` / `feature_key` 기반 조회를 지원합니다.
- Java builder는 Arrow vector batch를 DuckDB에 넘겨 raw parquet를 쓰고, raw/final parquet의 물리 row 정렬을 검증하는 helper를 제공합니다.

### Scalar

- 문서
  - [docs/scalar_parquet_shard_format.md](docs/scalar_parquet_shard_format.md)
- Python core
  - [python/fs/scalar](python/fs/scalar)
- Python wheel
  - [packages/scalar_feature_shard](packages/scalar_feature_shard)
- Java jar
  - [packages/scalar_feature_shard_java](packages/scalar_feature_shard_java)

Scalar shard 특징:

- dense `sample_id` / `feature_id`
- standalone artifact 구조
- direct-ingestion builder
- `selection_stats/<y>.parquet` 기반 fast path
- selection fallback 지원

### Java

- Java 사용법과 설명
  - [java/README.md](java/README.md)

## Python 서버

서버 구현:

- [python/scripts/serve_feature_query_api.py](python/scripts/serve_feature_query_api.py)
  - 현재 Python wheel/package API 기준의 권장 조회 서버입니다.
  - scalar blob shard, scalar dense-long shard, array sample parquet를 조회합니다.
- [python/scripts/serve_array_api.py](python/scripts/serve_array_api.py)
  - 기존 core `python/fs` 구현까지 포함한 legacy/통합 서버입니다.

권장 서버 엔드포인트:

- `GET /healthz`
- `GET /cache-stats`
- `POST /array-sample-parquet/schema`
- `POST /array-sample-parquet/traces`
- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

조회 요청은 dense id 또는 external key 중 하나만 받습니다. 같은 축에 id와 key를 동시에 주면 에러로 처리합니다.

실행:

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

## Python 패키지

### `array_binary_shard`

빌드:

```powershell
cd packages\array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 public API:

- `open_shard(...)`
- `ArrayDatasetBuilder`
- `build_shard(...)`

### `array_sample_parquet`

빌드:

```powershell
cd packages\array_sample_parquet
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 public API:

- `ArraySampleParquetDatasetBuilder.open_session(...)`
- `ArraySampleParquetRawDatasetBuilder.open_session(...)`
- `builder.status()`
- `builder.sample(...)`
- `sample.add_trace(...)`
- `builder.finish()`
- `builder.compact()`
- `open_array_sample_parquet(...)`
- `reader.get_traces(...)`
- `reader.get_traces_json(...)`

예제:

- [packages/array_sample_parquet/examples/build_array_sample_parquet_example.py](packages/array_sample_parquet/examples/build_array_sample_parquet_example.py)
- [packages/array_sample_parquet_java/examples/BuildArraySampleParquetWithJarExample.java](packages/array_sample_parquet_java/examples/BuildArraySampleParquetWithJarExample.java)

### `scalar_feature_shard`

빌드:

```powershell
cd packages\scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 public API:

- `open_shard(...)`
- `open_dense_long_shard(...)`
- `ScalarDatasetBuilder`
- `ScalarRawDatasetBuilder`
- `build_shard(...)`
- `build_dense_long_shards_from_sample_bundles(...)`
- `select_features(...)`

예제:

- [packages/scalar_feature_shard/examples/build_scalar_dense_long_example.py](packages/scalar_feature_shard/examples/build_scalar_dense_long_example.py)
- [packages/scalar_feature_shard_java/examples/BuildScalarFeatureShardWithJarExample.java](packages/scalar_feature_shard_java/examples/BuildScalarFeatureShardWithJarExample.java)

## VS Code launch

주요 launch 설정:

- [.vscode/launch.json](.vscode/launch.json)

예:

- `Serve Feature API (Python)`
- `Build Array Binary Shards (Python)`
- `Run Scalar Package Tests (Python)`
- `Build Scalar Feature Shard Wheel (Python)`
- `Run Scalar Builder Tests (java)`

## 데이터 정책

`data/` 아래 생성 산출물은 git으로 추적하지 않습니다.

저장소에는 주로 다음만 남깁니다.

- 소스 코드
- 패키지 스캐폴딩
- 문서
- 테스트

대용량 shard 산출물은 저장하지 않습니다.

## 어디부터 보면 되는가

- array binary format을 이해하려면
  - [docs/array_binary_shard_format_v3.md](docs/array_binary_shard_format_v3.md)
- array sample parquet format과 구현 방식을 이해하려면
  - [docs/array_sample_parquet_format_v1.md](docs/array_sample_parquet_format_v1.md)
- scalar shard format을 이해하려면
  - [docs/scalar_parquet_shard_format.md](docs/scalar_parquet_shard_format.md)
- Python 패키지를 쓰려면
  - [packages/array_binary_shard/README.md](packages/array_binary_shard/README.md)
  - [packages/array_sample_parquet/README.md](packages/array_sample_parquet/README.md)
  - [packages/scalar_feature_shard/README.md](packages/scalar_feature_shard/README.md)
- Java jar 패키지를 쓰려면
  - [packages/array_binary_shard_java/README.md](packages/array_binary_shard_java/README.md)
  - [packages/array_sample_parquet_java/README.md](packages/array_sample_parquet_java/README.md)
  - [packages/scalar_feature_shard_java/README.md](packages/scalar_feature_shard_java/README.md)
- Python core 흐름을 보려면
  - [python/README.md](python/README.md)
- Java 쪽을 보려면
  - [java/README.md](java/README.md)
