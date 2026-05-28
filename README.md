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

Array sample parquet v1 특징:

- row 하나가 `(sample_id, feature_id)` trace 하나
- point column을 Parquet `list<typed value>`로 저장
- sample-major layout
- `.parquet.tmp`에 streaming write 후 sample 경계에서 commit
- `target_part_bytes` / `targetPartBytes` 기반 자동 part 크기 조절
- `state.json`과 `parts.jsonl` 기반 resume
- `sample_key` / `feature_key` 기반 조회

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

- [python/scripts/serve_array_api.py](python/scripts/serve_array_api.py)

현재 엔드포인트:

- `GET /healthz`
- `GET /cache-stats`
- `POST /array-schema`
- `POST /array-feature`
- `POST /array-sample-parquet/schema`
- `POST /array-sample-parquet/traces`
- `POST /scalar-feature`
- `POST /run-selection`

엔드포인트에 따라 dense id 또는 external key를 사용할 수 있다.

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
- `builder.status()`
- `builder.sample(...)`
- `sample.add_trace(...)`
- `builder.finish()`
- `open_array_sample_parquet(...)`
- `reader.get_traces(...)`
- `reader.get_traces_json(...)`

### `scalar_feature_shard`

빌드:

```powershell
cd packages\scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 public API:

- `open_shard(...)`
- `ScalarDatasetBuilder`
- `build_shard(...)`
- `select_features(...)`

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
