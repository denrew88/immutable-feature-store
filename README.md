# immutable-feature-store

오프라인으로 생성하고 빠르게 조회하는 feature store 저장소입니다.

이 저장소는 두 가지 데이터 형태를 다룹니다.

- **array shard**
  - 샘플마다 길이가 다른 trace를 저장
  - manifest가 point schema를 정의
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
  array_binary_shard_format.md
  array_binary_shard_format_v3.md
  scalar_parquet_shard_format.md

python/
  fs/
    array/
    scalar/
    feature_selection/
  scripts/
  README.md

java/
  src/
  README.md

packages/
  array_binary_shard/
  scalar_feature_shard/
  array_binary_shard_java/
  scalar_feature_shard_java/
```

## 주요 구성

### Array

- 문서
  - [docs/array_binary_shard_format.md](docs/array_binary_shard_format.md)
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
- scalar shard format을 이해하려면
  - [docs/scalar_parquet_shard_format.md](docs/scalar_parquet_shard_format.md)
- Python 패키지를 쓰려면
  - [packages/array_binary_shard/README.md](packages/array_binary_shard/README.md)
  - [packages/scalar_feature_shard/README.md](packages/scalar_feature_shard/README.md)
- Python core 흐름을 보려면
  - [python/README.md](python/README.md)
- Java 쪽을 보려면
  - [java/README.md](java/README.md)
