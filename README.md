# immutable-feature-store

Offline-built, read-optimized feature store for two data shapes:

- **array shards**
  per-sample variable-length traces with manifest-defined point schema
- **scalar shards**
  dense feature-by-sample values for fast lookup and feature selection

This repository contains the format docs, Python reference implementations, Python wheels, and Java support code used to build and serve those artifacts.

## Layout

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

## Main pieces

### Array

- format docs:
  - [docs/array_binary_shard_format.md](docs/array_binary_shard_format.md)
  - [docs/array_binary_shard_format_v3.md](docs/array_binary_shard_format_v3.md)
- core implementation:
  - [python/fs/array](python/fs/array)
- Python wheel:
  - [packages/array_binary_shard](packages/array_binary_shard)

Array binary shard v3 supports:
- dense `sample_id` / `feature_id`
- manifest-defined point schema
- categorical dictionary sidecars
- direct-ingestion builder
- key-based lookup through `sample_key` / `feature_key`

### Scalar

- format doc:
  - [docs/scalar_parquet_shard_format.md](docs/scalar_parquet_shard_format.md)
- core implementation:
  - [python/fs/scalar](python/fs/scalar)
- Python wheel:
  - [packages/scalar_feature_shard](packages/scalar_feature_shard)

Scalar shard supports:
- dense `sample_id` / `feature_id`
- standalone artifact layout
- direct-ingestion builder
- precomputed `selection_stats/<y>.parquet`
- selection fast-path and shard-scan fallback

### Java

- Java code and notes:
  - [java/README.md](java/README.md)
- Java jar package:
  - [packages/array_binary_shard_java](packages/array_binary_shard_java)
  - [packages/scalar_feature_shard_java](packages/scalar_feature_shard_java)

## Python server

FastAPI server:
- [python/scripts/serve_array_api.py](python/scripts/serve_array_api.py)

Current endpoints:
- `GET /healthz`
- `GET /cache-stats`
- `POST /array-schema`
- `POST /array-feature`
- `POST /scalar-feature`
- `POST /run-selection`

The server accepts either dense ids or external keys, depending on the endpoint fields.

## Python packages

### 1. `array_binary_shard`

Build:
```powershell
cd packages\array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

Public API centers on:
- `open_shard(...)`
- `ArrayDatasetBuilder`
- `build_shard(...)`

### 2. `scalar_feature_shard`

Build:
```powershell
cd packages\scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

Public API centers on:
- `open_shard(...)`
- `ScalarDatasetBuilder`
- `build_shard(...)`
- `select_features(...)`

## VS Code launch configs

Useful launch entries live in:
- [.vscode/launch.json](.vscode/launch.json)

Notable entries:
- `Serve Feature API (Python)`
- `Build Array Binary Shards (Python)`
- `Run Scalar Package Tests (Python)`
- `Build Scalar Feature Shard Wheel (Python)`

## Data policy

Generated data under `data/` is intentionally not tracked by git.

The repo is meant to store:
- source code
- package scaffolding
- docs
- tests

and not large generated shard outputs.

## Where to start

If the goal is:

- **understand array binary format**
  start with [docs/array_binary_shard_format_v3.md](docs/array_binary_shard_format_v3.md)
- **understand scalar shard format**
  start with [docs/scalar_parquet_shard_format.md](docs/scalar_parquet_shard_format.md)
- **use the Python packages**
  see:
  - [packages/array_binary_shard/README.md](packages/array_binary_shard/README.md)
  - [packages/scalar_feature_shard/README.md](packages/scalar_feature_shard/README.md)
- **work on the core reference code**
  see [python/README.md](python/README.md)
