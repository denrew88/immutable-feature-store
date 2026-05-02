Java package for scalar feature shards.

## What It Includes

- dense-id scalar shard reader facade
- direct-ingestion builder with bundled sample-major stage
- feature selection facade
- sample/feature metadata helpers

Primary entry point:

- `fs.io.ScalarFeatureShards`

## Prereq

- Java 8
- DuckDB JDBC jar for Java 8 in `java/lib/duckdb_jdbc-1.1.3.jar`

If the jar is missing:

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

## Build

```powershell
cd packages\scalar_feature_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Output:

- `dist/scalar-feature-shard-java-0.1.0.jar`

This is a thin jar. Keep `duckdb_jdbc-1.1.3.jar` on the runtime classpath.

## Public API

`fs.io.ScalarFeatureShards` provides:

- `loadManifest(...)`
- `open(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`
- `buildCandidates(...)`
- `selectFeatures(...)`

`fs.io.ScalarShardDataset` provides:

- `getValue(...)`
- `getValues(...)`
- `getValueByKey(...)`
- `getValuesByKeys(...)`

`fs.io.ScalarDatasetBuilder` provides:

- `writeSample(...)`
- `openSample(...)`
- `finishSampleMajor()`
- `updateFeatureMeta(...)`
- `buildShards(keepSampleMajor)`

## Notes

- current scalar storage format is standalone parquet shard artifact
- ids are dense:
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- selection stats are stored per `y_col` under `selection_stats/*.parquet`
- intermediate sample-major stage is bundle-based, not file-per-sample
