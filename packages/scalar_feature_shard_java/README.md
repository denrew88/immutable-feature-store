Java용 scalar feature shard 라이브러리 패키지입니다.

## 포함 기능

- dense-id scalar shard reader facade
- bundled sample-major stage 기반 direct-ingestion builder
- selection facade
- sample/feature metadata helper

진입점:

- `fs.io.ScalarFeatureShards`

## 준비물

- Java 8
- `java/lib/duckdb_jdbc-1.1.3.jar`

jar가 없으면:

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

## 빌드

```powershell
cd packages\scalar_feature_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

출력:

- `dist/scalar-feature-shard-java-0.1.0.jar`

thin jar이므로 런타임 classpath에 DuckDB JDBC jar를 같이 둬야 한다.

## public API

`fs.io.ScalarFeatureShards`:

- `loadManifest(...)`
- `open(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`
- `buildCandidates(...)`
- `selectFeatures(...)`

`fs.io.ScalarShardDataset`:

- `getValue(...)`
- `getValues(...)`
- `getValueByKey(...)`
- `getValuesByKeys(...)`

`fs.io.ScalarDatasetBuilder`:

- `writeSample(...)`
- `openSample(...)`
- `finishSampleMajor()`
- `updateFeatureMeta(...)`
- `buildShards(keepSampleMajor)`

## 참고

- 최종 scalar artifact는 standalone 폴더 구조입니다.
- `selection_stats/<y>.parquet`를 통해 selection fast path를 탑니다.
- intermediate sample-major stage는 file-per-sample이 아니라 bundle 기반입니다.
