Java용 array binary shard 라이브러리 패키지입니다.

## 포함 기능

- array binary shard v3 reader facade
- direct-ingestion builder
- sample/feature metadata helper
- key resolver

진입점:

- `fs.io.ArrayBinaryShards`

## 준비물

- Java 8
- `java/lib/duckdb_jdbc-1.1.3.jar`

jar가 없으면:

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

## 빌드

```powershell
cd packages\array_binary_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

출력:

- `dist/array-binary-shard-java-0.3.0.jar`

thin jar이므로 런타임 classpath에 `duckdb_jdbc-1.1.3.jar`를 같이 둬야 한다.

## public API

`fs.io.ArrayBinaryShards`:

- `buildFromBundles(...)`
- `loadManifest(...)`
- `open(...)`
- `loadLocator(...)`
- `loadSampleIds(...)`
- `loadFeatureIds(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`

`fs.io.ArrayDatasetBuilder`:

- direct-ingestion builder
- known-feature / discovered-feature mode

## 참고

- dense id 규칙
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- manifest 기준 relative path를 사용한다.
- categorical column은 code로 저장하고 reader에서 label로 decode할 수 있다.
