Java 8 (no Maven) build & run

## Prereq

- Java 8
- DuckDB JDBC jar for Java 8: `duckdb_jdbc-1.1.3.jar`
  - this repository no longer tracks the jar itself
  - download it into `java/lib/` with:

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

Keep only one duckdb jar in `java/lib`.

## Compile

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
javac -cp "java\lib\duckdb_jdbc-1.1.3.jar" -d java\out @java\sources.txt
```

## Scalar CLI

Build scalar shards:

```powershell
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-meta C:\data\sample_meta.parquet ^
  --feature-meta C:\data\feature_meta.parquet ^
  --out-dir C:\data\shards ^
  --target-shard-mb 32
```

Build scalar shards from bundled sample-major stage:

```powershell
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-bundle-manifest C:\data\sample_major_stage\sample_major_manifest.json ^
  --out-dir C:\data\shards ^
  --target-shard-mb 32
```

Run selection:

```powershell
java -cp "java\lib\*;java\out" scripts.RunSelectionMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --top-m 100
```

Locate one scalar feature:

```powershell
java -cp "java\lib\*;java\out" scripts.LocateFeatureMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --feature-id 12345
```

## Java Scalar Public API

Use `fs.io.ScalarFeatureShards`.

It provides:

- `loadManifest(...)`
- `open(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`
- `buildCandidates(...)`
- `selectFeatures(...)`

The direct-ingestion builder is `fs.io.ScalarDatasetBuilder`.

Key builder methods:

- `writeSample(...)`
- `openSample(...)`
- `finishSampleMajor()`
- `updateFeatureMeta(...)`
- `buildShards(keepSampleMajor)`

## Array CLI

Build array shards from an existing bundle manifest:

```powershell
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.BuildArrayShardsMain ^
  --bundle-manifest C:\data\array_bundle_manifest.json ^
  --out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16
```

Locate one array feature:

```powershell
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.LocateArrayFeatureMain ^
  --manifest C:\data\array_shards\array_binary_shard_manifest.json ^
  --feature-id 12345
```

Generate synthetic array data and build shards:

```powershell
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.GenerateArraySynthMain ^
  --bundle-out-dir C:\data\array_bundles ^
  --sample-meta C:\data\array_sample_meta.parquet ^
  --shard-out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16 ^
  --n-samples 256 ^
  --n-features 64 ^
  --seed 7
```

## Array Binary Format Notes

Java array shards now follow the Python binary shard **v3** format.

Key points:

- standalone artifact layout:
  - `array_binary_shard_manifest.json`
  - `sample_meta.parquet`
  - `feature_meta.parquet`
  - `categorical_dictionaries/*.parquet` when needed
  - `array_binary_feature_shards/shard_XXXX.blocks.idx`
  - `array_binary_feature_shards/shard_XXXX.blocks.bin`
- dense ids:
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- `point_schema` is manifest-defined and may omit `time` / `value`
- supported logical/storage types:
  - `continuous + float64`
  - `integer + int32/int64/uint32/uint64`
  - `categorical + uint32`
  - `timestamp_ns + int64`
  - `timedelta_ns + int64`
- categorical columns are stored as dense integer codes and can be decoded back to labels by the Java reader

## Java Array Public API

Use `fs.io.ArrayBinaryShards`.

It provides:

- `buildFromBundles(...)`
- `loadManifest(...)`
- `open(...)`
- `loadLocator(...)`
- `loadSampleIds(...)`
- `loadFeatureIds(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`

The direct-ingestion builder is `fs.io.ArrayDatasetBuilder`.

## Tests

Run all Java tests:

```powershell
java -cp "java\lib\*;java\out" scripts.RunTestsMain --seed 0
```

Run array storage tests:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayStorageTestsMain
```

Run array synthetic tests:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArraySyntheticTestsMain
```

Run array v3 tests:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayV3TestsMain
```

Run scalar builder tests:

```powershell
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
```

## Notes

- `java/lib/duckdb_jdbc-1.1.3.jar` is intentionally excluded from git.
- current Java array binary codec is `none`
- `blocks.idx` is a fixed-size offset table with one 32-byte record per `(feature_id, block_id)` slot
- `blocks.bin` stores variable-length payloads
- array blocks are sample micro-blocks; default `samples_per_block=16`
- shard partitioning defaults to `target_shard_mb=32`
