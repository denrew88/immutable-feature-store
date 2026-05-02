Java 8 (no Maven) build & run

Prereq:
- Java 8
- DuckDB JDBC jar for Java 8: `duckdb_jdbc-1.1.3.jar`
  - This repository no longer tracks the jar itself.
  - Download it into `java/lib/` with:
```
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```
  - Keep only one duckdb jar in `java/lib` (remove incompatible jars)

Compile:
```
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
javac -cp "java\lib\duckdb_jdbc-1.1.3.jar" -d java\out @java\sources.txt
```

Run (build shards):
```
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-meta C:\data\sample_meta.parquet ^
  --out-dir C:\data\shards ^
  --n-shards 16
```

Run (selection):
```
java -cp "java\lib\*;java\out" scripts.RunSelectionMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --top-m 100
```

Run tests (in-memory synthetic):
```
java -cp "java\lib\*;java\out" scripts.RunTestsMain --seed 0
```

Run (feature -> shard lookup):
```
java -cp "java\lib\*;java\out" scripts.LocateFeatureMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --feature-id 12345
```

Run (array bundle -> array shards):
```
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.BuildArrayShardsMain ^
  --bundle-manifest C:\data\array_bundle_manifest.json ^
  --out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16
```

Run (array feature -> block lookup):
```
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.LocateArrayFeatureMain ^
  --manifest C:\data\array_shards\array_binary_shard_manifest.json ^
  --feature-id 12345
```

Run (synthetic array bundle + shards):
```
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

Run array storage tests:
```
java -cp "java\lib\*;java\out" scripts.RunArrayStorageTestsMain
```

Run array synthetic tests:
```
java -cp "java\lib\*;java\out" scripts.RunArraySyntheticTestsMain
```

Notes:
- `java/lib/duckdb_jdbc-1.1.3.jar` is intentionally excluded from git. Use `java\download_duckdb_jdbc.ps1` on a fresh clone.
- `feature_id_dtype` in manifest is unified to `INT32` (signed int32).
- Java selection defaults are tuned for single-thread execution: `initialCap=2048`, `maxStep=4096`, `batchSize=1024`.
- Shard parquet schema is: `feature_id INTEGER, value_len INTEGER, values_blob BLOB, valid_blob BLOB`
- BLOB payload: values are float64 little-endian (length = value_len), valid is uint8 mask (length = value_len)
- Feature locator schema is: `feature_id INTEGER, global_rank INTEGER, shard_id INTEGER, offset_in_shard INTEGER, r2y DOUBLE, n_y_overlap INTEGER`
- Shard files are stored under a subfolder: `<out-dir>/feature_shards/`.
- `shard_manifest.json` stores `shard_path` (the subfolder path), `n_shards`, and `stats_y_col`.
- `RunSelectionMain` reuses locator-side `r2y/n_y_overlap` when the requested `y_col` matches `stats_y_col`; otherwise it falls back to rescanning shard blobs.
- Array sample bundle schema is: `sample_id BIGINT, feature_id INTEGER, flags TINYINT, trace_len INTEGER, time_blob BLOB, value_blob BLOB`
- Java array shards now follow the Python binary shard v2 format and are standalone artifacts:
  - `array_binary_shard_manifest.json`
  - `sample_meta.parquet`
  - `feature_meta.parquet`
  - `array_binary_feature_shards/shard_XXXX.blocks.idx`
  - `array_binary_feature_shards/shard_XXXX.blocks.bin`
- Array binary v2 uses dense ids:
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- External stable identifiers live in metadata:
  - `sample_key` in `sample_meta.parquet`
  - `feature_key` in `feature_meta.parquet`
- `blocks.idx` is a fixed-size offset table with one 32-byte record per `(feature_id, block_id)` slot.
- `blocks.bin` stores variable-length payloads as `[48-byte payload header][sample_flags][sample_offsets][time][value]`.
- Empty blocks still have a `blocks.idx` record, but its `data_length` is `0` and it has no payload in `blocks.bin`.
- Array blocks are sample micro-blocks; default `samples_per_block=16`.
- Array shard partitioning defaults to `target_shard_mb=32`; `--n-shards` is kept only as a legacy override.
- Java array lookup is direct binary seek/read; Parquet/Hadoop libs are no longer required for the array path.
- Current Java array binary codec is `none`, which matches the current recommended Python serving format.
- Synthetic array `sample_meta.parquet` includes `sample_id, sample_key, y`.
- Synthetic array `feature_meta.parquet` includes `feature_id, feature_key, feature_kind, group_id, scale`.
- Synthetic array bundle rows also use dense `sample_id`; `sampleIdOffset` is no longer part of the binary shard contract.
- Java sample id lookup is now an identity mapping over dense ids, and `ArrayShardReader.loadFeatureSamplesBySampleIds(...)` reads traces by dense `sample_id`.
- Java also provides metadata key resolvers:
  - `ArraySampleIdIndex.load(sampleMetaPath, sampleKeyCol)` for `sample_key -> sample_id`
  - `ArrayFeatureIdIndex.load(featureMetaPath, featureKeyCol)` for `feature_key -> feature_id`
  - `ArrayShardReader.loadFeatureSamplesByKeys(...)` for direct key-based trace lookup
