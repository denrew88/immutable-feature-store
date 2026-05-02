# array-binary-shard-java

Standalone Java jar for the dense-id array binary shard v2 format.

This package is the Java counterpart to the Python `array_binary_shard` wheel.

## What It Provides

- Build binary array shards from sample-major bundle manifests
- Open and read binary array shard datasets
- Resolve `sample_key -> sample_id`
- Resolve `feature_key -> feature_id`
- Read traces by dense id or by external key

## Format Assumptions

- `sample_id == sample_meta.parquet` row index
- `feature_id == feature_meta.parquet` row index
- `sample_key` lives in `sample_meta.parquet`
- `feature_key` lives in `feature_meta.parquet`
- Artifact layout is standalone:
  - `array_binary_shard_manifest.json`
  - `sample_meta.parquet`
  - `feature_meta.parquet`
  - `array_binary_feature_shards/*.blocks.idx`
  - `array_binary_feature_shards/*.blocks.bin`

## Runtime Dependency

This jar is a thin artifact. It requires:

- `duckdb_jdbc-1.1.3.jar`

This repository does not track that runtime jar. Download it into `java/lib/`
from the repo root with:

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

Put both jars on the classpath.

## Build The Jar

```powershell
cd packages/array_binary_shard_java
powershell -ExecutionPolicy Bypass -File ..\..\java\download_duckdb_jdbc.ps1
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Output:

- `dist/array-binary-shard-java-0.2.3.jar`

## Public Entry Point

Use `fs.io.ArrayBinaryShards`.

### Build

```java
import fs.config.ArrayShardConfig;
import fs.io.ArrayBinaryShards;

ArrayShardConfig cfg = new ArrayShardConfig();
cfg.targetShardBytes = 32L * 1024L * 1024L;
cfg.samplesPerBlock = 16;

String manifestPath = ArrayBinaryShards.buildFromBundles(
        "C:/data/array_bundle_manifest.json",
        "C:/data/array_binary_shards",
        cfg);
```

### Open And Read By Dense Id

```java
import fs.io.ArrayBinaryShards;
import fs.io.ArrayFeatureLocatorIndex;
import fs.io.ArrayShardReader;
import fs.model.ArrayShardManifest;
import fs.model.ArrayTrace;

import java.util.Map;

ArrayShardManifest manifest = ArrayBinaryShards.loadManifest(
        "C:/data/array_binary_shards/array_binary_shard_manifest.json");
ArrayFeatureLocatorIndex locator = ArrayBinaryShards.loadLocator(manifest);

try (ArrayShardReader reader = ArrayBinaryShards.open(
        "C:/data/array_binary_shards/array_binary_shard_manifest.json")) {
    Map<Long, ArrayTrace> traces = reader.loadFeatureSamples(
            123,
            new long[]{0L, 5L, 9L},
            locator);
}
```

### Read By Keys

```java
import fs.io.ArrayBinaryShards;
import fs.io.ArrayFeatureIdIndex;
import fs.io.ArrayFeatureLocatorIndex;
import fs.io.ArraySampleIdIndex;
import fs.io.ArrayShardReader;
import fs.model.ArrayShardManifest;
import fs.model.ArrayTrace;

import java.util.Map;

ArrayShardManifest manifest = ArrayBinaryShards.loadManifest(
        "C:/data/array_binary_shards/array_binary_shard_manifest.json");
ArrayFeatureLocatorIndex locator = ArrayBinaryShards.loadLocator(manifest);
ArrayFeatureIdIndex featureIds = ArrayBinaryShards.loadFeatureIds(manifest);
ArraySampleIdIndex sampleIds = ArrayBinaryShards.loadSampleIds(manifest);

try (ArrayShardReader reader = ArrayBinaryShards.open(
        "C:/data/array_binary_shards/array_binary_shard_manifest.json")) {
    Map<String, ArrayTrace> traces = reader.loadFeatureSamplesByKeys(
            "feature_000123",
            new String[]{"sample_000000", "sample_000007"},
            locator,
            featureIds,
            sampleIds);
}
```

## Included Classes

The jar includes only the array binary shard library surface and supporting models/config:

- `fs.config.ArrayBundleConfig`
- `fs.config.ArrayShardConfig`
- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayShardBuilder`
- `fs.io.ArrayShardReader`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayFeatureLocatorIndex`
- `fs.io.ArraySampleIdIndex`
- `fs.io.ArrayFeatureIdIndex`
- `fs.io.ArrayShardManifestIO`
- related array binary model/config helpers

It does not include the CLI `scripts.*` classes.

## Fresh Clone Setup

```powershell
git clone https://github.com/denrew88/immutable-feature-store.git
cd immutable-feature-store
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
cd packages\array_binary_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```
