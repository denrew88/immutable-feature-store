# array-binary-shard-java

Standalone Java jar for the dense-id array binary shard format.

This package is the Java counterpart to the Python `array_binary_shard` wheel.
Current jar output follows the **v3** array binary shard spec:

- manifest-defined `point_schema`
- no special requirement that `time` / `value` exist
- categorical dictionary sidecars
- dense ids:
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index

## What It Provides

- Build binary array shards from sample-major bundle manifests
- Build binary array shards directly from traces with `ArrayDatasetBuilder`
- Open and read array shard datasets
- Resolve `sample_key -> sample_id`
- Resolve `feature_key -> feature_id`
- Read traces by dense id or by external key
- Optionally decode categorical columns back to labels

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

- `dist/array-binary-shard-java-0.3.0.jar`

## Public Entry Point

Use `fs.io.ArrayBinaryShards`.

## Direct-Ingestion Build

```java
import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.io.ArrayBinaryShards;
import fs.io.ArrayDatasetBuilder;
import fs.model.LogicalType;
import fs.model.PointColumnSpec;
import fs.model.StorageType;

import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

List<PointColumnSpec> pointSchema = Arrays.asList(
        new PointColumnSpec("ts", StorageType.INT64, LogicalType.TIMESTAMP_NS),
        new PointColumnSpec("dt", StorageType.INT64, LogicalType.TIMEDELTA_NS),
        new PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER),
        new PointColumnSpec("state_code", StorageType.UINT32, LogicalType.CATEGORICAL),
        new PointColumnSpec("event_type", StorageType.UINT32, LogicalType.CATEGORICAL));

ArrayShardConfig shardCfg = new ArrayShardConfig();
shardCfg.targetShardBytes = 32L * 1024L * 1024L;
shardCfg.samplesPerBlock = 16;

ArrayBundleConfig bundleCfg = new ArrayBundleConfig();
bundleCfg.maxBundleRows = 1 << 15;

try (ArrayDatasetBuilder builder = new ArrayDatasetBuilder(
        "C:/data/array_binary_shards",
        "C:/data/sample_meta.parquet",
        pointSchema,
        "",
        Arrays.asList("feature_a", "feature_b"),
        shardCfg,
        bundleCfg,
        "C:/data/array_bundle_stage")) {
    Map<String, Object> columns = new LinkedHashMap<String, Object>();
    columns.put("ts", new Instant[]{
            Instant.ofEpochSecond(1L, 0L),
            Instant.ofEpochSecond(2L, 0L)
    });
    columns.put("dt", new Duration[]{
            Duration.ZERO,
            Duration.ofSeconds(1L)
    });
    columns.put("phase", new int[]{10, 11});
    columns.put("state_code", new String[]{"OK", "WARN"});
    columns.put("event_type", new String[]{"START", "STOP"});

    builder.addTrace(0L, null, "feature_a", columns);
    builder.finishBundles();
    builder.updateFeatureMeta(Arrays.<Map<String, Object>>asList(
            new LinkedHashMap<String, Object>() {{
                put("feature_key", "feature_a");
                put("feature_group", "alpha");
            }},
            new LinkedHashMap<String, Object>() {{
                put("feature_key", "feature_b");
                put("feature_group", "beta");
            }}
    ), "feature_key", true);

    String manifestPath = builder.buildShards(true);
}
```

## Build From Existing Bundle Manifest

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

## Open And Read By Dense Id

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
            locator,
            true);
}
```

When `decodeCategorical=true`, categorical point columns are returned as `String[]`.
When `decodeCategorical=false`, they are returned as raw dense integer codes (`long[]`).

## Read By Keys

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
            "feature_a",
            new String[]{"sample_000000", "sample_000007"},
            locator,
            featureIds,
            sampleIds,
            true);
}
```

## Metadata Helpers

```java
import fs.io.ArrayBinaryShards;

ArrayBinaryShards.writeSampleMeta(Arrays.<Map<String, Object>>asList(
        new LinkedHashMap<String, Object>() {{
            put("sample_key", "sample_000000");
            put("split", "train");
        }}
), "C:/data/sample_meta.parquet");
```

Rules:

- `sample_id` / `feature_id` are auto-filled if omitted
- if provided, they must match dense row order exactly
- `sample_key` / `feature_key` are validated for nulls and duplicates

## Included Classes

The jar includes the array binary shard library surface and supporting models/config:

- `fs.config.ArrayBundleConfig`
- `fs.config.ArrayShardConfig`
- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayDatasetBuilder`
- `fs.io.ArrayShardBuilder`
- `fs.io.ArrayShardReader`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayFeatureLocatorIndex`
- `fs.io.ArraySampleIdIndex`
- `fs.io.ArrayFeatureIdIndex`
- `fs.io.ArrayMetadataWriter`
- `fs.io.ArrayShardManifestIO`
- `fs.model.PointColumnSpec`
- `fs.model.StorageType`
- `fs.model.LogicalType`

It does not include the CLI `scripts.*` classes.

## Fresh Clone Setup

```powershell
git clone https://github.com/denrew88/immutable-feature-store.git
cd immutable-feature-store
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
cd packages\array_binary_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```
