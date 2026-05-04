package scripts;

import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.io.ArrayBinaryShards;
import fs.io.array.ArrayFeatureIdIndex;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.io.array.ArraySampleIdIndex;
import fs.io.ArrayBinaryShardReader;
import fs.model.array.ArrayShardManifest;
import fs.model.array.ArrayTrace;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Array binary shard v3 전용 public API와 builder 흐름을 검증하는 테스트 엔트리포인트다.
 */
public class RunArrayV3TestsMain {
    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_array_v3_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        ArrayBinaryShards.writeSampleMeta(sampleRecords(), sampleMetaPath);

        ArrayShardConfig shardCfg = new ArrayShardConfig();
        shardCfg.targetShardBytes = 1L << 20;
        shardCfg.samplesPerBlock = 2;

        ArrayBundleConfig bundleCfg = new ArrayBundleConfig();
        bundleCfg.maxBundleRows = 4;
        bundleCfg.maxBundleBytes = 1L << 20;

        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("ts", StorageType.INT64, LogicalType.TIMESTAMP_NS),
                new PointColumnSpec("dt", StorageType.INT64, LogicalType.TIMEDELTA_NS),
                new PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER),
                new PointColumnSpec("state_code", StorageType.UINT32, LogicalType.CATEGORICAL),
                new PointColumnSpec("event_type", StorageType.UINT32, LogicalType.CATEGORICAL)
        );

        String shardManifestPath;
        try (fs.io.ArrayDatasetBuilder builder = new fs.io.ArrayDatasetBuilder(
                new File(root, "shards").getAbsolutePath(),
                sampleMetaPath,
                pointSchema,
                "",
                null,
                shardCfg,
                bundleCfg,
                new File(root, "bundle_stage").getAbsolutePath())) {
            try (fs.io.ArrayDatasetBuilder.ArraySampleContext sample0 = builder.sample(0L)) {
                sample0.addTrace(null, "feature_a", columns(
                        new Instant[]{Instant.ofEpochSecond(1L, 0L), Instant.ofEpochSecond(2L, 5L)},
                        new Duration[]{Duration.ZERO, Duration.ofSeconds(1L)},
                        new int[]{10, 11},
                        new String[]{"OK", "WARN"},
                        new String[]{"START", "STOP"}));
                sample0.addTrace(null, "feature_b", columns(
                        new Instant[]{Instant.ofEpochSecond(3L, 0L)},
                        new Duration[]{Duration.ofSeconds(2L)},
                        new int[]{20},
                        new String[]{"WARN"},
                        new String[]{"STOP"}));
            }
            try (fs.io.ArrayDatasetBuilder.ArraySampleContext sample2 = builder.sample(2L)) {
                sample2.addTrace(null, "feature_a", columns(
                        new Instant[]{Instant.ofEpochSecond(4L, 7L)},
                        new Duration[]{Duration.ofSeconds(3L)},
                        new int[]{12},
                        new String[]{"OK"},
                        new String[]{"START"}));
            }

            String bundleManifestPath = builder.finishBundles();
            require(new File(bundleManifestPath).exists(), "missing bundle manifest");

            ArrayList<Map<String, Object>> featureRows = new ArrayList<Map<String, Object>>();
            featureRows.add(row("feature_key", "feature_a", "feature_group", "alpha"));
            featureRows.add(row("feature_key", "feature_b", "feature_group", "beta"));
            builder.updateFeatureMeta(featureRows, "feature_key", true);
            shardManifestPath = builder.buildShards(true);
        }

        ArrayShardManifest manifest = ArrayBinaryShards.loadManifest(shardManifestPath);
        require(manifest.version == 3, "expected manifest version 3");
        require(manifest.pointSchema.size() == 5, "point_schema size mismatch");
        require(manifest.nFeatures == 2, "n_features mismatch");
        require(new File(manifest.sampleMetaPath).exists(), "missing copied sample meta");
        require(new File(manifest.featureMetaPath).exists(), "missing copied feature meta");
        for (PointColumnSpec spec : manifest.pointSchema) {
            if (spec.logicalType == LogicalType.CATEGORICAL) {
                require(spec.dictionaryPath != null && !spec.dictionaryPath.isEmpty(), "missing dictionary path for " + spec.name);
                require(new File(spec.dictionaryPath).exists(), "missing categorical dictionary for " + spec.name);
            }
        }

        ArrayFeatureLocatorIndex locator = ArrayBinaryShards.loadLocator(manifest);
        ArrayFeatureIdIndex featureIds = ArrayBinaryShards.loadFeatureIds(manifest);
        ArraySampleIdIndex sampleIds = ArrayBinaryShards.loadSampleIds(manifest);
        require(featureIds.findFeatureIdByKey("feature_a") != null && featureIds.findFeatureIdByKey("feature_a") == 0, "feature key lookup mismatch");
        require(sampleIds.findSampleIdByKey("sample_000002") != null && sampleIds.findSampleIdByKey("sample_000002") == 2L, "sample key lookup mismatch");

        try (ArrayBinaryShardReader reader = ArrayBinaryShards.open(shardManifestPath)) {
            Map<Long, ArrayTrace> raw = reader.loadFeatureSamples(0, new long[]{0L, 1L, 2L}, locator, false);
            ArrayTrace sample0 = raw.get(0L);
            require(sample0 != null, "missing feature_a sample0 trace");
            require(!sample0.columns.containsKey("time"), "v3 trace should not synthesize time compatibility column");
            require(!sample0.columns.containsKey("value"), "v3 trace should not synthesize value compatibility column");
            assertLongArray((long[]) sample0.columns.get("ts"), new long[]{1_000_000_000L, 2_000_000_005L}, "sample0 ts");
            assertLongArray((long[]) sample0.columns.get("dt"), new long[]{0L, 1_000_000_000L}, "sample0 dt");
            assertIntArray((int[]) sample0.columns.get("phase"), new int[]{10, 11}, "sample0 phase");
            assertLongArray((long[]) sample0.columns.get("state_code"), new long[]{1L, 2L}, "sample0 state_code");
            assertLongArray((long[]) sample0.columns.get("event_type"), new long[]{1L, 2L}, "sample0 event_type");

            ArrayTrace sample1 = raw.get(1L);
            require(sample1 != null, "missing empty trace for sample1");
            require(sample1.flags == 0, "sample1 should be missing");
            require(((long[]) sample1.columns.get("ts")).length == 0, "empty trace ts length mismatch");

            Map<String, ArrayTrace> decoded = reader.loadFeatureSamplesByKeys(
                    "feature_a",
                    new String[]{"sample_000000", "sample_000002"},
                    locator,
                    featureIds,
                    sampleIds,
                    true);
            ArrayTrace decoded0 = decoded.get("sample_000000");
            require(decoded0 != null, "missing decoded trace");
            assertStringArray((String[]) decoded0.columns.get("state_code"), new String[]{"OK", "WARN"}, "decoded state_code");
            assertStringArray((String[]) decoded0.columns.get("event_type"), new String[]{"START", "STOP"}, "decoded event_type");

            ArrayTrace decoded2 = decoded.get("sample_000002");
            require(decoded2 != null, "missing decoded trace sample2");
            assertStringArray((String[]) decoded2.columns.get("state_code"), new String[]{"OK"}, "decoded state_code sample2");
            assertStringArray((String[]) decoded2.columns.get("event_type"), new String[]{"START"}, "decoded event_type sample2");
        }

        System.out.println("java array v3 tests passed");
    }

    private static List<Map<String, Object>> sampleRecords() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        out.add(row("sample_key", "sample_000000", "split", "train"));
        out.add(row("sample_key", "sample_000001", "split", "train"));
        out.add(row("sample_key", "sample_000002", "split", "test"));
        return out;
    }

    private static LinkedHashMap<String, Object> columns(
            Instant[] ts,
            Duration[] dt,
            int[] phase,
            String[] stateCode,
            String[] eventType) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        out.put("ts", ts);
        out.put("dt", dt);
        out.put("phase", phase);
        out.put("state_code", stateCode);
        out.put("event_type", eventType);
        return out;
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            row.put(kvs[i].toString(), kvs[i + 1]);
        }
        return row;
    }

    private static void assertLongArray(long[] actual, long[] expected, String label) {
        require(actual != null, label + " should not be null");
        require(actual.length == expected.length, label + " length mismatch");
        for (int i = 0; i < expected.length; i++) {
            require(actual[i] == expected[i], label + " mismatch at index " + i);
        }
    }

    private static void assertIntArray(int[] actual, int[] expected, String label) {
        require(actual != null, label + " should not be null");
        require(actual.length == expected.length, label + " length mismatch");
        for (int i = 0; i < expected.length; i++) {
            require(actual[i] == expected[i], label + " mismatch at index " + i);
        }
    }

    private static void assertStringArray(String[] actual, String[] expected, String label) {
        require(actual != null, label + " should not be null");
        require(actual.length == expected.length, label + " length mismatch");
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == null) {
                require(actual[i] == null, label + " mismatch at index " + i);
            } else {
                require(expected[i].equals(actual[i]), label + " mismatch at index " + i);
            }
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            throw new IllegalStateException("failed to delete: " + file.getAbsolutePath());
        }
    }
}
