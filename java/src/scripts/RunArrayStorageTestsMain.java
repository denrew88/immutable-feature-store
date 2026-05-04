package scripts;

import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.io.ArrayFeatureFlags;
import fs.io.ArrayFeatureLocatorIndex;
import fs.io.ArraySampleBundleWriter;
import fs.io.ArrayShardBuilder;
import fs.io.ArrayShardManifestIO;
import fs.io.ArrayBinaryShardReader;
import fs.io.DuckDBUtils;
import fs.model.LogicalType;
import fs.model.ArrayShardManifest;
import fs.model.ArrayTrace;
import fs.model.PointColumnSpec;
import fs.model.StorageType;

import java.io.File;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class RunArrayStorageTestsMain {
    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_array_storage_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String bundleManifestPath;
        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        writeSampleMeta(sampleMetaPath, 10);
        writeFeatureMeta(featureMetaPath, 2);
        ArrayBundleConfig bundleCfg = new ArrayBundleConfig();
        bundleCfg.maxBundleRows = 3;
        bundleCfg.maxBundleBytes = 1L << 20;
        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS)
        );
        try (ArraySampleBundleWriter writer = new ArraySampleBundleWriter(root.getAbsolutePath(), sampleMetaPath, featureMetaPath, 10, bundleCfg, pointSchema)) {
            writer.appendTrace(0L, 0, columns(new double[]{0.0, 1.0, 2.0}, new double[]{10.0, 11.0, 12.0}));
            writer.appendTrace(2L, 0, columns(new double[]{0.0, 1.0}, new double[]{20.0, 21.0}));
            writer.appendTrace(5L, 0, columns(new double[0], new double[0]));
            writer.appendTrace(8L, 0, columns(new double[]{0.0, Double.NaN}, new double[]{80.0, Double.POSITIVE_INFINITY}));
            writer.appendTrace(1L, 1, columns(new double[]{1.0, 2.0, 3.0}, new double[]{1.0, 2.0, 3.0}));
            writer.appendTrace(7L, 1, columns(new double[]{5.0}, new double[]{7.0}));
            bundleManifestPath = writer.finish();
        }

        ArrayShardConfig shardCfg = new ArrayShardConfig();
        shardCfg.nShards = 0;
        shardCfg.targetShardBytes = 1L << 20;
        shardCfg.samplesPerBlock = 4;
        String shardManifestPath = ArrayShardBuilder.buildFromBundles(
                bundleManifestPath,
                new File(root, "shards").getAbsolutePath(),
                shardCfg);

        ArrayShardManifest manifest = ArrayShardManifestIO.read(shardManifestPath);
        require(manifest.samplesPerBlock == 4, "samples_per_block mismatch");
        require(manifest.nFeatures == 2, "n_features mismatch");
        require(manifest.nShards >= 1, "expected at least one shard");
        ArrayFeatureLocatorIndex locatorIndex = ArrayFeatureLocatorIndex.load(manifest);
        require(locatorIndex.blocksForFeature(0).size() == 3, "feature 0 should have 3 blocks");
        require(locatorIndex.blocksForFeature(1).size() == 3, "feature 1 should have 3 blocks");
        require(new File(manifest.blocksIndexPath(0)).exists(), "missing blocks.idx");
        require(new File(manifest.blocksDataPath(0)).exists(), "missing blocks.bin");
        require(new File(manifest.sampleMetaPath).exists(), "missing sample_meta.parquet");
        require(new File(manifest.featureMetaPath).exists(), "missing feature_meta.parquet");

        try (ArrayBinaryShardReader reader = new ArrayBinaryShardReader(manifest)) {
            Map<Long, ArrayTrace> traces0 = reader.loadFeatureSamples(0, new long[]{0L, 1L, 2L, 5L, 8L, 9L}, locatorIndex);
            assertTrace(traces0.get(0L), ArrayFeatureFlags.PRESENT, new double[]{0.0, 1.0, 2.0}, new double[]{10.0, 11.0, 12.0});
            assertTrace(traces0.get(1L), (byte) 0, new double[0], new double[0]);
            assertTrace(traces0.get(2L), ArrayFeatureFlags.PRESENT, new double[]{0.0, 1.0}, new double[]{20.0, 21.0});
            assertTrace(traces0.get(5L), (byte) (ArrayFeatureFlags.PRESENT | ArrayFeatureFlags.EMPTY), new double[0], new double[0]);
            assertTrace(traces0.get(8L), ArrayFeatureFlags.PRESENT,
                    new double[]{0.0, Double.NaN},
                    new double[]{80.0, Double.POSITIVE_INFINITY});
            assertTrace(traces0.get(9L), (byte) 0, new double[0], new double[0]);

            Map<Long, ArrayTrace> traces1 = reader.loadFeatureSamples(1, new long[]{1L, 7L}, locatorIndex);
            assertTrace(traces1.get(1L), ArrayFeatureFlags.PRESENT, new double[]{1.0, 2.0, 3.0}, new double[]{1.0, 2.0, 3.0});
            assertTrace(traces1.get(7L), ArrayFeatureFlags.PRESENT, new double[]{5.0}, new double[]{7.0});
        }

        System.out.println("array storage tests passed");
    }

    private static void assertTrace(ArrayTrace actual, byte expectedFlags, double[] expectedTime, double[] expectedValue) {
        require(actual != null, "trace should not be null");
        require(actual.flags == expectedFlags, "flags mismatch for sample_id=" + actual.sampleId);
        double[] actualTime = (double[]) actual.columns.get("time");
        double[] actualValue = (double[]) actual.columns.get("value");
        require(actualTime.length == expectedTime.length, "time length mismatch for sample_id=" + actual.sampleId);
        require(actualValue.length == expectedValue.length, "value length mismatch for sample_id=" + actual.sampleId);
        for (int i = 0; i < expectedTime.length; i++) {
            assertDoubleEquals(actualTime[i], expectedTime[i], "time[" + i + "] sample_id=" + actual.sampleId);
        }
        for (int i = 0; i < expectedValue.length; i++) {
            assertDoubleEquals(actualValue[i], expectedValue[i], "value[" + i + "] sample_id=" + actual.sampleId);
        }
    }

    private static LinkedHashMap<String, Object> columns(double[] time, double[] value) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        out.put("time", time);
        out.put("value", value);
        return out;
    }

    private static void writeSampleMeta(String sampleMetaPath, int nSamples) throws Exception {
        try (Connection conn = DuckDBUtils.connect(null)) {
            try (Statement st = conn.createStatement()) {
                st.execute("CREATE TEMP TABLE tmp_sample_meta (sample_id BIGINT, sample_key VARCHAR)");
            }
            try (PreparedStatement ps = conn.prepareStatement("INSERT INTO tmp_sample_meta VALUES (?, ?)")) {
                for (int sampleId = 0; sampleId < nSamples; sampleId++) {
                    ps.setLong(1, sampleId);
                    ps.setString(2, String.format("sample_%06d", sampleId));
                    ps.addBatch();
                }
                ps.executeBatch();
            }
            try (Statement st = conn.createStatement()) {
                st.execute("COPY tmp_sample_meta TO " + DuckDBUtils.quotePath(sampleMetaPath) + " (FORMAT PARQUET)");
            }
        }
    }

    private static void writeFeatureMeta(String featureMetaPath, int nFeatures) throws Exception {
        try (Connection conn = DuckDBUtils.connect(null)) {
            try (Statement st = conn.createStatement()) {
                st.execute("CREATE TEMP TABLE tmp_feature_meta (feature_id INTEGER, feature_key VARCHAR)");
            }
            try (PreparedStatement ps = conn.prepareStatement("INSERT INTO tmp_feature_meta VALUES (?, ?)")) {
                for (int featureId = 0; featureId < nFeatures; featureId++) {
                    ps.setInt(1, featureId);
                    ps.setString(2, String.format("feature_%06d", featureId));
                    ps.addBatch();
                }
                ps.executeBatch();
            }
            try (Statement st = conn.createStatement()) {
                st.execute("COPY tmp_feature_meta TO " + DuckDBUtils.quotePath(featureMetaPath) + " (FORMAT PARQUET)");
            }
        }
    }

    private static void assertDoubleEquals(double actual, double expected, String label) {
        if (Double.isNaN(expected)) {
            require(Double.isNaN(actual), label + " expected NaN");
            return;
        }
        if (Double.isInfinite(expected)) {
            require(Double.isInfinite(actual) && Math.signum(actual) == Math.signum(expected), label + " expected infinity");
            return;
        }
        require(Math.abs(actual - expected) <= 1e-9, label + " mismatch");
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
