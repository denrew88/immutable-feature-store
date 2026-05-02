package scripts;

import fs.config.BuildShardConfig;
import fs.config.SelectionConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.ScalarShardDataset;
import fs.model.Candidate;
import fs.model.ScalarFeatureValues;
import fs.model.ScalarValue;
import fs.model.ShardManifest;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class RunScalarBuilderTestsMain {
    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_scalar_builder_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        ScalarFeatureShards.writeSampleMeta(sampleRows(), sampleMetaPath);

        BuildShardConfig cfg = new BuildShardConfig();
        cfg.targetShardBytes = 1L << 20;
        cfg.statsYCols = java.util.Arrays.asList("y", "y_alt");

        String discoveredManifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.newBuilder(
                new File(root, "discovered_shards").getAbsolutePath(),
                sampleMetaPath,
                "",
                null,
                cfg,
                new File(root, "discovered_stage").getAbsolutePath())) {
            builder.writeSample(0L, values("feature_a", 1.0, "feature_b", 0.1));
            try (ScalarDatasetBuilder.ScalarSampleContext sample1 = builder.openSample(1L)) {
                sample1.writeValue("feature_b", 1.0);
                sample1.writeValue("feature_c", 0.2);
            }
            builder.writeSample(2L, values("feature_a", 0.0, "feature_c", 1.0));
            builder.writeSample(3L, values("feature_a", 0.8, "feature_b", 0.2, "feature_c", 0.9));
            String stageManifestPath = builder.finishSampleMajor();
            require(new File(stageManifestPath).exists(), "missing sample-major manifest");
            builder.updateFeatureMeta(featureRows(), "feature_key", true);
            discoveredManifestPath = builder.buildShards(true);
        }

        ShardManifest discoveredManifest = ScalarFeatureShards.loadManifest(discoveredManifestPath);
        require(discoveredManifest.nSamples == 4, "n_samples mismatch");
        require(discoveredManifest.nFeatures == 3, "n_features mismatch");
        require(!discoveredManifest.selectionStats.isEmpty(), "selection_stats should be populated");
        require(discoveredManifest.selectionStatsPath("y") != null && !discoveredManifest.selectionStatsPath("y").isEmpty(), "missing y selection stats");
        require(new File(discoveredManifest.featureMetaPath).exists(), "missing feature_meta.parquet");
        require(new File(discoveredManifest.sampleMetaPath).exists(), "missing sample_meta.parquet");

        try (ScalarShardDataset dataset = ScalarFeatureShards.open(discoveredManifestPath)) {
            ScalarFeatureValues featureB = dataset.getValuesByKeys("feature_b", new String[]{"sample_000000", "sample_000001", "sample_000002", "sample_000003"});
            require(featureB.featureId == 1, "feature_b id mismatch");
            assertValue(featureB.values.get(0), 0L, "sample_000000", true, 0.1);
            assertValue(featureB.values.get(1), 1L, "sample_000001", true, 1.0);
            assertValue(featureB.values.get(2), 2L, "sample_000002", false, null);
            assertValue(featureB.values.get(3), 3L, "sample_000003", true, 0.2);

            ScalarValue single = dataset.getValueByKey("feature_c", "sample_000002");
            assertValue(single, 2L, "sample_000002", true, 1.0);
        }

        SelectionConfig selCfg = new SelectionConfig();
        selCfg.topM = 2;
        selCfg.minNonNullY = 1;
        selCfg.minNonNullPair = 1;
        selCfg.yR2Threshold = -1.0;
        selCfg.ffR2Threshold = 1.1;
        List<Candidate> selected = ScalarFeatureShards.selectFeatures(discoveredManifestPath, "y", selCfg);
        require(!selected.isEmpty(), "selection should return at least one feature");

        String featureMetaPath = new File(root, "known_feature_meta.parquet").getAbsolutePath();
        ScalarFeatureShards.writeFeatureMeta(knownFeatureRows(), featureMetaPath);
        String knownManifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.newBuilder(
                new File(root, "known_shards").getAbsolutePath(),
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg,
                new File(root, "known_stage").getAbsolutePath())) {
            builder.writeSample(0L, values("feature_x", 1.0));
            builder.writeSample(1L, values(Integer.valueOf(1), 2.0));
            knownManifestPath = builder.buildShards(false);
        }
        require(new File(knownManifestPath).exists(), "known-feature manifest missing");
        require(!new File(root, "known_stage").exists(), "known-stage dir should be removed when keepSampleMajor=false");

        System.out.println("java scalar builder tests passed");
    }

    private static List<Map<String, Object>> sampleRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        rows.add(row("sample_key", "sample_000000", "y", 0.0, "y_alt", 1.0, "split", "train"));
        rows.add(row("sample_key", "sample_000001", "y", 1.0, "y_alt", 0.0, "split", "train"));
        rows.add(row("sample_key", "sample_000002", "y", 0.0, "y_alt", 1.0, "split", "test"));
        rows.add(row("sample_key", "sample_000003", "y", 1.0, "y_alt", 0.0, "split", "test"));
        return rows;
    }

    private static List<Map<String, Object>> featureRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        rows.add(row("feature_key", "feature_a", "group", "alpha"));
        rows.add(row("feature_key", "feature_b", "group", "beta"));
        rows.add(row("feature_key", "feature_c", "group", "gamma"));
        return rows;
    }

    private static List<Map<String, Object>> knownFeatureRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        rows.add(row("feature_key", "feature_x", "group", "known"));
        rows.add(row("feature_key", "feature_y", "group", "known"));
        return rows;
    }

    private static LinkedHashMap<Object, Object> values(Object... kvs) {
        LinkedHashMap<Object, Object> out = new LinkedHashMap<Object, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(kvs[i], kvs[i + 1]);
        }
        return out;
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(kvs[i].toString(), kvs[i + 1]);
        }
        return out;
    }

    private static void assertValue(ScalarValue actual, long sampleId, String sampleKey, boolean present, Double value) {
        require(actual != null, "value result should not be null");
        require(actual.sampleId == sampleId, "sample_id mismatch");
        require(sampleKey.equals(actual.sampleKey), "sample_key mismatch");
        require(actual.present == present, "present flag mismatch for sample_id=" + sampleId);
        if (!present) {
            require(actual.value == null, "missing value should be null");
            return;
        }
        require(actual.value != null, "present value should not be null");
        require(Math.abs(actual.value.doubleValue() - value.doubleValue()) <= 1e-12, "value mismatch for sample_id=" + sampleId);
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
