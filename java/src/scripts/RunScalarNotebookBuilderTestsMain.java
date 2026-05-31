package scripts;

import fs.config.BuildShardConfig;
import fs.config.SyntheticConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarDenseLongDataset;
import fs.io.ScalarFeatureShards;
import fs.model.selection.Candidate;
import fs.model.scalar.ScalarDenseLongManifest;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;
import fs.model.synthetic.SyntheticData;
import fs.synth.SyntheticGenerator;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * check_scalar.ipynb의 builder test 흐름을 Java dense-long API로 검증한다.
 */
public final class RunScalarNotebookBuilderTestsMain {
    private static final int N_SAMPLES = 1000;
    private static final int N_FEATURES = 1024;
    private static final int QUERY_SAMPLE_COUNT = 50;
    private static final long SEED = 100L;
    private static final String[] Y_COLS = new String[]{"y1", "y2", "y3"};

    private RunScalarNotebookBuilderTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_scalar_notebook_builder_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        SyntheticConfig syntheticConfig = new SyntheticConfig();
        syntheticConfig.nSamples = N_SAMPLES;
        syntheticConfig.nFeatures = N_FEATURES;
        syntheticConfig.seed = SEED;
        SyntheticData data = SyntheticGenerator.generate(syntheticConfig);

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        List<String> featureKeys = buildFeatureKeys(N_FEATURES);
        ScalarFeatureShards.writeSampleMeta(sampleRows(data), sampleMetaPath);
        ScalarFeatureShards.writeFeatureMeta(featureRows(featureKeys), featureMetaPath);

        BuildShardConfig buildConfig = new BuildShardConfig();
        buildConfig.targetShardBytes = 4L * 1024L * 1024L;
        buildConfig.yCol = Y_COLS[0];
        buildConfig.statsYCols = Arrays.asList(Y_COLS);
        buildConfig.denseLongRowGroupFeatures = 128;

        String outDir = new File(root, "scalar_dense_long").getAbsolutePath();
        String stageDir = new File(root, "sample_major_stage").getAbsolutePath();
        String manifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                buildConfig,
                stageDir)) {
            require(builder.status().nextExpectedSampleId == 0L, "new session should start from sample 0");
            for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
                LinkedHashMap<Object, Object> values = new LinkedHashMap<Object, Object>(N_FEATURES);
                for (int featureId = 0; featureId < N_FEATURES; featureId++) {
                    values.put(featureKeys.get(featureId), Double.valueOf(data.X[featureId][sampleId]));
                }
                builder.writeSample((long) sampleId, values);
            }
            String stageManifestPath = builder.finishStage();
            require(new File(stageManifestPath).exists(), "missing sample-major stage manifest");
            manifestPath = builder.buildShards(true);
        }

        ScalarDenseLongManifest manifest = ScalarFeatureShards.loadManifest(manifestPath);
        require(manifest.nSamples == N_SAMPLES, "nSamples mismatch");
        require(manifest.nFeatures == N_FEATURES, "nFeatures mismatch");
        for (String yCol : Y_COLS) {
            String statsPath = manifest.selectionStatsPath(yCol);
            require(statsPath != null && !statsPath.isEmpty(), "missing selection stats for " + yCol);
            require(new File(statsPath).exists(), "selection stats file missing for " + yCol);
        }

        Random rng = new Random(SEED);
        int[] queryFeatureIds = shuffledFeatureIds(N_FEATURES, rng);
        long[] querySampleIds = shuffledSampleIds(N_SAMPLES, QUERY_SAMPLE_COUNT, rng);

        try (ScalarDenseLongDataset dataset = ScalarFeatureShards.open(manifestPath)) {
            ScalarFeatureValues feature0 = dataset.loadFeatureByKey(featureKeys.get(0));
            assertExpectedValue(feature0.values.get(0), 0, 0, data);

            for (int i = 0; i < 32; i++) {
                int featureId = queryFeatureIds[i];
                ScalarFeatureValues featureValues = dataset.loadFeatureById(featureId);
                require(featureValues.featureId == featureId, "feature id mismatch");
                require(featureKeys.get(featureId).equals(featureValues.featureKey), "feature key mismatch");
                for (long sampleId : querySampleIds) {
                    assertExpectedValue(featureValues.values.get((int) sampleId), featureId, (int) sampleId, data);
                }
            }

            ScalarDenseLongDataset.SampleValues sample = dataset.loadSampleById(querySampleIds[0]);
            for (int i = 0; i < 32; i++) {
                int featureId = queryFeatureIds[i];
                boolean expectedPresent = data.M[featureId][(int) querySampleIds[0]] != 0;
                require((sample.valid[featureId] != 0) == expectedPresent, "sample mask mismatch");
            }

            List<Candidate> candidates = dataset.topFeaturesFromStats(Y_COLS[0], 32);
            require(!candidates.isEmpty(), "selection candidates should not be empty");
            assertCandidatesSorted(candidates);
        }

        System.out.println("java scalar notebook-style builder tests passed");
    }

    private static List<Map<String, Object>> sampleRows(SyntheticData data) {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>(N_SAMPLES);
        for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
            rows.add(row(
                    "sample_key", sampleKey(sampleId),
                    "y1", data.y[sampleId],
                    "y2", -data.y[sampleId],
                    "y3", 0.5 * data.y[sampleId] + Math.sin(sampleId * 0.01),
                    "split", (sampleId % 5 == 0) ? "test" : "train"
            ));
        }
        return rows;
    }

    private static List<Map<String, Object>> featureRows(List<String> featureKeys) {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>(featureKeys.size());
        for (int featureId = 0; featureId < featureKeys.size(); featureId++) {
            rows.add(row(
                    "feature_key", featureKeys.get(featureId),
                    "group", "group_" + (featureId % 8),
                    "family", "family_" + (featureId % 16)
            ));
        }
        return rows;
    }

    private static List<String> buildFeatureKeys(int nFeatures) {
        ArrayList<String> keys = new ArrayList<String>(nFeatures);
        for (int featureId = 0; featureId < nFeatures; featureId++) {
            keys.add(String.format("feature_%06d", Integer.valueOf(featureId)));
        }
        return keys;
    }

    private static void assertExpectedValue(ScalarValue actual, int featureId, int sampleId, SyntheticData data) {
        require(actual != null, "scalar value should not be null");
        require(actual.sampleId == sampleId, "sample_id mismatch");
        require(sampleKey(sampleId).equals(actual.sampleKey), "sample_key mismatch");
        boolean expectedPresent = data.M[featureId][sampleId] != 0;
        require(actual.present == expectedPresent, "present flag mismatch");
        if (!expectedPresent) {
            require(actual.value == null, "missing scalar value should be null");
            return;
        }
        require(actual.value != null, "present scalar value should not be null");
        long actualBits = Double.doubleToLongBits(actual.value.doubleValue());
        long expectedBits = Double.doubleToLongBits(data.X[featureId][sampleId]);
        require(actualBits == expectedBits, "scalar value mismatch");
    }

    private static void assertCandidatesSorted(List<Candidate> candidates) {
        for (int i = 1; i < candidates.size(); i++) {
            Candidate prev = candidates.get(i - 1);
            Candidate curr = candidates.get(i);
            boolean sorted = prev.r2y > curr.r2y
                    || (Double.doubleToLongBits(prev.r2y) == Double.doubleToLongBits(curr.r2y) && prev.featureId <= curr.featureId);
            require(sorted, "candidate order is not sorted by r2y desc, feature_id asc");
        }
    }

    private static int[] shuffledFeatureIds(int nFeatures, Random rng) {
        int[] ids = new int[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            ids[i] = i;
        }
        shuffle(ids, rng);
        return ids;
    }

    private static long[] shuffledSampleIds(int nSamples, int count, Random rng) {
        int[] ids = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            ids[i] = i;
        }
        shuffle(ids, rng);
        long[] out = new long[count];
        for (int i = 0; i < count; i++) {
            out[i] = ids[i];
        }
        return out;
    }

    private static void shuffle(int[] values, Random rng) {
        for (int i = values.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = values[i];
            values[i] = values[j];
            values[j] = tmp;
        }
    }

    private static String sampleKey(int sampleId) {
        return String.format("sample_%06d", Integer.valueOf(sampleId));
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(kvs[i].toString(), kvs[i + 1]);
        }
        return out;
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
