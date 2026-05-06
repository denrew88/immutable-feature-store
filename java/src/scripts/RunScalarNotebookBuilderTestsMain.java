package scripts;

import fs.config.BuildShardConfig;
import fs.config.SelectionConfig;
import fs.config.SyntheticConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.ScalarShardDataset;
import fs.model.selection.Candidate;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;
import fs.model.scalar.ShardManifest;
import fs.model.synthetic.SyntheticData;
import fs.synth.SyntheticGenerator;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * check_scalar.ipynb의 Builder Test 흐름을 자바 public API로 재현하는 종합 테스트 스크립트다.
 *
 * <p>이 스크립트는 다음 순서로 동작한다.
 * 1) synthetic scalar X/Y 원본 생성
 * 2) sample/feature metadata parquet 작성
 * 3) ScalarDatasetBuilder로 최종 shard build
 * 4) getValues, iterMany 경로로 원본과 복원 결과 비교
 * 5) buildCandidates, selectFeatures 예제 실행
 *
 * <p>기본 크기는 sample 1000개, feature 1024개로 맞춰 두었다.
 * 필요하면 상수만 바꿔서 더 큰 조건으로 다시 돌릴 수 있다.
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

        String outDir = new File(root, "scalar_shards").getAbsolutePath();
        String stageDir = new File(root, "sample_major_stage").getAbsolutePath();
        String manifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.newBuilder(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                buildConfig,
                stageDir)) {
            for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
                LinkedHashMap<Object, Object> values = new LinkedHashMap<Object, Object>(N_FEATURES);
                for (int featureId = 0; featureId < N_FEATURES; featureId++) {
                    values.put(featureKeys.get(featureId), Double.valueOf(data.X[featureId][sampleId]));
                }
                builder.writeSample((long) sampleId, values);
            }
            manifestPath = builder.buildShards(true);
        }

        require(new File(manifestPath).exists(), "missing shard manifest");
        require(new File(stageDir).exists(), "sample-major stage should remain when keepSampleMajor=true");

        ShardManifest manifest = ScalarFeatureShards.loadManifest(manifestPath);
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

        try (ScalarShardDataset dataset = ScalarFeatureShards.open(manifestPath)) {
            ScalarValue single = dataset.getValueByKey(featureKeys.get(0), sampleKey(0));
            assertExpectedValue(single, 0, 0, data);

            verifyGetValues(dataset, queryFeatureIds, querySampleIds, data, featureKeys);
            verifyIterMany(dataset, queryFeatureIds, querySampleIds, data, featureKeys, false, 128);
            verifyIterMany(dataset, queryFeatureIds, querySampleIds, data, featureKeys, true, 8);
        }

        SelectionConfig selectionConfig = new SelectionConfig(
                0.01,
                100,
                0.95,
                100,
                16,
                256,
                256,
                128,
                0,
                0
        );

        List<Candidate> candidates = ScalarFeatureShards.buildCandidates(manifestPath, Y_COLS[0], selectionConfig);
        require(!candidates.isEmpty(), "selection candidates should not be empty");
        assertCandidatesSorted(candidates);

        List<Candidate> selected = ScalarFeatureShards.selectFeatures(manifestPath, Y_COLS[0], selectionConfig);
        require(!selected.isEmpty(), "selected feature list should not be empty");
        require(selected.size() <= selectionConfig.topM, "selected feature count exceeds topM");

        HashSet<Integer> candidateFeatureIds = new HashSet<Integer>();
        for (Candidate candidate : candidates) {
            candidateFeatureIds.add(Integer.valueOf(candidate.featureId));
        }
        for (Candidate candidate : selected) {
            require(candidateFeatureIds.contains(Integer.valueOf(candidate.featureId)), "selected feature is not present in candidate set");
        }

        System.out.println("java scalar notebook-style builder tests passed");
    }

    private static List<Map<String, Object>> sampleRows(SyntheticData data) {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>(N_SAMPLES);
        for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
            double y1 = data.y[sampleId];
            double y2 = -data.y[sampleId];
            double y3 = 0.5 * data.y[sampleId] + Math.sin(sampleId * 0.01);
            rows.add(row(
                    "sample_key", sampleKey(sampleId),
                    "y1", y1,
                    "y2", y2,
                    "y3", y3,
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

    private static void verifyGetValues(
            ScalarShardDataset dataset,
            int[] queryFeatureIds,
            long[] querySampleIds,
            SyntheticData data,
            List<String> featureKeys) throws Exception {
        for (int featureId : queryFeatureIds) {
            ScalarFeatureValues batch = dataset.getValues(featureId, querySampleIds);
            assertFeatureBatch(batch, featureId, querySampleIds, data, featureKeys);
        }
    }

    private static void verifyIterMany(
            ScalarShardDataset dataset,
            int[] queryFeatureIds,
            long[] querySampleIds,
            SyntheticData data,
            List<String> featureKeys,
            boolean maintainOrder,
            int batchSize) throws Exception {
        int position = 0;
        for (ScalarFeatureValues batch : dataset.iterMany(queryFeatureIds, querySampleIds, batchSize, maintainOrder)) {
            assertFeatureBatch(batch, batch.featureId, querySampleIds, data, featureKeys);
            if (maintainOrder) {
                require(batch.featureId == queryFeatureIds[position], "iterMany maintainOrder=true feature order mismatch");
            }
            position += 1;
        }
        require(position == queryFeatureIds.length, "iterMany did not yield expected feature count");
    }

    private static void assertFeatureBatch(
            ScalarFeatureValues batch,
            int expectedFeatureId,
            long[] querySampleIds,
            SyntheticData data,
            List<String> featureKeys) {
        require(batch.featureId == expectedFeatureId, "feature id mismatch");
        require(featureKeys.get(expectedFeatureId).equals(batch.featureKey), "feature key mismatch");
        require(batch.values.size() == querySampleIds.length, "sample count mismatch in feature batch");
        for (int i = 0; i < querySampleIds.length; i++) {
            assertExpectedValue(batch.values.get(i), expectedFeatureId, (int) querySampleIds[i], data);
        }
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
        double expectedValue = data.X[featureId][sampleId];
        long actualBits = Double.doubleToLongBits(actual.value.doubleValue());
        long expectedBits = Double.doubleToLongBits(expectedValue);
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
