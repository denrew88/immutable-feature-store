package scripts;

import fs.config.BuildShardConfig;
import fs.config.SyntheticConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.synth.SyntheticGenerator;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * `GenerateArrayShardTestsMain`와 같은 흐름으로 scalar resumable session을 보여 주는 예제 스크립트다.
 *
 * <p>이 예제는 다음 순서로 동작한다.
 * 1) 작은 synthetic scalar 원본 X/Y를 만든다.
 * 2) sample/feature metadata parquet를 쓴다.
 * 3) 첫 번째 session에서 sample 0~19까지만 `writeSample(...)`로 기록한다.
 * 4) 같은 stage 디렉터리를 다시 열어 `status().nextExpectedSampleId`부터 이어서 기록한다.
 * 5) `finishStage()`로 stage manifest를 확정하고 `buildShards(...)`로 최종 shard를 만든다.
 *
 * <p>selection 자체는 실행하지 않지만, 현재 scalar shard build 경로는 sample metadata에 target 컬럼이
 * 있어야 하므로 `y` 컬럼은 함께 기록한다.
 */
public final class GenerateScalarShardTestsMain {
    private static final int N_SAMPLES = 5000;
    private static final int N_FEATURES = 160000;
    private static final int RESUME_SPLIT_SAMPLE = 20;
    private static final long SEED = 10L;

    private GenerateScalarShardTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        File dataRoot = new File("data");
        if (!dataRoot.exists() && !dataRoot.mkdirs()) {
            throw new IllegalStateException("failed to create data dir: " + dataRoot.getAbsolutePath());
        }
        File root = Files.createTempDirectory(dataRoot.toPath(), "tmp_java_scalar_shard_example_").toFile();
        System.out.println("output root: " + root.getAbsolutePath());

        String outDir = new File(root, "scalar_shards").getAbsolutePath();
        String stageDir = new File(root, "scalar_stage").getAbsolutePath();
        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();

        SyntheticConfig syntheticConfig = new SyntheticConfig();
        syntheticConfig.nSamples = N_SAMPLES;
        syntheticConfig.nFeatures = N_FEATURES;
        syntheticConfig.seed = SEED;
        SyntheticGenerator.StreamingSource synthetic = SyntheticGenerator.stream(syntheticConfig);

        List<String> featureKeys = buildFeatureKeys(N_FEATURES);
        ScalarFeatureShards.writeSampleMeta(sampleRows(synthetic), sampleMetaPath);
        ScalarFeatureShards.writeFeatureMeta(featureRows(featureKeys), featureMetaPath);

        BuildShardConfig buildConfig = new BuildShardConfig();
        buildConfig.targetShardBytes = 4L * 1024L * 1024L;
        buildConfig.yCol = "y";
        buildConfig.statsYCols = Arrays.asList("y");

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                buildConfig,
                stageDir)) {
            long nextSampleId = builder.status().nextExpectedSampleId;
            writeSamples(builder, synthetic, nextSampleId, RESUME_SPLIT_SAMPLE);
            System.out.println("paused after sample " + (RESUME_SPLIT_SAMPLE - 1));
        }

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                buildConfig,
                stageDir)) {
            long nextSampleId = builder.status().nextExpectedSampleId;
            writeSamples(builder, synthetic, nextSampleId, N_SAMPLES);

            String stageManifestPath = builder.finishStage();
            System.out.println("stage manifest: " + stageManifestPath);

            String manifestPath = builder.buildShards(false);
            System.out.println("final manifest: " + manifestPath);
        }
    }

    /**
     * 주어진 구간의 sample을 순서대로 builder에 기록한다.
     *
     * <p>resume 예제에서는 반드시 {@code status().nextExpectedSampleId}부터 다시 써야 하므로,
     * caller가 시작 sample을 builder status에서 받아 넘겨주는 형태로 유지한다.
     */
    private static void writeSamples(
            ScalarDatasetBuilder builder,
            SyntheticGenerator.StreamingSource synthetic,
            long startSampleId,
            int endExclusiveSampleId) throws Exception {
        for (SyntheticGenerator.SyntheticSample sample : synthetic.samples((int) startSampleId, endExclusiveSampleId)) {
            builder.writeSample((long) sample.sampleId, sample.values);
        }
    }

    private static List<Map<String, Object>> sampleRows(SyntheticGenerator.StreamingSource synthetic) {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>(N_SAMPLES);
        for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
            rows.add(row(
                    "sample_key", sampleKey(sampleId),
                    "y", Double.valueOf(synthetic.yValue(sampleId)),
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
}
