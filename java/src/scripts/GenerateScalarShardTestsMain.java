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
 * scalar resumable session을 보여주는 예제 스크립트입니다.
 *
 * <p>흐름은 다음과 같습니다.</p>
 * <ol>
 *   <li>synthetic scalar X/Y를 streaming source로 준비합니다.</li>
 *   <li>sample/feature metadata parquet를 만듭니다.</li>
 *   <li>첫 번째 session에서 sample 0~19까지만 {@code writeSample(...)}로 기록합니다.</li>
 *   <li>같은 stage directory를 다시 열고 {@code status().pendingSampleIds}로 남은 sample을 확인해 이어서 씁니다.</li>
 *   <li>{@code finishStage()}로 stage manifest를 확정하고 {@code buildShards(...)}로 최종 shard를 만듭니다.</li>
 * </ol>
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
                stageDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                buildConfig)) {
            long nextSampleId = firstPendingSampleId(builder);
            writeSamples(builder, synthetic, nextSampleId, RESUME_SPLIT_SAMPLE);
            System.out.println("paused after sample " + (RESUME_SPLIT_SAMPLE - 1));
        }

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                stageDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                buildConfig)) {
            long nextSampleId = firstPendingSampleId(builder);
            writeSamples(builder, synthetic, nextSampleId, N_SAMPLES);

            String stageManifestPath = builder.finishStage();
            System.out.println("stage manifest: " + stageManifestPath);

            String manifestPath = builder.buildShards(false);
            System.out.println("final manifest: " + manifestPath);
        }
    }

    /**
     * 주어진 sample 구간을 순서대로 builder에 기록합니다.
     *
     * <p>이 예제는 순차적으로 쓰지만, 실제 작업에서는 {@code pendingSampleIds}를 여러
     * worker에게 나눠주고 각 worker가 서로 다른 sample id를 처리해도 됩니다.</p>
     */
    private static void writeSamples(
            ScalarDatasetBuilder builder,
            SyntheticGenerator.StreamingSource synthetic,
            long startSampleId,
            int endExclusiveSampleId) throws Exception {
        for (SyntheticGenerator.SyntheticSample sample : synthetic.samples((int) startSampleId, endExclusiveSampleId)) {
            builder.writeSample((long) sample.sampleId, sample.values, true);
        }
    }

    private static long firstPendingSampleId(ScalarDatasetBuilder builder) throws Exception {
        List<Long> pending = builder.status().pendingSampleIds;
        return pending.isEmpty() ? N_SAMPLES : pending.get(0).longValue();
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
