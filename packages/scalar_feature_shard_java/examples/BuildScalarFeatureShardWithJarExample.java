import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarDenseLongDataset;
import fs.io.ScalarFeatureShards;
import fs.model.scalar.ScalarBuildSessionStatus;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * scalar-feature-shard-java jar만 classpath에 넣어서 sample meta, feature meta,
 * resumable raw sample stage, dense-long scalar shard를 만드는 end-to-end 예제입니다.
 *
 * <p>기본 출력 위치는 {@code data/tmp_scalar_feature_shard_jar_example}입니다.
 * 첫 번째 인자로 출력 root directory를 넘기면 해당 경로를 사용합니다.</p>
 */
public class BuildScalarFeatureShardWithJarExample {
    public static void main(String[] args) throws Exception {
        File root = new File(args.length > 0 ? args[0] : "data/tmp_scalar_feature_shard_jar_example").getAbsoluteFile();
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create root dir: " + root.getAbsolutePath());
        }

        // 1. sample metadata를 만듭니다. row 순서가 dense sample_id가 됩니다.
        String sampleMetaPath = ScalarFeatureShards.writeSampleMeta(
                Arrays.<Map<String, Object>>asList(
                        row("sample_key", "sample_000000", "split", "train", "y", 1.0),
                        row("sample_key", "sample_000001", "split", "train", "y", 2.0),
                        row("sample_key", "sample_000002", "split", "valid", "y", 3.0),
                        row("sample_key", "sample_000003", "split", "valid", "y", 4.0)
                ),
                new File(root, "sample_meta.parquet").getAbsolutePath()
        );

        // 2. feature metadata를 만듭니다. row 순서가 dense feature_id가 됩니다.
        String featureMetaPath = ScalarFeatureShards.writeFeatureMeta(
                Arrays.<Map<String, Object>>asList(
                        row("feature_key", "feature_a", "group", "alpha"),
                        row("feature_key", "feature_b", "group", "beta"),
                        row("feature_key", "feature_c", "group", "gamma")
                ),
                new File(root, "feature_meta.parquet").getAbsolutePath()
        );

        BuildShardConfig config = new BuildShardConfig();
        config.targetShardBytes = 16L * 1024L * 1024L;
        config.statsYCols = Arrays.asList("y");
        config.denseLongRowGroupFeatures = 128;

        File stageDir = new File(root, "scalar_stage");
        File denseDir = new File(root, "scalar_dense_long");

        // 3. 첫 실행에서는 일부 sample만 완료하고 종료한 상황을 가정합니다.
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                stageDir.getAbsolutePath(),
                sampleMetaPath,
                featureMetaPath,
                null,
                config)) {
            builder.writeSample(2L, row("feature_b", 20.0), true);
            builder.writeSample(0L, row("feature_a", 10.0, "feature_c", null), true);
            System.out.println("first_pending=" + builder.status().pendingSampleIds);
        }

        String denseManifestPath;

        // 4. 같은 session을 다시 열면 raw_samples.jsonl을 읽고 남은 sample만 이어서 씁니다.
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                stageDir.getAbsolutePath(),
                sampleMetaPath,
                featureMetaPath,
                null,
                config)) {
            ScalarBuildSessionStatus status = builder.status();
            for (Long sampleId : status.pendingSampleIds) {
                if (sampleId.longValue() == 1L) {
                    builder.writeSample(sampleId.longValue(), row("feature_a", 11.0, "feature_b", 21.0), true);
                } else {
                    builder.writeSample(sampleId.longValue(), row("feature_c", 40.0), true);
                }
            }
            denseManifestPath = builder.buildDenseLongShards(true, denseDir.getAbsolutePath(), true);
        }

        // 5. dense-long reader로 결과를 확인합니다.
        try (ScalarDenseLongDataset dataset = ScalarFeatureShards.openDenseLong(denseManifestPath)) {
            ScalarFeatureValues featureA = dataset.loadFeatureByKey("feature_a");
            System.out.println("feature_key=" + featureA.featureKey + ", feature_id=" + featureA.featureId);
            for (ScalarValue value : featureA.values) {
                System.out.println(
                        "sample_id=" + value.sampleId
                                + ", sample_key=" + value.sampleKey
                                + ", present=" + value.present
                                + ", value=" + value.value);
            }
            System.out.println("top_features=" + dataset.topFeaturesFromStats("y", 2).size());
        }

        System.out.println("sample_meta=" + sampleMetaPath);
        System.out.println("feature_meta=" + featureMetaPath);
        System.out.println("dense_manifest=" + denseManifestPath);
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            row.put(kvs[i].toString(), kvs[i + 1]);
        }
        return row;
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
