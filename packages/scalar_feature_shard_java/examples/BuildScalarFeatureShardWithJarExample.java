import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.ScalarShardDataset;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * scalar-feature-shard-java jar만 classpath에 넣어서 sample meta, feature meta,
 * scalar feature shard dataset을 생성하는 최소 예제다.
 *
 * <p>실행 결과는 기본적으로 data/tmp_scalar_feature_shard_jar_example 아래에 생성된다.
 * 첫 번째 인자로 출력 root directory를 넘기면 그 경로를 사용한다.
 */
public class BuildScalarFeatureShardWithJarExample {
    public static void main(String[] args) throws Exception {
        File root = new File(args.length > 0 ? args[0] : "data/tmp_scalar_feature_shard_jar_example").getAbsoluteFile();
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create root dir: " + root.getAbsolutePath());
        }

        // 1. sample metadata를 만든다. row 순서가 dense sample_id가 된다.
        String sampleMetaPath = ScalarFeatureShards.writeSampleMeta(
                Arrays.<Map<String, Object>>asList(
                        row("sample_key", "sample_000000", "split", "train", "y", 1.0),
                        row("sample_key", "sample_000001", "split", "train", "y", 2.0),
                        row("sample_key", "sample_000002", "split", "valid", "y", 3.0)
                ),
                new File(root, "sample_meta.parquet").getAbsolutePath()
        );

        // 2. feature metadata를 만든다. row 순서가 dense feature_id가 된다.
        String featureMetaPath = ScalarFeatureShards.writeFeatureMeta(
                Arrays.<Map<String, Object>>asList(
                        row("feature_key", "feature_a", "group", "alpha"),
                        row("feature_key", "feature_b", "group", "beta"),
                        row("feature_key", "feature_c", "group", "gamma")
                ),
                new File(root, "feature_meta.parquet").getAbsolutePath()
        );

        BuildShardConfig config = new BuildShardConfig();
        config.targetShardBytes = 1024L * 1024L;
        config.statsYCols = Arrays.asList("y");

        File shardDir = new File(root, "scalar_shards");
        File stageDir = new File(root, "scalar_stage");
        String manifestPath;

        // 3. sample 순서대로 scalar 값을 넣고 shard를 만든다.
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                shardDir.getAbsolutePath(),
                sampleMetaPath,
                featureMetaPath,
                null,
                config,
                stageDir.getAbsolutePath())) {
            long start = builder.status().nextExpectedSampleId;
            for (long sampleId = start; sampleId < 3L; sampleId++) {
                if (sampleId == 0L) {
                    builder.writeSample(sampleId, row(
                            "feature_a", 1.25,
                            "feature_b", 10.0));
                } else if (sampleId == 1L) {
                    builder.writeSample(sampleId, row(
                            "feature_a", 2.50,
                            "feature_c", 20.0));
                } else {
                    builder.writeSample(sampleId, row(
                            "feature_b", 30.0,
                            "feature_c", 40.0));
                }
            }
            builder.finishStage();
            manifestPath = builder.buildShards(false);
        }

        // 4. jar reader로 key 기반 조회를 검증한다.
        try (ScalarShardDataset dataset = ScalarFeatureShards.open(manifestPath)) {
            ScalarFeatureValues featureA = dataset.getValuesByKeys(
                    "feature_a",
                    new String[]{"sample_000000", "sample_000001", "sample_000002"});
            System.out.println("feature_key=" + featureA.featureKey + ", feature_id=" + featureA.featureId);
            for (ScalarValue value : featureA.values) {
                System.out.println(
                        "sample_id=" + value.sampleId
                                + ", sample_key=" + value.sampleKey
                                + ", present=" + value.present
                                + ", value=" + value.value);
            }
        }

        System.out.println("sample_meta=" + sampleMetaPath);
        System.out.println("feature_meta=" + featureMetaPath);
        System.out.println("manifest=" + manifestPath);
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
