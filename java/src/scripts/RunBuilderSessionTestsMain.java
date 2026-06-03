package scripts;

import com.fasterxml.jackson.databind.JsonNode;
import fs.config.ArrayBinaryBuildOptions;
import fs.config.BuildShardConfig;
import fs.io.ArrayBinaryShards;
import fs.io.ArrayDatasetBuilder;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.common.JsonUtils;
import fs.model.array.ArrayBuildSessionStatus;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;
import fs.model.scalar.ScalarBuildSessionStatus;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * array/scalar resumable session 경로를 전용으로 검증하는 스크립트다.
 *
 * <p>builder correctness 전체 테스트와 별도로,
 * 1) 새 세션 상태
 * 2) close() 후 committed watermark 저장
 * 3) 재오픈 후 resume 지점 복원
 * 4) stage finalize와 shard build
 * 를 좁게 확인한다.
 */
public final class RunBuilderSessionTestsMain {
    private RunBuilderSessionTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_builder_session_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        runScalarSessionTest(new File(root, "scalar"));
        runArraySessionTest(new File(root, "array"));
        System.out.println("java builder session tests passed");
    }

    private static void runScalarSessionTest(File root) throws Exception {
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create scalar test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        ScalarFeatureShards.writeSampleMeta(Arrays.<Map<String, Object>>asList(
                row("sample_key", "sample_000000", "y", 0.0),
                row("sample_key", "sample_000001", "y", 1.0),
                row("sample_key", "sample_000002", "y", 0.0)
        ), sampleMetaPath);
        ScalarFeatureShards.writeFeatureMeta(Arrays.<Map<String, Object>>asList(
                row("feature_key", "feature_a"),
                row("feature_key", "feature_b")
        ), featureMetaPath);

        String outDir = new File(root, "scalar_shards").getAbsolutePath();
        File stateFile = new File(outDir, "raw_state.json");
        File logFile = new File(outDir, "raw_samples.jsonl");
        File rawSamplesDir = new File(outDir, "raw_samples");

        BuildShardConfig cfg = new BuildShardConfig();
        cfg.targetShardBytes = 1L << 20;

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            ScalarBuildSessionStatus status0 = builder.status();
            require(status0.completedSampleCount == 0, "new scalar session should not have committed samples");
            require(status0.pendingSampleIds.equals(Arrays.asList(0L, 1L, 2L)), "new scalar pending ids mismatch");
            builder.writeSample(0L, values("feature_a", 1.0));
            builder.writeSample(1L, values("feature_b", 2.0));
            ScalarBuildSessionStatus status1 = builder.status();
            require(status1.completedSampleCount == 2, "scalar session should commit per sample");
            require(status1.pendingSampleIds.equals(Arrays.asList(2L)), "scalar pending ids after write mismatch");
        }

        require(stateFile.exists(), "scalar session state.json missing");
        require(logFile.exists(), "scalar session bundles.jsonl missing");
        JsonNode scalarState1 = JsonUtils.readJson(stateFile.getAbsolutePath());
        List<JsonNode> scalarLog1 = JsonUtils.readJsonLines(logFile.getAbsolutePath());
        require("scalar_raw_stage_v1".equals(scalarState1.path("format").asText()), "scalar raw state format mismatch");
        require(scalarLog1.size() == 2, "scalar committed sample count mismatch");
        require(scalarLog1.get(0).path("sample_id").asLong(-1L) == 0L, "scalar first sample log mismatch");
        require(scalarLog1.get(1).path("sample_id").asLong(-1L) == 1L, "scalar second sample log mismatch");

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            ScalarBuildSessionStatus resumed = builder.status();
            require(resumed.completedSampleCount == 2, "scalar resumed completed count mismatch");
            require(resumed.pendingSampleIds.equals(Arrays.asList(2L)), "scalar resumed pending sample mismatch");
            builder.writeSample(2L, values("feature_a", 3.0, "feature_b", 4.0));
            String stageManifestPath = builder.finishStage();
            require(new File(stageManifestPath).exists(), "scalar stage manifest missing");
            require(builder.status().finishedStage, "scalar stage should be marked finished");
            String shardManifestPath = builder.buildShards(true, true);
            require(new File(shardManifestPath).exists(), "scalar shard manifest missing");
            require(!rawSamplesDir.exists(), "scalar raw_samples should be removed when cleanupRaw=true");
        }
    }

    private static void runArraySessionTest(File root) throws Exception {
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create array test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        ArrayBinaryShards.writeSampleMeta(Arrays.<Map<String, Object>>asList(
                row("sample_key", "sample_000000"),
                row("sample_key", "sample_000001"),
                row("sample_key", "sample_000002")
        ), sampleMetaPath);
        ArrayBinaryShards.writeFeatureMeta(Arrays.<Map<String, Object>>asList(
                row("feature_key", "feature_a"),
                row("feature_key", "feature_b")
        ), featureMetaPath);

        String outDir = new File(root, "array_shards").getAbsolutePath();
        File stageDir = new File(outDir, "bundle_stage");
        File stateFile = new File(stageDir, "state.json");
        File logFile = new File(stageDir, "bundles.jsonl");

        ArrayBinaryBuildOptions opts = new ArrayBinaryBuildOptions();
        opts.samplesPerBlock = 2;
        opts.targetShardMb = 1;
        opts.codec = "none";
        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS)
        );

        try (ArrayDatasetBuilder builder = ArrayBinaryShards.openSession(
                outDir,
                sampleMetaPath,
                pointSchema,
                featureMetaPath,
                opts)) {
            ArrayBuildSessionStatus status0 = builder.status();
            require(status0.lastCommittedSampleId == null, "new array session should not have committed samples");
            require(status0.nextExpectedSampleId == 0L, "new array session should start at sample 0");
            try (ArrayDatasetBuilder.ArraySampleContext sample = builder.sample(0L)) {
                sample.addTrace(null, "feature_a", row("time", new double[]{0.0, 1.0}, "value", new double[]{10.0, 11.0}));
            }
            try (ArrayDatasetBuilder.ArraySampleContext sample = builder.sample(1L)) {
                sample.addTrace(null, "feature_b", row("time", new double[]{0.5}, "value", new double[]{20.0}));
            }
            ArrayBuildSessionStatus status1 = builder.status();
            require(status1.lastCommittedSampleId == null, "array session should buffer before close");
            require(Long.valueOf(1L).equals(status1.bufferedThroughSampleId), "array buffered sample mismatch");
        }

        require(stateFile.exists(), "array session state.json missing");
        require(logFile.exists(), "array session bundles.jsonl missing");
        JsonNode arrayState1 = JsonUtils.readJson(stateFile.getAbsolutePath());
        List<JsonNode> arrayLog1 = JsonUtils.readJsonLines(logFile.getAbsolutePath());
        require(arrayState1.path("last_committed_sample_id").asLong(-1L) == 1L, "array committed watermark mismatch");
        require(arrayState1.path("next_expected_sample_id").asLong(-1L) == 2L, "array next expected mismatch");
        require(arrayLog1.size() == 1, "array committed bundle count mismatch");
        require(arrayLog1.get(0).path("first_sample_id").asLong(-1L) == 0L, "array first sample log mismatch");
        require(arrayLog1.get(0).path("last_sample_id").asLong(-1L) == 1L, "array last sample log mismatch");

        try (ArrayDatasetBuilder builder = ArrayBinaryShards.openSession(
                outDir,
                sampleMetaPath,
                pointSchema,
                featureMetaPath,
                opts)) {
            ArrayBuildSessionStatus resumed = builder.status();
            require(Long.valueOf(1L).equals(resumed.lastCommittedSampleId), "array resumed committed watermark mismatch");
            require(resumed.nextExpectedSampleId == 2L, "array resumed next sample mismatch");
            try (ArrayDatasetBuilder.ArraySampleContext sample = builder.sample(2L)) {
                sample.addTrace(null, "feature_a", row("time", new double[]{2.0}, "value", new double[]{30.0}));
            }
            String stageManifestPath = builder.finishStage();
            require(new File(stageManifestPath).exists(), "array stage manifest missing");
            require(builder.status().finishedStage, "array stage should be marked finished");
            String shardManifestPath = builder.buildShards(false);
            require(new File(shardManifestPath).exists(), "array shard manifest missing");
        }
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(String.valueOf(kvs[i]), kvs[i + 1]);
        }
        return out;
    }

    private static LinkedHashMap<Object, Object> values(Object... kvs) {
        LinkedHashMap<Object, Object> out = new LinkedHashMap<Object, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(kvs[i], kvs[i + 1]);
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
