package scripts;

import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarDenseLongDataset;
import fs.io.ScalarFeatureShards;
import fs.io.common.JsonUtils;
import fs.io.scalar.ScalarSampleMajorManifestIO;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarSampleMajorManifest;
import fs.model.scalar.ScalarValue;

import com.fasterxml.jackson.databind.JsonNode;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * scalar raw stage recovery/finalize-lock 계약 테스트.
 *
 * <p>이 테스트는 구현보다 먼저 추가하는 TDD 계약이다. 현재 raw stage가
 * commit log 누락 orphan parquet를 reconcile하고, finish/build 중 새 write를
 * 막는 stage-level lock을 지원하지 않으면 실패해야 한다.</p>
 */
public final class RunScalarRawStageRecoveryContractTestsMain {
    private static final int RAW_SAMPLE_PADDING = 12;

    private RunScalarRawStageRecoveryContractTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("fs.scalar.rawStageLockTimeoutMillis", "200");
        File root = new File("data/tmp_java_scalar_raw_stage_recovery_contract_tests");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        ArrayList<String> failures = new ArrayList<String>();
        runCase(failures, "testReconcilesFinalParquetsMissingFromCommitLog", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testReconcilesFinalParquetsMissingFromCommitLog(new File(root, "reconcile"));
            }
        });
        runCase(failures, "testFinishStageRefusesActiveSampleLock", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testFinishStageRefusesActiveSampleLock(new File(root, "active_sample_lock"));
            }
        });
        runCase(failures, "testStageLockBlocksNewWrite", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testStageLockBlocksNewWrite(new File(root, "stage_lock"));
            }
        });
        if (!failures.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("java scalar raw stage recovery contract tests failed:");
            for (String failure : failures) {
                sb.append(System.lineSeparator()).append(System.lineSeparator()).append(failure);
            }
            throw new AssertionError(sb.toString());
        }
        System.out.println("java scalar raw stage recovery contract tests passed");
    }

    private static void testReconcilesFinalParquetsMissingFromCommitLog(File root) throws Exception {
        Context ctx = context(root);
        File stageDir = new File(root, "stage");
        try (ScalarDatasetBuilder builder = openBuilder(stageDir, ctx)) {
            builder.writeSample(0L, values("feature_a", 10.0), true);
            builder.writeSample(1L, values("feature_b", 21.0), true);
            builder.writeSample(2L, values(), true); // empty sample도 파일명만으로 완료 상태를 복구해야 한다.
            builder.writeSample(3L, values("feature_c", 33.0), true);
        }

        File rawLog = new File(stageDir, "raw_samples.jsonl");
        dropCommitRecords(rawLog, Arrays.asList(Long.valueOf(1L), Long.valueOf(2L)));
        require(rawSampleFile(stageDir, 1L).exists(), "orphan non-empty sample parquet missing");
        require(rawSampleFile(stageDir, 2L).exists(), "orphan empty sample parquet missing");
        require(readCommitIds(rawLog).equals(Arrays.asList(Long.valueOf(0L), Long.valueOf(3L))), "test setup failed to drop commit rows");

        String finalManifestPath;
        try (ScalarDatasetBuilder recovered = openBuilder(stageDir, ctx)) {
            require(recovered.status().completedSampleIds.equals(allSampleIds()), "reconcile should recover orphan final parquets");
            require(recovered.pendingSampleIds().isEmpty(), "reconcile should clear pending orphan samples");
            require(readCommitIds(rawLog).equals(allSampleIds()), "reconcile should repair raw_samples.jsonl");

            String stageManifestPath = recovered.finishStage(true);
            ScalarSampleMajorManifest stage = ScalarSampleMajorManifestIO.read(stageManifestPath);
            require(stage.sampleIds.equals(allSampleIds()), "stage manifest should include reconciled sample ids");
            finalManifestPath = recovered.buildDenseLongShards(true, new File(root, "scalar_shard").getAbsolutePath(), false);
        }

        assertDirectValues(finalManifestPath);
    }

    private static void testFinishStageRefusesActiveSampleLock(File root) throws Exception {
        Context ctx = context(root);
        File stageDir = new File(root, "stage");
        try (ScalarDatasetBuilder builder = openBuilder(stageDir, ctx)) {
            builder.writeSample(0L, values("feature_a", 10.0), true);
            writeFreshLock(rawSampleLock(stageDir, 1L), "sample");
            expectFailure(new ThrowingRunnable() {
                @Override
                public void run() throws Exception {
                    builder.finishStage(false);
                }
            }, "finishStage(false) must not finalize while an active sample lock exists");
        }
        require(!new File(stageDir, "sample_major_manifest.json").exists(), "finishStage should not materialize manifest under active sample lock");
    }

    private static void testStageLockBlocksNewWrite(File root) throws Exception {
        final Context ctx = context(root);
        final File stageDir = new File(root, "stage");
        try (final ScalarDatasetBuilder builder = openBuilder(stageDir, ctx)) {
            writeFreshLock(rawStageLock(stageDir), "stage");
            expectFailure(new ThrowingRunnable() {
                @Override
                public void run() throws Exception {
                    builder.writeSample(1L, values("feature_b", 21.0), true);
                }
            }, "writeSample must not commit while the raw stage finalize lock is active");
        }
        require(!rawSampleFile(stageDir, 1L).exists(), "writeSample should not create final parquet under active stage lock");
        require(readCommitIds(new File(stageDir, "raw_samples.jsonl")).isEmpty(), "writeSample should not append commit log under active stage lock");
    }

    private static Context context(File root) throws Exception {
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create case dir: " + root.getAbsolutePath());
        }
        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        ScalarFeatureShards.writeSampleMeta(sampleRows(), sampleMetaPath);
        ScalarFeatureShards.writeFeatureMeta(featureRows(), featureMetaPath);
        BuildShardConfig cfg = new BuildShardConfig();
        cfg.targetShardBytes = 1L << 20;
        cfg.statsYCols = Arrays.asList("y");
        return new Context(sampleMetaPath, featureMetaPath, cfg);
    }

    private static ScalarDatasetBuilder openBuilder(File stageDir, Context ctx) throws Exception {
        return ScalarFeatureShards.openSession(
                stageDir.getAbsolutePath(),
                ctx.sampleMetaPath,
                ctx.featureMetaPath,
                null,
                ctx.config);
    }

    private static List<Map<String, Object>> sampleRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        rows.add(row("sample_key", "sample_000000", "y", 1.0));
        rows.add(row("sample_key", "sample_000001", "y", 2.0));
        rows.add(row("sample_key", "sample_000002", "y", 3.0));
        rows.add(row("sample_key", "sample_000003", "y", 4.0));
        return rows;
    }

    private static List<Map<String, Object>> featureRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        rows.add(row("feature_key", "feature_a"));
        rows.add(row("feature_key", "feature_b"));
        rows.add(row("feature_key", "feature_c"));
        return rows;
    }

    private static LinkedHashMap<String, Object> row(Object... kv) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kv.length; i += 2) {
            out.put(kv[i].toString(), kv[i + 1]);
        }
        return out;
    }

    private static LinkedHashMap<Object, Object> values(Object... kv) {
        LinkedHashMap<Object, Object> out = new LinkedHashMap<Object, Object>();
        for (int i = 0; i < kv.length; i += 2) {
            out.put(kv[i], kv[i + 1]);
        }
        return out;
    }

    private static void dropCommitRecords(File rawLog, List<Long> sampleIds) throws Exception {
        ArrayList<String> kept = new ArrayList<String>();
        for (String line : Files.readAllLines(rawLog.toPath(), StandardCharsets.UTF_8)) {
            boolean drop = false;
            for (Long sampleId : sampleIds) {
                if (line.contains("\"sample_id\":" + sampleId) || line.contains("\"sample_id\": " + sampleId)) {
                    drop = true;
                    break;
                }
            }
            if (!drop && !line.trim().isEmpty()) {
                kept.add(line);
            }
        }
        Files.write(rawLog.toPath(), kept, StandardCharsets.UTF_8);
    }

    private static List<Long> readCommitIds(File rawLog) throws Exception {
        ArrayList<Long> out = new ArrayList<Long>();
        if (!rawLog.exists()) {
            return out;
        }
        for (JsonNode node : JsonUtils.readJsonLines(rawLog.getAbsolutePath())) {
            out.add(Long.valueOf(node.path("sample_id").asLong()));
        }
        java.util.Collections.sort(out);
        return out;
    }

    private static void assertDirectValues(String manifestPath) throws Exception {
        try (ScalarDenseLongDataset dataset = ScalarFeatureShards.openDenseLong(manifestPath)) {
            ScalarFeatureValues featureA = dataset.loadFeatureByKey("feature_a");
            assertValue(featureA.values.get(0), true, 10.0);
            assertValue(featureA.values.get(1), false, null);
            assertValue(featureA.values.get(2), false, null);

            ScalarFeatureValues featureB = dataset.loadFeatureByKey("feature_b");
            assertValue(featureB.values.get(0), false, null);
            assertValue(featureB.values.get(1), true, 21.0);
            assertValue(featureB.values.get(2), false, null);

            ScalarFeatureValues featureC = dataset.loadFeatureByKey("feature_c");
            assertValue(featureC.values.get(3), true, 33.0);
        }
    }

    private static void assertValue(ScalarValue actual, boolean present, Double value) {
        require(actual.present == present, "present mismatch for sample_id=" + actual.sampleId);
        if (!present) {
            require(actual.value == null, "missing scalar value should be null");
            return;
        }
        require(actual.value != null, "present scalar value should not be null");
        require(Math.abs(actual.value.doubleValue() - value.doubleValue()) <= 1e-12, "scalar value mismatch");
    }

    private static File rawSampleFile(File stageDir, long sampleId) {
        return new File(new File(stageDir, "raw_samples"), String.format("sample_%0" + RAW_SAMPLE_PADDING + "d.parquet", Long.valueOf(sampleId)));
    }

    private static File rawSampleLock(File stageDir, long sampleId) {
        return new File(rawSampleFile(stageDir, sampleId).getAbsolutePath() + ".lock");
    }

    private static File rawStageLock(File stageDir) {
        return new File(stageDir, "raw_stage.lock");
    }

    private static void writeFreshLock(File path, String kind) throws Exception {
        File parent = path.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("failed to create lock dir: " + parent.getAbsolutePath());
        }
        String json = "{\"kind\":\"" + kind + "\",\"pid\":\"test\",\"host\":\"local\",\"created_at_ms\":" + System.currentTimeMillis() + "}";
        Files.write(path.toPath(), Arrays.asList(json), StandardCharsets.UTF_8);
    }

    private static List<Long> allSampleIds() {
        return Arrays.asList(Long.valueOf(0L), Long.valueOf(1L), Long.valueOf(2L), Long.valueOf(3L));
    }

    private static void expectFailure(ThrowingRunnable runnable, String message) throws Exception {
        try {
            runnable.run();
        } catch (Exception expected) {
            return;
        }
        throw new AssertionError(message);
    }

    private static void runCase(ArrayList<String> failures, String name, ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Throwable error) {
            StringBuilder sb = new StringBuilder();
            sb.append(name).append(": ").append(error.toString());
            for (StackTraceElement element : error.getStackTrace()) {
                sb.append(System.lineSeparator()).append("  at ").append(element.toString());
            }
            failures.add(sb.toString());
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void deleteRecursively(File path) {
        if (path == null || !path.exists()) {
            return;
        }
        if (path.isDirectory()) {
            File[] children = path.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!path.delete()) {
            throw new IllegalStateException("failed to delete: " + path.getAbsolutePath());
        }
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private static final class Context {
        final String sampleMetaPath;
        final String featureMetaPath;
        final BuildShardConfig config;

        Context(String sampleMetaPath, String featureMetaPath, BuildShardConfig config) {
            this.sampleMetaPath = sampleMetaPath;
            this.featureMetaPath = featureMetaPath;
            this.config = config;
        }
    }
}
