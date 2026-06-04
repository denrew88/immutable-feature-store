package scripts;

import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.ScalarDenseLongDataset;
import fs.model.scalar.ScalarBuildSessionStatus;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * scalar raw stage에 여러 worker가 동시에 sample을 commit해도 되는지 검증한다.
 *
 * <p>각 worker는 같은 {@code outDir}을 독립적으로 {@link ScalarDatasetBuilder#openSession}
 * 한 뒤 자신에게 배정된 sample만 쓴다. 이 패턴은 실제 supervisor/worker 구조에서 가장
 * 중요한 병렬 사용 형태이며, sample 파일 lock, commit log lock, state JSON write가 서로
 * 충돌하지 않는지 확인한다.</p>
 */
public final class RunScalarConcurrentBuilderTestsMain {
    private RunScalarConcurrentBuilderTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        int nSamples = intArg(args, "--n-samples", 24);
        int nFeatures = intArg(args, "--n-features", 128);
        int nWorkers = intArg(args, "--n-workers", 6);
        boolean skipBuild = hasFlag(args, "--skip-build");
        File root = new File(stringArg(args, "--out-dir", "data/tmp_java_scalar_concurrent_builder_test"));
        long started = System.nanoTime();
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        ScalarFeatureShards.writeSampleMeta(sampleRows(nSamples), sampleMetaPath);
        ScalarFeatureShards.writeFeatureMeta(featureRows(nFeatures), featureMetaPath);

        final String outDir = new File(root, "stage").getAbsolutePath();
        final String finalSampleMetaPath = sampleMetaPath;
        final String finalFeatureMetaPath = featureMetaPath;
        final int finalNFeatures = nFeatures;
        final BuildShardConfig cfg = new BuildShardConfig();
        cfg.targetShardBytes = 1L << 20;

        // 병렬 worker가 동시에 처음 초기화하지 않도록 supervisor가 먼저 stage를 만든다.
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            require(builder.status().pendingSampleIds.equals(allSampleIds(nSamples)), "initial pending sample ids mismatch");
        }
        double initSec = elapsedSec(started);

        long writeStarted = System.nanoTime();
        ExecutorService executor = Executors.newFixedThreadPool(nWorkers);
        ArrayList<Future<List<Long>>> futures = new ArrayList<Future<List<Long>>>();
        for (int workerId = 0; workerId < nWorkers; workerId++) {
            final List<Long> assigned = assignedSamples(workerId, nSamples, nWorkers);
            futures.add(executor.submit(new Callable<List<Long>>() {
                @Override
                public List<Long> call() throws Exception {
                    try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                            outDir,
                            finalSampleMetaPath,
                            finalFeatureMetaPath,
                            null,
                            cfg)) {
                        for (Long sampleId : assigned) {
                            builder.writeSample(sampleId.longValue(), valuesFor(sampleId.longValue(), finalNFeatures), true);
                        }
                    }
                    return assigned;
                }
            }));
        }
        executor.shutdown();

        ArrayList<Long> committed = new ArrayList<Long>();
        for (Future<List<Long>> future : futures) {
            committed.addAll(future.get());
        }
        java.util.Collections.sort(committed);
        require(committed.equals(allSampleIds(nSamples)), "committed sample ids mismatch");
        double writeSec = elapsedSec(writeStarted);

        long finishStarted = System.nanoTime();
        String manifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            ScalarBuildSessionStatus status = builder.status();
            require(status.completedSampleCount == nSamples, "completed sample count mismatch");
            require(status.pendingSampleIds.isEmpty(), "pending samples should be empty");
            if (skipBuild) {
                manifestPath = builder.finishStage();
            } else {
                manifestPath = builder.buildDenseLongShards(true, new File(root, "dense_long").getAbsolutePath(), false);
            }
        }
        double finishSec = elapsedSec(finishStarted);

        if (!skipBuild) {
            try (ScalarDenseLongDataset ds = ScalarFeatureShards.openDenseLong(manifestPath)) {
                long sampleId = Math.min(7L, nSamples - 1L);
                int featureId = Math.min(11, nFeatures - 1);
                ScalarDenseLongDataset.SampleValues sample = ds.loadSampleById(sampleId);
                require(sample.valid[featureId] != 0, "sample feature should be present");
                require(Math.abs(sample.values[featureId] - (sampleId * 1000.0 + featureId)) < 1e-9, "sample feature value mismatch");
            }
        }

        double totalSec = elapsedSec(started);
        System.out.println(
                "java scalar concurrent builder tests passed"
                        + " n_samples=" + nSamples
                        + " n_features=" + nFeatures
                        + " n_workers=" + nWorkers
                        + " skip_build=" + skipBuild
                        + " init_sec=" + formatSeconds(initSec)
                        + " write_sec=" + formatSeconds(writeSec)
                        + " finish_sec=" + formatSeconds(finishSec)
                        + " total_sec=" + formatSeconds(totalSec));
    }

    private static List<Map<String, Object>> sampleRows(int nSamples) {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        for (int sampleId = 0; sampleId < nSamples; sampleId++) {
            rows.add(row("sample_key", String.format("sample_%06d", sampleId), "y", Double.valueOf(sampleId % 3)));
        }
        return rows;
    }

    private static List<Map<String, Object>> featureRows(int nFeatures) {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        for (int featureId = 0; featureId < nFeatures; featureId++) {
            rows.add(row("feature_key", String.format("feature_%06d", featureId)));
        }
        return rows;
    }

    private static List<Long> allSampleIds(int nSamples) {
        ArrayList<Long> ids = new ArrayList<Long>();
        for (int sampleId = 0; sampleId < nSamples; sampleId++) {
            ids.add(Long.valueOf(sampleId));
        }
        return ids;
    }

    private static List<Long> assignedSamples(int workerId, int nSamples, int nWorkers) {
        ArrayList<Long> ids = new ArrayList<Long>();
        for (int sampleId = workerId; sampleId < nSamples; sampleId += nWorkers) {
            ids.add(Long.valueOf(sampleId));
        }
        return ids;
    }

    private static Map<Integer, Double> valuesFor(long sampleId, int nFeatures) {
        LinkedHashMap<Integer, Double> values = new LinkedHashMap<Integer, Double>();
        for (int featureId = 0; featureId < nFeatures; featureId++) {
            if ((sampleId + featureId) % 5 != 0) {
                values.put(Integer.valueOf(featureId), Double.valueOf(sampleId * 1000.0 + featureId));
            }
        }
        return values;
    }

    private static Map<String, Object> row(Object... kv) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kv.length; i += 2) {
            out.put((String) kv[i], kv[i + 1]);
        }
        return out;
    }

    private static int intArg(String[] args, String key, int defaultValue) {
        String value = stringArg(args, key, null);
        return value == null ? defaultValue : Integer.parseInt(value);
    }

    private static String stringArg(String[] args, String key, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }

    private static boolean hasFlag(String[] args, String key) {
        for (String arg : args) {
            if (arg.equals(key)) {
                return true;
            }
        }
        return false;
    }

    private static double elapsedSec(long startedNanos) {
        return (System.nanoTime() - startedNanos) / 1_000_000_000.0;
    }

    private static String formatSeconds(double value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
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
}
