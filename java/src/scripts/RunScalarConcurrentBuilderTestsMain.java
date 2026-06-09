package scripts;

import fs.config.BuildShardConfig;
import fs.io.common.DuckDBUtils;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.model.scalar.ScalarBuildSessionStatus;
import fs.model.scalar.ScalarDenseLongManifest;
import fs.model.scalar.ScalarDenseLongPart;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * scalar raw stage에 여러 worker가 동시에 sample을 commit해도 되는지 검증합니다.
 *
 * <p>각 worker는 같은 {@code outDir}을 독립적으로 {@link ScalarDatasetBuilder#openSession}
 * 하고, 자신에게 배정된 sample만 씁니다. 이 패턴은 실제 supervisor/worker 구조에서 가장
 * 중요한 병렬 사용 형태입니다. 따라서 sample 파일 lock, commit log lock, state JSON write가
 * 서로 충돌하지 않는지 이 스크립트에서 함께 확인합니다.</p>
 *
 * <p>{@code --lock-interferer}를 주면 별도 JVM이 백신/인덱서처럼 {@code .lock} 파일을 짧게
 * 열었다 닫습니다. Windows에서는 외부 프로세스가 파일 handle을 들고 있으면 삭제가 실패할 수
 * 있으므로, release retry가 실제 방해 상황에서도 동작하는지 확인하기 위한 옵션입니다.</p>
 */
public final class RunScalarConcurrentBuilderTestsMain {
    private RunScalarConcurrentBuilderTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && "--interfere-locks".equals(args[0])) {
            runLockInterferer(args);
            return;
        }

        int nSamples = intArg(args, "--n-samples", 24);
        int nFeatures = intArg(args, "--n-features", 128);
        int nWorkers = intArg(args, "--n-workers", 6);
        boolean skipBuild = hasFlag(args, "--skip-build");
        boolean lockInterferer = hasFlag(args, "--lock-interferer");
        long lockHoldMillis = longArg(args, "--lock-hold-millis", 30L);
        long lockScanSleepMillis = longArg(args, "--lock-scan-sleep-millis", 10L);
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

        // 병렬 worker가 동시에 최초 초기화를 수행하지 않도록 supervisor가 stage를 먼저 만듭니다.
        // 이후 worker들은 이미 존재하는 stage를 열어 자신에게 배정된 sample만 추가합니다.
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            require(builder.status().pendingSampleIds.equals(allSampleIds(nSamples)), "initial pending sample ids mismatch");
        }
        double initSec = elapsedSec(started);

        File lockInterfererStopPath = new File(root, "lock_interferer.stop");
        File lockInterfererReadyPath = new File(root, "lock_interferer.ready");
        File lockInterfererCountPath = new File(root, "lock_interferer_count.txt");
        Process lockInterfererProcess = null;
        if (lockInterferer) {
            lockInterfererProcess = startLockInterferer(
                    new File(outDir),
                    lockInterfererStopPath,
                    lockInterfererReadyPath,
                    lockInterfererCountPath,
                    lockHoldMillis,
                    lockScanSleepMillis);
            waitForFile(lockInterfererReadyPath, 5000L, "lock interferer did not become ready");
        }

        double writeSec;
        double finishSec;
        long lockInterfererOpenCount = 0L;
        try {
            long writeStarted = System.nanoTime();
            ExecutorService executor = Executors.newFixedThreadPool(nWorkers);
            ArrayList<Future<List<Long>>> futures = new ArrayList<Future<List<Long>>>();
            for (int workerId = 0; workerId < nWorkers; workerId++) {
                final List<Long> assigned = assignedSamples(workerId, nSamples, nWorkers);
                futures.add(executor.submit(new Callable<List<Long>>() {
                    @Override
                    public List<Long> call() throws Exception {
                        // 각 worker는 builder 객체를 공유하지 않습니다. 공유 대상은 파일 시스템의
                        // stage 디렉터리뿐이므로, 동시성 제어는 FilePathLock과 commit log lock이 담당합니다.
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
            writeSec = elapsedSec(writeStarted);

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
                    manifestPath = builder.buildDenseLongShards(true, new File(root, "scalar_shard").getAbsolutePath(), false);
                }
            }
            finishSec = elapsedSec(finishStarted);

            if (!skipBuild) {
                assertFinalShardMatchesRaw(manifestPath, new File(outDir, "raw_samples"), nSamples, nFeatures);
            }
        } finally {
            if (lockInterfererProcess != null) {
                stopLockInterferer(lockInterfererProcess, lockInterfererStopPath);
            }
        }
        if (lockInterferer) {
            lockInterfererOpenCount = readLongFile(lockInterfererCountPath);
            require(lockInterfererOpenCount > 0L, "lock interferer did not open any lock file");
        }

        double totalSec = elapsedSec(started);
        System.out.println(
                "java scalar concurrent builder tests passed"
                        + " n_samples=" + nSamples
                        + " n_features=" + nFeatures
                        + " n_workers=" + nWorkers
                        + " skip_build=" + skipBuild
                        + " lock_interferer=" + lockInterferer
                        + " lock_interferer_open_count=" + lockInterfererOpenCount
                        + " init_sec=" + formatSeconds(initSec)
                        + " write_sec=" + formatSeconds(writeSec)
                        + " finish_sec=" + formatSeconds(finishSec)
                        + " total_sec=" + formatSeconds(totalSec));
    }

    private static Process startLockInterferer(
            File scanRoot,
            File stopPath,
            File readyPath,
            File countPath,
            long holdMillis,
            long scanSleepMillis) throws Exception {
        // 테스트 본문과 같은 classpath로 자식 JVM을 띄웁니다. 같은 프로세스 안의 thread가 아니라
        // 별도 프로세스를 쓰는 이유는 Windows 파일 handle 충돌이 process boundary에서 더 현실적으로
        // 재현되기 때문입니다.
        String javaExe = new File(new File(System.getProperty("java.home"), "bin"), isWindows() ? "java.exe" : "java").getAbsolutePath();
        return new ProcessBuilder(
                javaExe,
                "-cp",
                System.getProperty("java.class.path"),
                RunScalarConcurrentBuilderTestsMain.class.getName(),
                "--interfere-locks",
                scanRoot.getAbsolutePath(),
                stopPath.getAbsolutePath(),
                readyPath.getAbsolutePath(),
                countPath.getAbsolutePath(),
                String.valueOf(holdMillis),
                String.valueOf(scanSleepMillis))
                .redirectErrorStream(true)
                .start();
    }

    private static void stopLockInterferer(Process process, File stopPath) throws Exception {
        // 자식 JVM은 stop 파일을 발견하면 정상 종료합니다. 강제 종료는 테스트 실패로 처리해서
        // lock 방해 프로세스가 뒤에 남아 다음 테스트에 영향을 주지 않게 합니다.
        Files.write(stopPath.toPath(), "stop".getBytes(StandardCharsets.UTF_8));
        if (!process.waitFor(10L, java.util.concurrent.TimeUnit.SECONDS)) {
            process.destroy();
            throw new IllegalStateException("lock interferer did not stop");
        }
        if (process.exitValue() != 0) {
            throw new IllegalStateException("lock interferer failed with exit code " + process.exitValue());
        }
    }

    private static void runLockInterferer(String[] args) throws Exception {
        if (args.length != 7) {
            throw new IllegalArgumentException("usage: --interfere-locks <scan-root> <stop-path> <ready-path> <count-path> <hold-ms> <scan-sleep-ms>");
        }
        File scanRoot = new File(args[1]);
        File stopPath = new File(args[2]);
        File readyPath = new File(args[3]);
        File countPath = new File(args[4]);
        long holdMillis = Long.parseLong(args[5]);
        long scanSleepMillis = Long.parseLong(args[6]);
        long opened = 0L;
        Files.write(readyPath.toPath(), "ready".getBytes(StandardCharsets.UTF_8));
        while (!stopPath.isFile()) {
            ArrayList<File> locks = new ArrayList<File>();
            collectLockFiles(scanRoot, locks);
            for (File lock : locks) {
                if (stopPath.isFile()) {
                    break;
                }
                try (RandomAccessFile ignored = new RandomAccessFile(lock, "r")) {
                    // lock 파일을 읽기 handle로 잠깐 잡아 백신/인덱서가 파일을 스캔하는 상황을 흉내냅니다.
                    // builder의 release/delete retry가 충분하면 이 방해가 있어도 최종 stage는 정상 완료됩니다.
                    opened++;
                    Thread.sleep(holdMillis);
                } catch (Exception ignored) {
                    // 스캔 직후 builder가 lock을 지울 수 있으므로, 실패한 open은 정상 race로 보고 계속 진행합니다.
                }
            }
            Thread.sleep(scanSleepMillis);
        }
        Files.write(countPath.toPath(), String.valueOf(opened).getBytes(StandardCharsets.UTF_8));
    }

    private static void collectLockFiles(File root, List<File> out) {
        // builder가 lock 파일을 만들고 지우는 중에 스캔하므로, 경로가 사라지거나 listFiles가 null을
        // 반환하는 race는 정상입니다. 테스트 방해 프로세스는 가능한 lock만 열고 나머지는 무시합니다.
        if (root == null || !root.exists()) {
            return;
        }
        if (root.isFile()) {
            if (root.getName().endsWith(".lock")) {
                out.add(root);
            }
            return;
        }
        File[] children = root.listFiles();
        if (children == null) {
            return;
        }
        for (File child : children) {
            collectLockFiles(child, out);
        }
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
        // 공식 기반 값이 아니라 sample별 seed를 둔 random sparse 값을 씁니다. 그래야 빌더가 우연히
        // 같은 공식을 재현해서 통과하는 문제가 없고, raw parquet와 final shard를 실제 값으로 비교합니다.
        LinkedHashMap<Integer, Double> values = new LinkedHashMap<Integer, Double>();
        Random rng = new Random(0x5EED0000L + sampleId);
        for (int featureId = 0; featureId < nFeatures; featureId++) {
            if (rng.nextDouble() >= 0.23) {
                double value = rng.nextGaussian() * 10.0 + (rng.nextDouble() * 6.0 - 3.0);
                values.put(Integer.valueOf(featureId), Double.valueOf(value));
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

    private static void assertFinalShardMatchesRaw(
            String manifestPath,
            File rawSamplesDir,
            int nSamples,
            int nFeatures) throws Exception {
        // 최종 dense-long shard는 모든 (feature_id, sample_id) 조합을 row로 가집니다.
        // raw sample parquet에는 실제 값이 있는 sparse row만 있으므로, DuckDB에서 dense grid를 만든 뒤
        // raw를 LEFT JOIN하여 최종 shard가 가져야 할 mask/value를 직접 구성합니다.
        ScalarDenseLongManifest manifest = ScalarFeatureShards.loadManifest(manifestPath);
        ArrayList<String> partPaths = new ArrayList<String>();
        for (ScalarDenseLongPart part : manifest.parts) {
            partPaths.add(part.path);
        }
        ArrayList<String> rawPaths = new ArrayList<String>();
        File[] rawFiles = rawSamplesDir.listFiles();
        if (rawFiles != null) {
            for (File file : rawFiles) {
                if (file.getName().endsWith(".parquet")) {
                    rawPaths.add(file.getAbsolutePath());
                }
            }
        }
        require(!rawPaths.isEmpty(), "raw sample parquet files are missing");

        // finalQuery는 빌더가 만든 결과입니다. 비교 안정성을 위해 parquet column type을 명시적으로 cast합니다.
        String finalQuery = "SELECT CAST(feature_id AS INTEGER) AS feature_id, "
                + "CAST(sample_id AS BIGINT) AS sample_id, "
                + "CAST(mask AS UTINYINT) AS mask, "
                + "CAST(value AS DOUBLE) AS value "
                + "FROM read_parquet(" + parquetList(partPaths) + ")";
        // expectedQuery는 raw sample parquet만 기준으로 만든 기대 결과입니다. raw에 값이 없거나 NaN이면
        // missing으로 보고 mask=0/value=NaN이어야 합니다. 이 기준을 final과 EXCEPT로 전수 비교합니다.
        String expectedQuery =
                "WITH features AS (SELECT CAST(range AS INTEGER) AS feature_id FROM range(0, " + nFeatures + ")), "
                        + "samples AS (SELECT CAST(range AS BIGINT) AS sample_id FROM range(0, " + nSamples + ")), "
                        + "dense AS (SELECT f.feature_id, s.sample_id FROM features f CROSS JOIN samples s), "
                        + "raw AS (SELECT CAST(feature_id AS INTEGER) AS feature_id, "
                        + "CAST(sample_id AS BIGINT) AS sample_id, CAST(value AS DOUBLE) AS value "
                        + "FROM read_parquet(" + parquetList(rawPaths) + ") "
                        + "WHERE value IS NOT NULL AND NOT isnan(CAST(value AS DOUBLE))) "
                        + "SELECT d.feature_id, d.sample_id, "
                        + "CAST(CASE WHEN raw.value IS NULL THEN 0 ELSE 1 END AS UTINYINT) AS mask, "
                        + "CASE WHEN raw.value IS NULL THEN CAST('NaN' AS DOUBLE) ELSE CAST(raw.value AS DOUBLE) END AS value "
                        + "FROM dense d LEFT JOIN raw USING(feature_id, sample_id)";
        String sql = "WITH final AS (" + finalQuery + "), expected AS (" + expectedQuery + ") "
                + "SELECT "
                + "(SELECT COUNT(*) FROM (SELECT * FROM final EXCEPT SELECT * FROM expected)) AS final_extra, "
                + "(SELECT COUNT(*) FROM (SELECT * FROM expected EXCEPT SELECT * FROM final)) AS final_missing";
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            require(rs.next(), "direct raw/final comparison did not return a row");
            long extra = rs.getLong(1);
            long missing = rs.getLong(2);
            require(extra == 0L && missing == 0L, "final shard differs from raw samples: extra=" + extra + " missing=" + missing);
        }
    }

    private static String parquetList(List<String> paths) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < paths.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(DuckDBUtils.quotePath(paths.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    private static int intArg(String[] args, String key, int defaultValue) {
        String value = stringArg(args, key, null);
        return value == null ? defaultValue : Integer.parseInt(value);
    }

    private static long longArg(String[] args, String key, long defaultValue) {
        String value = stringArg(args, key, null);
        return value == null ? defaultValue : Long.parseLong(value);
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

    private static void waitForFile(File file, long timeoutMillis, String message) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (System.currentTimeMillis() < deadline) {
            if (file.isFile()) {
                return;
            }
            Thread.sleep(25L);
        }
        throw new IllegalStateException(message + ": " + file.getAbsolutePath());
    }

    private static long readLongFile(File file) throws Exception {
        waitForFile(file, 5000L, "expected count file is missing");
        String text = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8).trim();
        return text.isEmpty() ? 0L : Long.parseLong(text);
    }

    private static boolean isWindows() {
        try {
            // "darwin" 같은 문자열에 걸리지 않도록 startsWith("windows")만 Windows로 봅니다.
            return System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).startsWith("windows");
        } catch (SecurityException exc) {
            return false;
        }
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
