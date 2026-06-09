package scripts;

import fs.io.common.FilePathLock;

import java.io.File;
import java.io.RandomAccessFile;
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/**
 * {@link FilePathLock}의 장애 방어 계약을 직접 검증하는 작은 실행형 테스트입니다.
 *
 * <p>일반 builder 테스트만으로는 owner token mismatch, stale lock 회수, 외부 프로세스가
 * lock 파일 handle을 잡는 상황을 안정적으로 만들기 어렵습니다. 이 스크립트는 그런
 * 경계 조건만 분리해서 확인합니다.</p>
 */
public final class RunFileLockContractTestsMain {
    private RunFileLockContractTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && "--hold-open".equals(args[0])) {
            holdOpenChild(args);
            return;
        }

        File root = new File("data/tmp_java_file_lock_contract_tests");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        ArrayList<String> failures = new ArrayList<String>();
        runCase(failures, "testReleaseDeletesOwnedLock", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testReleaseDeletesOwnedLock(new File(root, "owned.lock"));
            }
        });
        runCase(failures, "testReleaseRefusesDifferentOwnerToken", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testReleaseRefusesDifferentOwnerToken(new File(root, "mismatch.lock"));
            }
        });
        runCase(failures, "testAcquireCanBreakExplicitlyStaleDeadPidLock", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testAcquireCanBreakExplicitlyStaleDeadPidLock(new File(root, "stale.lock"));
            }
        });
        runCase(failures, "testReleaseRetriesWhileExternalProcessTemporarilyHoldsLock", new ThrowingRunnable() {
            @Override
            public void run() throws Exception {
                testReleaseRetriesWhileExternalProcessTemporarilyHoldsLock(
                        new File(root, "external_holder.lock"),
                        new File(root, "external_holder.ready"));
            }
        });

        if (!failures.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("file lock contract tests failed:");
            for (String failure : failures) {
                sb.append(System.lineSeparator()).append(System.lineSeparator()).append(failure);
            }
            throw new AssertionError(sb.toString());
        }
        System.out.println("file lock contract tests passed");
    }

    private static void testReleaseDeletesOwnedLock(File path) throws Exception {
        // 정상 acquire/release 경로입니다. lock 파일에 owner metadata가 생기고,
        // release 후에는 파일과 acquired 상태가 모두 정리되어야 합니다.
        FilePathLock lock = new FilePathLock(path.getAbsolutePath(), 500L);
        lock.acquire();
        assertTrue(path.isFile(), "lock file should exist after acquire");
        String text = read(path);
        assertTrue(text.contains("token="), "lock file should include token");
        assertTrue(text.contains("pid="), "lock file should include pid");
        lock.release();
        assertTrue(!path.exists(), "lock file should be deleted after release");
        assertTrue(!lock.isAcquired(), "lock object should no longer be acquired");
    }

    private static void testReleaseRefusesDifferentOwnerToken(File path) throws Exception {
        // release 직전에 lock 파일 token을 바꿔 다른 프로세스가 lock을 새로 잡은
        // 상황을 흉내 냅니다. 이때 release가 파일을 삭제하면 안 됩니다.
        FilePathLock lock = new FilePathLock(path.getAbsolutePath(), 500L);
        lock.acquire();
        String original = read(path);
        Files.write(path.toPath(), original.replaceFirst("token=[^\\r\\n]+", "token=someone-else").getBytes(StandardCharsets.UTF_8));
        boolean failed = false;
        try {
            lock.release();
        } catch (Exception e) {
            failed = true;
            assertTrue(e.getMessage().contains("another token"), "release failure should explain token mismatch");
        }
        assertTrue(failed, "release should fail for a different owner token");
        assertTrue(lock.isAcquired(), "failed release should keep acquired=true so caller can retry");
        Files.write(path.toPath(), original.getBytes(StandardCharsets.UTF_8));
        lock.release();
        assertTrue(!path.exists(), "restored owner token should allow release");
    }

    private static void testAcquireCanBreakExplicitlyStaleDeadPidLock(File path) throws Exception {
        // 운영자가 stale 회수를 명시적으로 켠 경우만 죽은 pid의 오래된 lock을 회수합니다.
        // pid는 실제로 존재하지 않는 큰 값으로 둬서 "죽은 프로세스"로 판단되게 합니다.
        String stale = ""
                + "token=stale-token" + System.lineSeparator()
                + "pid=999999999" + System.lineSeparator()
                + "thread=dead-worker" + System.lineSeparator()
                + "host=" + hostName() + System.lineSeparator()
                + "created_at_ms=1" + System.lineSeparator()
                + "path=" + path.getAbsolutePath() + System.lineSeparator();
        Files.write(path.toPath(), stale.getBytes(StandardCharsets.UTF_8));
        path.setLastModified(System.currentTimeMillis() - 10000L);

        FilePathLock lock = new FilePathLock(path.getAbsolutePath(), 5000L, 1L);
        lock.acquire();
        String current = read(path);
        assertTrue(current.contains("token="), "new lock file should include token");
        assertTrue(!current.contains("stale-token"), "stale token should have been replaced");
        lock.release();
        assertTrue(!path.exists(), "stale replacement lock should be releasable");
    }

    private static void testReleaseRetriesWhileExternalProcessTemporarilyHoldsLock(File path, File readyPath) throws Exception {
        // Windows에서 흔한 문제를 재현합니다. 다른 프로세스가 lock 파일을 read handle로
        // 잠깐 열어두면 delete가 실패할 수 있으므로 release retry가 실제로 기다리는지 봅니다.
        FilePathLock lock = new FilePathLock(path.getAbsolutePath(), 500L);
        lock.acquire();
        Process holder = startHolderProcess(path, readyPath, 350L);
        try {
            waitForReadyFile(readyPath, 3000L);
            long start = System.currentTimeMillis();
            lock.release();
            long elapsed = System.currentTimeMillis() - start;
            boolean exited = holder.waitFor(5L, TimeUnit.SECONDS);
            assertTrue(exited, "external holder process should exit");
            assertTrue(holder.exitValue() == 0, "external holder process failed with exit code " + holder.exitValue());
            assertTrue(!path.exists(), "lock file should be deleted after holder releases file handle");
            assertTrue(!lock.isAcquired(), "successful retry release should clear acquired state");
            if (isWindows()) {
                assertTrue(elapsed >= 200L, "release should have waited for external file handle; elapsed_ms=" + elapsed);
            }
        } finally {
            holder.destroy();
        }
    }

    private static Process startHolderProcess(File path, File readyPath, long holdMillis) throws Exception {
        String javaExe = new File(new File(System.getProperty("java.home"), "bin"), isWindows() ? "java.exe" : "java").getAbsolutePath();
        return new ProcessBuilder(
                javaExe,
                "-cp",
                System.getProperty("java.class.path"),
                RunFileLockContractTestsMain.class.getName(),
                "--hold-open",
                path.getAbsolutePath(),
                readyPath.getAbsolutePath(),
                String.valueOf(holdMillis))
                .redirectErrorStream(true)
                .start();
    }

    private static void holdOpenChild(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException("usage: --hold-open <path> <ready-path> <hold-millis>");
        }
        File path = new File(args[1]);
        File readyPath = new File(args[2]);
        long holdMillis = Long.parseLong(args[3]);
        RandomAccessFile handle = new RandomAccessFile(path, "r");
        try {
            Files.write(readyPath.toPath(), "ready".getBytes(StandardCharsets.UTF_8));
            Thread.sleep(holdMillis);
        } finally {
            handle.close();
        }
    }

    private static void waitForReadyFile(File readyPath, long timeoutMillis) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (System.currentTimeMillis() < deadline) {
            if (readyPath.isFile()) {
                return;
            }
            Thread.sleep(25L);
        }
        throw new AssertionError("external holder process did not become ready: " + readyPath.getAbsolutePath());
    }

    private static String read(File file) throws Exception {
        return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
    }

    private static String hostName() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            return "unknown";
        }
    }

    @SuppressWarnings("unused")
    private static String pid() {
        String name = ManagementFactory.getRuntimeMXBean().getName();
        int at = name.indexOf('@');
        return at > 0 ? name.substring(0, at) : name;
    }

    private static void runCase(ArrayList<String> failures, String name, ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Throwable t) {
            failures.add(name + ": " + stackTrace(t));
        }
    }

    private static String stackTrace(Throwable t) {
        java.io.StringWriter sw = new java.io.StringWriter();
        t.printStackTrace(new java.io.PrintWriter(sw));
        return sw.toString();
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("win");
    }

    private static void deleteRecursively(File file) {
        if (!file.exists()) {
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
        if (!file.delete() && file.exists()) {
            throw new IllegalStateException("failed to delete " + file.getAbsolutePath());
        }
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
