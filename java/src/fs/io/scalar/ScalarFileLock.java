package fs.io.scalar;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * scalar raw stage에서 sample 파일과 JSONL commit log를 보호하는 간단한 파일 락.
 *
 * <p>Java {@code FileLock}은 같은 JVM 안에서 같은 파일에 중복 lock을 잡으려 하면
 * {@code OverlappingFileLockException}이 날 수 있다. builder를 여러 thread가 같은
 * JVM에서 동시에 열어 sample을 쓰는 테스트까지 지원하려면 파일 생성의 원자성을 쓰는
 * lock 파일 방식이 더 단순하고 안전하다.</p>
 */
public final class ScalarFileLock implements AutoCloseable {
    private final File path;
    private final long timeoutMillis;
    private boolean acquired;

    public ScalarFileLock(String path) {
        this(path, 30000L);
    }

    public ScalarFileLock(String path, long timeoutMillis) {
        this.path = new File(path).getAbsoluteFile();
        this.timeoutMillis = timeoutMillis;
    }

    public void acquire() throws IOException {
        File parent = path.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create lock directory: " + parent.getAbsolutePath());
        }
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (true) {
            if (path.createNewFile()) {
                try (FileOutputStream out = new FileOutputStream(path, false)) {
                    out.write(Thread.currentThread().getName().getBytes(StandardCharsets.UTF_8));
                }
                acquired = true;
                return;
            }
            if (System.currentTimeMillis() >= deadline) {
                throw new IOException("timed out acquiring file lock: " + path.getAbsolutePath());
            }
            try {
                Thread.sleep(50L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted while acquiring file lock: " + path.getAbsolutePath(), e);
            }
        }
    }

    public void release() {
        if (!acquired) {
            return;
        }
        acquired = false;
        if (path.exists() && !path.delete()) {
            // best-effort cleanup; a stale lock will be surfaced by the next acquire timeout.
        }
    }

    @Override
    public void close() {
        release();
    }
}
