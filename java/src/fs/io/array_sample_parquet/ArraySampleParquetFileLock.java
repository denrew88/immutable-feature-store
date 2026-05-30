package fs.io.array_sample_parquet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * array_sample_parquet raw stage에서 sample 파일과 JSONL commit log를 보호하는 간단한 파일 락.
 *
 * <p>여러 Java 프로세스가 서로 다른 sample을 동시에 쓸 수 있으므로 JVM 내부 synchronized만으로는
 * 부족하다. 이 락은 {@code CREATE_NEW}에 해당하는 {@link File#createNewFile()}을 이용해 같은
 * sample 또는 같은 log append 구간에 동시에 들어오지 못하게 한다.</p>
 */
final class ArraySampleParquetFileLock implements AutoCloseable {
    private final File path;
    private final long timeoutMillis;
    private boolean acquired;

    ArraySampleParquetFileLock(String path) {
        this(path, 30000L);
    }

    ArraySampleParquetFileLock(String path, long timeoutMillis) {
        this.path = new File(path).getAbsoluteFile();
        this.timeoutMillis = timeoutMillis;
    }

    void acquire() throws IOException {
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

    void release() {
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
