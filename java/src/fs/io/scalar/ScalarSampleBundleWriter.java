package fs.io.scalar;

import fs.io.common.DuckDBUtils;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;

import java.io.File;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * sample 단위 scalar 값을 bundle parquet 여러 개로 spill하는 내부 writer다.
 *
 * <p>builder는 sample 하나를 완성할 때마다 feature/value row를 이 writer에 넘기고,
 * writer는 일정 row 수를 넘으면 sample bundle parquet 하나를 만든다.
 */
public final class ScalarSampleBundleWriter implements AutoCloseable {
    private final File bundleDir;
    private final DuckDBConnection conn;
    private final DuckDBAppender appender;
    private final List<String> bundlePaths;
    private final int maxBundleRows;
    private final boolean autoFlush;

    private int bundleIndex;
    private int currentRows;
    private boolean finished;

    /**
     * seal된 scalar sample bundle parquet 하나의 commit 결과다.
     */
    public static final class BundleCommit {
        public final int bundleId;
        public final String path;
        public final int rowCount;

        BundleCommit(int bundleId, String path, int rowCount) {
            this.bundleId = bundleId;
            this.path = path;
            this.rowCount = rowCount;
        }
    }

    /**
     * sample bundle writer를 초기화한다.
     *
     * @param outDir sample bundle 출력 디렉터리
     * @param maxBundleRows bundle 하나에 담을 최대 row 수
     */
    public ScalarSampleBundleWriter(String outDir, int maxBundleRows) throws Exception {
        this(outDir, maxBundleRows, 0, true);
    }

    /**
     * resume-safe builder가 bundle 번호와 auto flush를 직접 제어할 수 있게 하는 생성자다.
     */
    public ScalarSampleBundleWriter(String outDir, int maxBundleRows, int startBundleIndex, boolean autoFlush) throws Exception {
        File out = new File(outDir);
        if (!out.exists() && !out.mkdirs()) {
            throw new IllegalStateException("failed to create sample-bundle dir: " + out.getAbsolutePath());
        }
        this.bundleDir = out;
        this.maxBundleRows = maxBundleRows;
        this.autoFlush = autoFlush;
        this.bundlePaths = new ArrayList<String>();
        this.conn = (DuckDBConnection) DuckDBUtils.connect(null);
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_scalar_sample_bundle (sample_id BIGINT, feature_id INTEGER, value DOUBLE)");
        }
        this.appender = conn.createAppender(DuckDBConnection.DEFAULT_SCHEMA, "tmp_scalar_sample_bundle");
        this.bundleIndex = startBundleIndex;
        this.currentRows = 0;
        this.finished = false;
    }

    /**
     * sample 하나의 feature/value row를 현재 bundle 버퍼에 추가한다.
     */
    public void appendSample(long sampleId, Map<Integer, Double> featureValues) throws SQLException {
        ensureOpen();
        if (featureValues == null || featureValues.isEmpty()) {
            return;
        }
        for (Map.Entry<Integer, Double> entry : featureValues.entrySet()) {
            Double value = entry.getValue();
            if (value == null || Double.isNaN(value.doubleValue())) {
                continue;
            }
            appender.beginRow();
            appender.append(sampleId);
            appender.append(entry.getKey().intValue());
            appender.append(value.doubleValue());
            appender.endRow();
            currentRows++;
        }
        if (autoFlush && shouldFlushBundle()) {
            flushBundle();
        }
    }

    /**
     * 현재 bundle 버퍼가 flush 후보인지 알려준다.
     */
    public boolean shouldFlushBundle() {
        return currentRows >= maxBundleRows;
    }

    /**
     * 다음 bundle id를 돌려준다.
     */
    public int nextBundleId() {
        return bundleIndex;
    }

    /**
     * 남은 버퍼를 flush하고 bundle 파일 목록을 반환한다.
     */
    public List<String> finish() throws Exception {
        if (finished) {
            return new ArrayList<String>(bundlePaths);
        }
        flushBundle();
        finished = true;
        return new ArrayList<String>(bundlePaths);
    }

    @Override
    public void close() throws Exception {
        try {
            finish();
        } finally {
            try {
                appender.close();
            } finally {
                conn.close();
            }
        }
    }

    /**
     * 현재 임시 테이블을 parquet bundle 하나로 materialize한다.
     */
    public BundleCommit flushBundle() throws SQLException {
        if (currentRows == 0) {
            return null;
        }
        appender.flush();
        int bundleId = bundleIndex;
        String bundlePath = new File(bundleDir, String.format("bundle_%06d.parquet", bundleId)).getAbsolutePath();
        String tmpBundlePath = bundlePath + ".tmp";
        int rowCount = currentRows;
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_scalar_sample_bundle TO " + DuckDBUtils.quotePath(tmpBundlePath) + " (FORMAT PARQUET)");
            st.execute("DELETE FROM tmp_scalar_sample_bundle");
        }
        try {
            java.nio.file.Files.move(
                    new File(tmpBundlePath).toPath(),
                    new File(bundlePath).toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (java.io.IOException e) {
            throw new SQLException("failed to finalize bundle parquet: " + bundlePath, e);
        }
        bundlePaths.add(bundlePath);
        bundleIndex++;
        currentRows = 0;
        return new BundleCommit(bundleId, bundlePath, rowCount);
    }

    private void ensureOpen() {
        if (finished) {
            throw new IllegalStateException("scalar sample bundle writer is already finished");
        }
    }
}
