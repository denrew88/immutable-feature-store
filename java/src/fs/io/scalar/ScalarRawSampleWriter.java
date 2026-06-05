package fs.io.scalar;

import fs.io.common.DuckDBUtils;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;

import java.io.File;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Map;
import java.util.TreeMap;

/**
 * scalar raw-sample parquet 파일 하나를 쓰는 내부 writer다.
 *
 * <p>raw stage에서는 sample 하나가 파일 하나로 commit된다. 이 writer는 sample 안의
 * present scalar 값만 {@code (sample_id, feature_id, value)} row로 기록하고,
 * missing 값은 row를 쓰지 않는다. 최종 dense-long shard를 만들 때 missing row는
 * scalar Double value column 기준 {@code mask=0, value=NaN}으로 다시 materialize된다.
 */
public final class ScalarRawSampleWriter {
    private ScalarRawSampleWriter() {
    }

    public static int write(String tmpPath, long sampleId, Map<Integer, Double> featureValues) throws Exception {
        File out = new File(tmpPath).getAbsoluteFile();
        File parent = out.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new SQLException("failed to create raw sample dir: " + parent.getAbsolutePath());
        }
        if (out.exists() && !out.delete()) {
            throw new SQLException("failed to remove stale tmp raw sample: " + out.getAbsolutePath());
        }

        TreeMap<Integer, Double> sorted = new TreeMap<Integer, Double>();
        if (featureValues != null) {
            for (Map.Entry<Integer, Double> entry : featureValues.entrySet()) {
                Double value = entry.getValue();
                if (value == null || Double.isNaN(value.doubleValue())) {
                    continue;
                }
                sorted.put(entry.getKey(), value);
            }
        }

        try (Connection rawConn = DuckDBUtils.connect(null);
             Statement st = rawConn.createStatement()) {
            DuckDBConnection conn = (DuckDBConnection) rawConn;
            st.execute("CREATE TEMP TABLE tmp_scalar_raw_sample (sample_id BIGINT, feature_id INTEGER, value DOUBLE)");
            try (DuckDBAppender appender = conn.createAppender(DuckDBConnection.DEFAULT_SCHEMA, "tmp_scalar_raw_sample")) {
                for (Map.Entry<Integer, Double> entry : sorted.entrySet()) {
                    appender.beginRow();
                    appender.append(sampleId);
                    appender.append(entry.getKey().intValue());
                    appender.append(entry.getValue().doubleValue());
                    appender.endRow();
                }
                appender.flush();
            }
            st.execute("COPY tmp_scalar_raw_sample TO " + DuckDBUtils.quotePath(out.getAbsolutePath()) + " (FORMAT PARQUET, COMPRESSION ZSTD)");
        }
        return sorted.size();
    }
}
