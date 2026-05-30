package fs.io.array_sample_parquet;

import fs.io.common.DuckDBUtils;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;

import java.io.File;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * array_sample_parquet Java 구현에서 DuckDB 설정과 COPY 옵션을 한곳에 모은 helper.
 */
final class ArraySampleParquetDuckDB {
    private ArraySampleParquetDuckDB() {
    }

    static void configure(Connection conn, File outDir, ArraySampleParquetBuildOptions options) throws SQLException {
        File tmpDir = new File(outDir, ".duckdb_tmp").getAbsoluteFile();
        if (!tmpDir.exists() && !tmpDir.mkdirs()) {
            throw new SQLException("failed to create DuckDB temp directory: " + tmpDir.getAbsolutePath());
        }
        try (Statement st = conn.createStatement()) {
            st.execute("SET temp_directory=" + DuckDBUtils.quotePath(tmpDir.getAbsolutePath()));
            if (options != null && options.duckdbThreads > 0) {
                st.execute("SET threads=" + options.duckdbThreads);
            }
        }
    }

    static String parquetCopyOptions(String compression) {
        String value = compression == null ? "" : compression.trim().toLowerCase();
        if (value.isEmpty() || "none".equals(value) || "uncompressed".equals(value)) {
            return "(FORMAT PARQUET, COMPRESSION 'uncompressed')";
        }
        return "(FORMAT PARQUET, COMPRESSION " + DuckDBUtils.quotePath(value) + ")";
    }

    static String pathListLiteral(java.util.List<String> paths) {
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
}
