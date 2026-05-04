package fs.io.common;

import java.io.File;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;
import java.sql.Types;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * sample/feature metadata parquet를 dense row-id 규칙에 맞게 생성하는 유틸리티다.
 *
 * <p>입력은 일반적인 row map 목록이지만, 결과 파일은 항상
 * {@code 0..N-1} dense id를 row 순서와 맞춰 갖도록 정규화한다.
 */
public final class ArrayMetadataWriter {
    private ArrayMetadataWriter() {
    }

    /**
     * sample metadata parquet를 작성한다.
     *
     * @param records row map 목록
     * @param path 출력 parquet 경로
     * @return 절대 경로
     */
    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return writeDenseMetadata(records, path, "sample_id", "sample_key", true);
    }

    /**
     * feature metadata parquet를 작성한다.
     *
     * @param records row map 목록
     * @param path 출력 parquet 경로
     * @return 절대 경로
     */
    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return writeDenseMetadata(records, path, "feature_id", "feature_key", false);
    }

    /**
     * parquet 파일 전체를 행 단위 map 목록으로 읽는다.
     *
     * <p>builder가 metadata를 보강하거나 dense id를 검증할 때 다시 읽는 공용 helper다.
     *
     * @param path parquet 경로
     * @return 읽은 row 목록
     */
    public static List<LinkedHashMap<String, Object>> readRows(String path) throws Exception {
        ArrayList<LinkedHashMap<String, Object>> rows = new ArrayList<LinkedHashMap<String, Object>>();
        try (Connection conn = DuckDBUtils.connect(null)) {
            String sql = "SELECT * FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                ResultSetMetaData meta = rs.getMetaData();
                while (rs.next()) {
                    LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
                    for (int col = 1; col <= meta.getColumnCount(); col++) {
                        row.put(meta.getColumnLabel(col), rs.getObject(col));
                    }
                    rows.add(row);
                }
            }
        }
        return rows;
    }

    /**
     * dense id, unique key 제약을 검사한 뒤 parquet를 작성한다.
     *
     * <p>입력 row에 id 컬럼이 없으면 row 순서 기준으로 자동 생성하고, 있으면
     * row 순서와 정확히 일치하는지 검증한다. key 컬럼이 있으면 null과 중복을 막는다.
     */
    private static String writeDenseMetadata(
            List<Map<String, Object>> records,
            String path,
            String idCol,
            String keyCol,
            boolean useBigIntIds) throws Exception {
        if (records == null) {
            throw new IllegalArgumentException("metadata records must not be null");
        }
        ensureParentDir(path);

        ArrayList<String> columns = new ArrayList<String>();
        HashSet<String> seenColumns = new HashSet<String>();
        if (!seenColumns.add(idCol)) {
            throw new IllegalStateException("unexpected duplicate id column");
        }
        columns.add(idCol);
        for (Map<String, Object> row : records) {
            if (row == null) {
                throw new IllegalArgumentException("metadata rows must not be null");
            }
            for (String key : row.keySet()) {
                if (seenColumns.add(key)) {
                    columns.add(key);
                }
            }
        }

        ArrayList<LinkedHashMap<String, Object>> normalizedRows = new ArrayList<LinkedHashMap<String, Object>>(records.size());
        HashSet<String> seenKeys = new HashSet<String>();
        for (int rowIdx = 0; rowIdx < records.size(); rowIdx++) {
            Map<String, Object> source = records.get(rowIdx);
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
            for (String column : columns) {
                row.put(column, source.get(column));
            }
            Object idValue = row.get(idCol);
            long expectedId = rowIdx;
            if (idValue == null) {
                row.put(idCol, expectedId);
            } else if (!(idValue instanceof Number) || ((Number) idValue).longValue() != expectedId) {
                throw new IllegalArgumentException(idCol + " must equal dense row ids 0..N-1 in row order");
            }
            if (row.containsKey(keyCol)) {
                Object keyValue = row.get(keyCol);
                if (keyValue == null) {
                    throw new IllegalArgumentException(keyCol + " must not contain nulls");
                }
                String key = keyValue.toString();
                if (!seenKeys.add(key)) {
                    throw new IllegalArgumentException(keyCol + " must be unique: " + key);
                }
                row.put(keyCol, key);
            }
            normalizedRows.add(row);
        }

        LinkedHashMap<String, String> columnTypes = inferColumnTypes(columns, normalizedRows, idCol, useBigIntIds);
        writeRows(path, normalizedRows, columns, columnTypes);
        return new File(path).getAbsolutePath();
    }

    /**
     * 첫 non-null 값을 기준으로 각 컬럼의 DuckDB SQL 타입을 정한다.
     */
    private static LinkedHashMap<String, String> inferColumnTypes(
            List<String> columns,
            List<LinkedHashMap<String, Object>> rows,
            String idCol,
            boolean useBigIntIds) {
        LinkedHashMap<String, String> out = new LinkedHashMap<String, String>();
        for (String column : columns) {
            if (column.equals(idCol)) {
                out.put(column, useBigIntIds ? "BIGINT" : "INTEGER");
                continue;
            }
            String inferred = null;
            for (LinkedHashMap<String, Object> row : rows) {
                Object value = row.get(column);
                if (value == null) {
                    continue;
                }
                inferred = inferType(value);
                break;
            }
            out.put(column, (inferred == null) ? "VARCHAR" : inferred);
        }
        return out;
    }

    /**
     * 정규화된 row 목록을 DuckDB 임시 테이블을 거쳐 parquet로 내보낸다.
     */
    static void writeRows(
            String path,
            List<LinkedHashMap<String, Object>> rows,
            List<String> columns,
            LinkedHashMap<String, String> columnTypes) throws Exception {
        String tableName = "tmp_meta_" + Math.abs(path.hashCode());
        try (Connection conn = DuckDBUtils.connect(null)) {
            try (Statement st = conn.createStatement()) {
                StringBuilder create = new StringBuilder();
                create.append("CREATE TEMP TABLE ").append(tableName).append(" (");
                for (int i = 0; i < columns.size(); i++) {
                    if (i > 0) {
                        create.append(", ");
                    }
                    String column = columns.get(i);
                    create.append(DuckDBUtils.quoteIdentifier(column)).append(" ").append(columnTypes.get(column));
                }
                create.append(")");
                st.execute(create.toString());
            }
            StringBuilder insert = new StringBuilder();
            insert.append("INSERT INTO ").append(tableName).append(" VALUES (");
            for (int i = 0; i < columns.size(); i++) {
                if (i > 0) {
                    insert.append(", ");
                }
                insert.append("?");
            }
            insert.append(")");
            try (PreparedStatement ps = conn.prepareStatement(insert.toString())) {
                int batch = 0;
                for (LinkedHashMap<String, Object> row : rows) {
                    for (int i = 0; i < columns.size(); i++) {
                        String column = columns.get(i);
                        String sqlType = columnTypes.get(column);
                        setPreparedValue(ps, i + 1, row.get(column), sqlType);
                    }
                    ps.addBatch();
                    batch++;
                    if (batch >= 1024) {
                        ps.executeBatch();
                        batch = 0;
                    }
                }
                if (batch > 0) {
                    ps.executeBatch();
                }
            }
            try (Statement st = conn.createStatement()) {
                st.execute("COPY " + tableName + " TO " + DuckDBUtils.quotePath(path) + " (FORMAT PARQUET)");
                st.execute("DROP TABLE " + tableName);
            }
        }
    }

    private static void setPreparedValue(PreparedStatement ps, int index, Object value, String sqlType) throws Exception {
        if (value == null) {
            if ("DOUBLE".equals(sqlType)) {
                ps.setNull(index, Types.DOUBLE);
            } else if ("BOOLEAN".equals(sqlType)) {
                ps.setNull(index, Types.BOOLEAN);
            } else if ("BIGINT".equals(sqlType)) {
                ps.setNull(index, Types.BIGINT);
            } else if ("INTEGER".equals(sqlType)) {
                ps.setNull(index, Types.INTEGER);
            } else {
                ps.setNull(index, Types.VARCHAR);
            }
            return;
        }
        if ("DOUBLE".equals(sqlType)) {
            ps.setDouble(index, ((Number) value).doubleValue());
            return;
        }
        if ("BOOLEAN".equals(sqlType)) {
            ps.setBoolean(index, ((Boolean) value).booleanValue());
            return;
        }
        if ("BIGINT".equals(sqlType)) {
            ps.setLong(index, ((Number) value).longValue());
            return;
        }
        if ("INTEGER".equals(sqlType)) {
            ps.setInt(index, ((Number) value).intValue());
            return;
        }
        ps.setString(index, value.toString());
    }

    private static String inferType(Object value) {
        if (value instanceof Byte
                || value instanceof Short
                || value instanceof Integer) {
            return "INTEGER";
        }
        if (value instanceof Long) {
            return "BIGINT";
        }
        if (value instanceof Float || value instanceof Double) {
            return "DOUBLE";
        }
        if (value instanceof Boolean) {
            return "BOOLEAN";
        }
        return "VARCHAR";
    }

    private static void ensureParentDir(String path) {
        File parent = new File(path).getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("failed to create parent dir: " + parent.getAbsolutePath());
        }
    }
}
