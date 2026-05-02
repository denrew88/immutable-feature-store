package fs.io;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;

public class ArraySampleIdIndex {
    private final long nSamples;
    private final String[] sampleKeysById;
    private final HashMap<String, Long> sampleIdByKey;

    private ArraySampleIdIndex(long nSamples, String[] sampleKeysById, HashMap<String, Long> sampleIdByKey) {
        this.nSamples = nSamples;
        this.sampleKeysById = sampleKeysById;
        this.sampleIdByKey = sampleIdByKey;
    }

    public static ArraySampleIdIndex load(String sampleMetaPath) throws SQLException {
        return load(sampleMetaPath, "sample_key");
    }

    public static ArraySampleIdIndex load(String sampleMetaPath, String sampleKeyCol) throws SQLException {
        if (sampleMetaPath == null || sampleMetaPath.isEmpty()) {
            throw new IllegalArgumentException("sampleMetaPath must not be empty");
        }
        try (Connection conn = DuckDBUtils.connect(null)) {
            long count = countRows(conn, sampleMetaPath);
            validateDenseIds(conn, sampleMetaPath, "sample_id");
            if (count > Integer.MAX_VALUE) {
                throw new SQLException("sample count too large for in-memory key index: " + count);
            }
            String[] sampleKeysById = new String[(int) count];
            HashMap<String, Long> sampleIdByKey = new HashMap<String, Long>(Math.max(16, sampleKeysById.length * 2));
            loadKeys(conn, sampleMetaPath, sampleKeyCol, sampleKeysById, sampleIdByKey);
            return new ArraySampleIdIndex(count, sampleKeysById, sampleIdByKey);
        }
    }

    public Long findSampleId(long sampleId) {
        if (sampleId < 0L || sampleId >= nSamples) {
            return null;
        }
        return sampleId;
    }

    public Long findSampleIdByKey(String sampleKey) {
        if (sampleKey == null) {
            return null;
        }
        return sampleIdByKey.get(sampleKey);
    }

    public String sampleKey(long sampleId) {
        if (sampleId < 0L || sampleId >= sampleKeysById.length) {
            return null;
        }
        return sampleKeysById[(int) sampleId];
    }

    private static long countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getLong(1);
        }
    }

    private static void validateDenseIds(Connection conn, String path, String idCol) throws SQLException {
        String sql = "SELECT COUNT(*) FROM ("
                + "SELECT " + idCol + ", row_number() OVER () - 1 AS dense_id "
                + "FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")"
                + ") t WHERE " + idCol + " <> dense_id";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            long mismatches = rs.getLong(1);
            if (mismatches != 0L) {
                throw new SQLException("metadata ids are not dense row ids for " + path);
            }
        }
    }

    private static void loadKeys(
            Connection conn,
            String path,
            String sampleKeyCol,
            String[] sampleKeysById,
            HashMap<String, Long> sampleIdByKey) throws SQLException {
        String sql = "SELECT sample_id, " + sampleKeyCol
                + " FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") ORDER BY sample_id";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            int expectedId = 0;
            while (rs.next()) {
                int sampleId = rs.getInt(1);
                String sampleKey = rs.getString(2);
                if (sampleId != expectedId) {
                    throw new SQLException("sample_id sequence mismatch while loading keys for " + path);
                }
                if (sampleKey == null || sampleKey.isEmpty()) {
                    throw new SQLException("sample_key must be non-empty for sample_id=" + sampleId);
                }
                if (sampleIdByKey.put(sampleKey, (long) sampleId) != null) {
                    throw new SQLException("duplicate sample_key: " + sampleKey);
                }
                sampleKeysById[sampleId] = sampleKey;
                expectedId++;
            }
            if (expectedId != sampleKeysById.length) {
                throw new SQLException("sample key row count mismatch for " + path);
            }
        }
    }
}
