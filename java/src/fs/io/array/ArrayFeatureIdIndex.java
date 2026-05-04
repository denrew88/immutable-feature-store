package fs.io.array;

import fs.io.common.DuckDBUtils;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;

/**
 * Array feature id와 feature key 사이의 양방향 lookup 인덱스를 제공한다.
 */
public class ArrayFeatureIdIndex {
    private final int nFeatures;
    private final String[] featureKeysById;
    private final HashMap<String, Integer> featureIdByKey;

    private ArrayFeatureIdIndex(int nFeatures, String[] featureKeysById, HashMap<String, Integer> featureIdByKey) {
        this.nFeatures = nFeatures;
        this.featureKeysById = featureKeysById;
        this.featureIdByKey = featureIdByKey;
    }

    public static ArrayFeatureIdIndex load(String featureMetaPath) throws SQLException {
        return load(featureMetaPath, "feature_key");
    }

    public static ArrayFeatureIdIndex load(String featureMetaPath, String featureKeyCol) throws SQLException {
        if (featureMetaPath == null || featureMetaPath.isEmpty()) {
            throw new IllegalArgumentException("featureMetaPath must not be empty");
        }
        try (Connection conn = DuckDBUtils.connect(null)) {
            int count = countRows(conn, featureMetaPath);
            validateDenseIds(conn, featureMetaPath, "feature_id");
            String[] featureKeysById = new String[count];
            HashMap<String, Integer> featureIdByKey = new HashMap<String, Integer>(Math.max(16, count * 2));
            loadKeys(conn, featureMetaPath, featureKeyCol, featureKeysById, featureIdByKey);
            return new ArrayFeatureIdIndex(count, featureKeysById, featureIdByKey);
        }
    }

    public Integer findFeatureId(int featureId) {
        if (featureId < 0 || featureId >= nFeatures) {
            return null;
        }
        return featureId;
    }

    public Integer findFeatureIdByKey(String featureKey) {
        if (featureKey == null) {
            return null;
        }
        return featureIdByKey.get(featureKey);
    }

    public String featureKey(int featureId) {
        if (featureId < 0 || featureId >= featureKeysById.length) {
            return null;
        }
        return featureKeysById[featureId];
    }

    private static int countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
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
            String featureKeyCol,
            String[] featureKeysById,
            HashMap<String, Integer> featureIdByKey) throws SQLException {
        String sql = "SELECT feature_id, " + featureKeyCol
                + " FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") ORDER BY feature_id";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            int expectedId = 0;
            while (rs.next()) {
                int featureId = rs.getInt(1);
                String featureKey = rs.getString(2);
                if (featureId != expectedId) {
                    throw new SQLException("feature_id sequence mismatch while loading keys for " + path);
                }
                if (featureKey == null || featureKey.isEmpty()) {
                    throw new SQLException("feature_key must be non-empty for feature_id=" + featureId);
                }
                if (featureIdByKey.put(featureKey, featureId) != null) {
                    throw new SQLException("duplicate feature_key: " + featureKey);
                }
                featureKeysById[featureId] = featureKey;
                expectedId++;
            }
            if (expectedId != featureKeysById.length) {
                throw new SQLException("feature key row count mismatch for " + path);
            }
        }
    }
}
