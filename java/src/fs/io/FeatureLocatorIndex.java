package fs.io;

import fs.model.FeatureLocation;
import fs.model.ShardManifest;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;

public class FeatureLocatorIndex implements AutoCloseable {
    private final Connection conn;
    private final HashMap<Integer, Integer> featureToRow;
    private final int[] featureIds;
    private final int[] globalRanks;
    private final int[] shardIds;
    private final int[] offsets;
    private final double[] r2y;
    private final int[] nYOverlap;

    private FeatureLocatorIndex(
            Connection conn,
            HashMap<Integer, Integer> featureToRow,
            int[] featureIds,
            int[] globalRanks,
            int[] shardIds,
            int[] offsets,
            double[] r2y,
            int[] nYOverlap) {
        this.conn = conn;
        this.featureToRow = featureToRow;
        this.featureIds = featureIds;
        this.globalRanks = globalRanks;
        this.shardIds = shardIds;
        this.offsets = offsets;
        this.r2y = r2y;
        this.nYOverlap = nYOverlap;
    }

    public static FeatureLocatorIndex load(ShardManifest manifest) throws SQLException {
        Connection conn = DuckDBUtils.connect(null);
        String locatorPath = manifest.featureLocatorPath;
        int rows = countRows(conn, locatorPath);
        HashMap<Integer, Integer> featureToRow = new HashMap<Integer, Integer>(Math.max(16, rows * 2));
        int[] featureIds = new int[rows];
        int[] globalRanks = new int[rows];
        int[] shardIds = new int[rows];
        int[] offsets = new int[rows];
        double[] r2y = new double[rows];
        int[] nYOverlap = new int[rows];
        boolean hasR2y = hasColumn(conn, locatorPath, "r2y");
        boolean hasNYOverlap = hasColumn(conn, locatorPath, "n_y_overlap");
        for (int j = 0; j < rows; j++) {
            r2y[j] = Double.NaN;
            nYOverlap[j] = -1;
        }

        StringBuilder sql = new StringBuilder();
        sql.append("SELECT feature_id, global_rank, shard_id, offset_in_shard");
        if (hasR2y) {
            sql.append(", r2y");
        }
        if (hasNYOverlap) {
            sql.append(", n_y_overlap");
        }
        sql.append(" FROM read_parquet(").append(DuckDBUtils.quotePath(locatorPath)).append(")");
        int i = 0;
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql.toString())) {
            while (rs.next()) {
                int fid = rs.getInt(1);
                int globalRank = rs.getInt(2);
                int shardId = rs.getInt(3);
                int offset = rs.getInt(4);
                featureIds[i] = fid;
                globalRanks[i] = globalRank;
                shardIds[i] = shardId;
                offsets[i] = offset;
                int col = 5;
                if (hasR2y) {
                    r2y[i] = rs.getDouble(col);
                    col++;
                }
                if (hasNYOverlap) {
                    nYOverlap[i] = rs.getInt(col);
                }
                featureToRow.put(fid, i);
                i++;
            }
        }
        if (i != rows) {
            throw new SQLException("feature locator row count mismatch: expected " + rows + " got " + i);
        }
        return new FeatureLocatorIndex(conn, featureToRow, featureIds, globalRanks, shardIds, offsets, r2y, nYOverlap);
    }

    public FeatureLocation find(int featureId) {
        Integer rowObj = featureToRow.get(featureId);
        if (rowObj == null) {
            return null;
        }
        int row = rowObj.intValue();
        return new FeatureLocation(featureIds[row], globalRanks[row], shardIds[row], offsets[row], r2y[row], nYOverlap[row]);
    }

    @Override
    public void close() throws SQLException {
        conn.close();
    }

    private static int countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
    }

    private static boolean hasColumn(Connection conn, String path, String colName) throws SQLException {
        String sql = "SELECT * FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") LIMIT 0";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            ResultSetMetaData meta = rs.getMetaData();
            for (int i = 1; i <= meta.getColumnCount(); i++) {
                if (colName.equalsIgnoreCase(meta.getColumnLabel(i))) {
                    return true;
                }
            }
            return false;
        }
    }
}
