package fs.io.scalar;

import fs.config.BuildShardConfig;
import fs.io.common.DuckDBUtils;
import fs.model.common.SampleMeta;

import java.io.File;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Sample metadata parquet에서 selection에 필요한 y 컬럼과 기본 메타를 읽는 helper다.
 */
public class SampleMetaLoader {
    public static SampleMeta load(String sampleMetaPath, BuildShardConfig config, boolean orderBySampleId) throws SQLException {
        long[] sampleIds;
        double[] y;
        byte[] yMask;
        try (Connection conn = DuckDBUtils.connect(null)) {
            int count = countRows(conn, sampleMetaPath);
            sampleIds = new long[count];
            y = new double[count];
            yMask = new byte[count];

            StringBuilder sql = new StringBuilder();
            sql.append("SELECT ").append(config.sampleIdCol).append(", ").append(config.yCol);
            if (hasColumn(conn, sampleMetaPath, config.pathCol)) {
                sql.append(", ").append(config.pathCol);
            }
            sql.append(" FROM read_parquet(").append(DuckDBUtils.quotePath(sampleMetaPath)).append(")");
            if (orderBySampleId) {
                sql.append(" ORDER BY ").append(config.sampleIdCol);
            }

            String[] samplePaths = null;
            File baseDir = new File(sampleMetaPath).getParentFile();
            int i = 0;
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql.toString())) {
                java.sql.ResultSetMetaData meta = rs.getMetaData();
                boolean hasPath = meta.getColumnCount() >= 3;
                if (hasPath) {
                    samplePaths = new String[count];
                }
                while (rs.next()) {
                    long sampleId = rs.getLong(1);
                    sampleIds[i] = sampleId;
                    if (sampleId != i) {
                        throw new SQLException("sample_meta sample_id must equal dense row order 0..n-1; row=" + i + " value=" + sampleId);
                    }
                    Object yObj = rs.getObject(2);
                    double yVal = (yObj == null) ? Double.NaN : rs.getDouble(2);
                    if (Double.isNaN(yVal)) {
                        y[i] = Double.NaN;
                        yMask[i] = 0;
                    } else {
                        y[i] = yVal;
                        yMask[i] = 1;
                    }
                    if (hasPath) {
                        String path = rs.getString(3);
                        if (path != null && baseDir != null) {
                            File f = new File(path);
                            if (!f.isAbsolute()) {
                                path = new File(baseDir, path).getAbsolutePath();
                            }
                        }
                        samplePaths[i] = path;
                    }
                    i++;
                }
            }
            return new SampleMeta(sampleIds, y, yMask, samplePaths);
        }
    }

    public static SampleMeta loadTargets(String sampleMetaPath, BuildShardConfig config, boolean orderBySampleId) throws SQLException {
        BuildShardConfig copy = copyConfig(config);
        copy.pathCol = "";
        SampleMeta meta = load(sampleMetaPath, copy, orderBySampleId);
        return new SampleMeta(meta.sampleIds, meta.y, meta.yMask, null);
    }

    public static SampleMeta loadTargets(String sampleMetaPath, String yCol, String sampleIdCol) throws SQLException {
        BuildShardConfig cfg = new BuildShardConfig();
        cfg.yCol = yCol;
        cfg.sampleIdCol = sampleIdCol;
        cfg.pathCol = "";
        return loadTargets(sampleMetaPath, cfg, true);
    }

    public static SampleMeta load(String sampleMetaPath) throws SQLException {
        BuildShardConfig cfg = new BuildShardConfig();
        return load(sampleMetaPath, cfg, true);
    }

    private static BuildShardConfig copyConfig(BuildShardConfig cfg) {
        BuildShardConfig out = new BuildShardConfig();
        out.nShards = cfg.nShards;
        out.targetShardBytes = cfg.targetShardBytes;
        out.featureIdCol = cfg.featureIdCol;
        out.valueCol = cfg.valueCol;
        out.sampleIdCol = cfg.sampleIdCol;
        out.sampleKeyCol = cfg.sampleKeyCol;
        out.featureKeyCol = cfg.featureKeyCol;
        out.featureMetaPath = cfg.featureMetaPath;
        out.pathCol = cfg.pathCol;
        out.yCol = cfg.yCol;
        out.statsYCols = cfg.statsYCols;
        out.valuesType = cfg.valuesType;
        out.validType = cfg.validType;
        return out;
    }

    private static int countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
    }

    private static boolean hasColumn(Connection conn, String path, String columnName) throws SQLException {
        if (columnName == null || columnName.isEmpty()) {
            return false;
        }
        String sql = "SELECT * FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") LIMIT 0";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            java.sql.ResultSetMetaData meta = rs.getMetaData();
            for (int i = 1; i <= meta.getColumnCount(); i++) {
                if (columnName.equalsIgnoreCase(meta.getColumnLabel(i))) {
                    return true;
                }
            }
            return false;
        }
    }
}
