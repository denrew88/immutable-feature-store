package fs.io;

import fs.config.BuildShardConfig;
import fs.model.SampleMeta;

import java.io.File;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class SampleMetaLoader {
    public static SampleMeta load(String sampleMetaPath, BuildShardConfig config, boolean orderBySampleId) throws SQLException {
        try (Connection conn = DuckDBUtils.connect(null)) {
            String orderClause = orderBySampleId ? (" ORDER BY " + config.sampleIdCol) : "";
            String countSql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(sampleMetaPath) + ")";
            int count;
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(countSql)) {
                rs.next();
                count = rs.getInt(1);
            }

            long[] sampleIds = new long[count];
            double[] y = new double[count];
            byte[] yMask = new byte[count];
            String[] samplePaths = new String[count];

            String sql = "SELECT " + config.sampleIdCol + ", " + config.yCol + ", " + config.pathCol
                    + " FROM read_parquet(" + DuckDBUtils.quotePath(sampleMetaPath) + ")"
                    + orderClause;
            File baseDir = new File(sampleMetaPath).getParentFile();
            int i = 0;
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                while (rs.next()) {
                    sampleIds[i] = rs.getLong(1);
                    Object yObj = rs.getObject(2);
                    double yVal = (yObj == null) ? Double.NaN : rs.getDouble(2);
                    if (Double.isNaN(yVal)) {
                        y[i] = Double.NaN;
                        yMask[i] = 0;
                    } else {
                        y[i] = yVal;
                        yMask[i] = 1;
                    }
                    String path = rs.getString(3);
                    if (path != null && baseDir != null) {
                        File f = new File(path);
                        if (!f.isAbsolute()) {
                            path = new File(baseDir, path).getAbsolutePath();
                        }
                    }
                    samplePaths[i] = path;
                    i++;
                }
            }
            return new SampleMeta(sampleIds, y, yMask, samplePaths);
        }
    }

    public static SampleMeta load(String sampleMetaPath) throws SQLException {
        BuildShardConfig cfg = new BuildShardConfig();
        return load(sampleMetaPath, cfg, true);
    }
}
