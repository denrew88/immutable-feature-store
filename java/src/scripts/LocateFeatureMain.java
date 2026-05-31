package scripts;

import fs.io.common.DuckDBUtils;
import fs.io.scalar.ScalarDenseLongManifestIO;
import fs.model.scalar.ScalarDenseLongManifest;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

/**
 * Dense-long scalar feature가 어느 part row에 있는지 출력하는 디버그 CLI다.
 */
public class LocateFeatureMain {
    public static void main(String[] args) throws Exception {
        String manifestPath = getArg(args, "--manifest", null);
        String featureIdArg = getArg(args, "--feature-id", null);
        if (manifestPath == null || featureIdArg == null) {
            System.err.println("Usage: --manifest <path> --feature-id <signed_int32>");
            System.exit(1);
        }
        int featureId = Integer.parseInt(featureIdArg);
        ScalarDenseLongManifest manifest = ScalarDenseLongManifestIO.read(manifestPath);
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT feature_id, global_rank, part_id, offset_in_part, first_row_in_part "
                             + "FROM read_parquet(" + DuckDBUtils.quotePath(manifest.featureLocatorPath) + ") "
                             + "WHERE feature_id = " + featureId)) {
            if (!rs.next()) {
                System.out.println("NOT_FOUND\tfeature_id=" + featureId);
                return;
            }
            int partId = rs.getInt(3);
            System.out.println("feature_id=" + rs.getInt(1)
                    + "\tglobal_rank=" + rs.getInt(2)
                    + "\tpart_id=" + partId
                    + "\toffset_in_part=" + rs.getInt(4)
                    + "\tfirst_row_in_part=" + rs.getLong(5)
                    + "\tpart_path=" + manifest.parts.get(partId).path);
        }
    }

    private static String getArg(String[] args, String key, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
