package fs.io;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class DuckDBUtils {
    public static Connection connect(String dbPath) throws SQLException {
        try {
            Class.forName("org.duckdb.DuckDBDriver");
        } catch (ClassNotFoundException e) {
            throw new SQLException("DuckDB JDBC driver not found. Add duckdb_jdbc.jar to classpath.", e);
        }
        String url = (dbPath == null || dbPath.isEmpty()) ? "jdbc:duckdb:" : ("jdbc:duckdb:" + dbPath);
        return DriverManager.getConnection(url);
    }

    public static String quotePath(String path) {
        if (path == null) {
            return "''";
        }
        return "'" + path.replace("'", "''") + "'";
    }
}
