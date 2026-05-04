package fs.io.common;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

/**
 * DuckDB JDBC 연결과 경로 quoting 같은 공통 helper를 제공한다.
 */
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

    public static String quoteIdentifier(String name) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("identifier must not be empty");
        }
        return "\"" + name.replace("\"", "\"\"") + "\"";
    }
}
