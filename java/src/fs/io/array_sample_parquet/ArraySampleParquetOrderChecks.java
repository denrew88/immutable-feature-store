package fs.io.array_sample_parquet;

import fs.io.common.DuckDBUtils;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * array_sample_parquet point/trace-index parquet의 물리 row 순서 검증 helper.
 *
 * <p>포맷은 최종 part와 raw sample 파일이 모두 {@code (sample_id, feature_id, point_idx)}
 * 순서라고 가정한다. 이 클래스는 parquet를 다시 정렬해서 비교하지 않고, DuckDB window 함수로
 * 파일에 저장된 순서 그대로 이전 row와 현재 row를 비교해 첫 역전 지점만 찾는다.</p>
 */
public final class ArraySampleParquetOrderChecks {
    private ArraySampleParquetOrderChecks() {
    }

    public static boolean pointRowsSorted(String parquetPath) throws SQLException {
        return firstPointOrderViolation(parquetPath) == null;
    }

    public static boolean traceIndexRowsSorted(String parquetPath) throws SQLException {
        return firstTraceIndexOrderViolation(parquetPath) == null;
    }

    public static void requirePointRowsSorted(String parquetPath) throws SQLException {
        OrderViolation violation = firstPointOrderViolation(parquetPath);
        if (violation != null) {
            throw new IllegalStateException("point parquet is not sorted by sample_id, feature_id, point_idx: " + violation);
        }
    }

    public static void requireTraceIndexRowsSorted(String parquetPath) throws SQLException {
        OrderViolation violation = firstTraceIndexOrderViolation(parquetPath);
        if (violation != null) {
            throw new IllegalStateException("trace index parquet is not sorted by sample_id, feature_id: " + violation);
        }
    }

    public static OrderViolation firstPointOrderViolation(String parquetPath) throws SQLException {
        String sql = "WITH ordered AS ("
                + " SELECT row_number() OVER () AS row_no,"
                + " sample_id, feature_id, point_idx,"
                + " lag(sample_id) OVER () AS prev_sample_id,"
                + " lag(feature_id) OVER () AS prev_feature_id,"
                + " lag(point_idx) OVER () AS prev_point_idx"
                + " FROM read_parquet(" + DuckDBUtils.quotePath(parquetPath) + ")"
                + ")"
                + " SELECT row_no, prev_sample_id, prev_feature_id, prev_point_idx,"
                + " sample_id, feature_id, point_idx"
                + " FROM ordered"
                + " WHERE prev_sample_id IS NOT NULL AND ("
                + " sample_id < prev_sample_id"
                + " OR (sample_id = prev_sample_id AND feature_id < prev_feature_id)"
                + " OR (sample_id = prev_sample_id AND feature_id = prev_feature_id AND point_idx < prev_point_idx)"
                + " ) LIMIT 1";
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            if (!rs.next()) {
                return null;
            }
            return new OrderViolation(
                    rs.getLong("row_no"),
                    rs.getLong("prev_sample_id"),
                    rs.getInt("prev_feature_id"),
                    rs.getInt("prev_point_idx"),
                    rs.getLong("sample_id"),
                    rs.getInt("feature_id"),
                    rs.getInt("point_idx"));
        }
    }

    public static OrderViolation firstTraceIndexOrderViolation(String parquetPath) throws SQLException {
        String sql = "WITH ordered AS ("
                + " SELECT row_number() OVER () AS row_no,"
                + " sample_id, feature_id,"
                + " lag(sample_id) OVER () AS prev_sample_id,"
                + " lag(feature_id) OVER () AS prev_feature_id"
                + " FROM read_parquet(" + DuckDBUtils.quotePath(parquetPath) + ")"
                + ")"
                + " SELECT row_no, prev_sample_id, prev_feature_id, sample_id, feature_id"
                + " FROM ordered"
                + " WHERE prev_sample_id IS NOT NULL AND ("
                + " sample_id < prev_sample_id"
                + " OR (sample_id = prev_sample_id AND feature_id < prev_feature_id)"
                + " ) LIMIT 1";
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            if (!rs.next()) {
                return null;
            }
            return new OrderViolation(
                    rs.getLong("row_no"),
                    rs.getLong("prev_sample_id"),
                    rs.getInt("prev_feature_id"),
                    null,
                    rs.getLong("sample_id"),
                    rs.getInt("feature_id"),
                    null);
        }
    }

    public static final class OrderViolation {
        public final long rowNo;
        public final long prevSampleId;
        public final int prevFeatureId;
        public final Integer prevPointIdx;
        public final long sampleId;
        public final int featureId;
        public final Integer pointIdx;

        OrderViolation(
                long rowNo,
                long prevSampleId,
                int prevFeatureId,
                Integer prevPointIdx,
                long sampleId,
                int featureId,
                Integer pointIdx) {
            this.rowNo = rowNo;
            this.prevSampleId = prevSampleId;
            this.prevFeatureId = prevFeatureId;
            this.prevPointIdx = prevPointIdx;
            this.sampleId = sampleId;
            this.featureId = featureId;
            this.pointIdx = pointIdx;
        }

        @Override
        public String toString() {
            return "rowNo=" + rowNo
                    + ", prev=(" + prevSampleId + ", " + prevFeatureId + formatPoint(prevPointIdx) + ")"
                    + ", current=(" + sampleId + ", " + featureId + formatPoint(pointIdx) + ")";
        }

        private static String formatPoint(Integer pointIdx) {
            return pointIdx == null ? "" : ", " + pointIdx.intValue();
        }
    }
}
