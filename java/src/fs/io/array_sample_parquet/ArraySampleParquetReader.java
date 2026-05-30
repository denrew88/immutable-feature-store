package fs.io.array_sample_parquet;

import fs.io.common.ArrayMetadataWriter;
import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.model.array_sample_parquet.ArraySampleParquetManifest;
import fs.model.array_sample_parquet.ArraySampleParquetPart;
import fs.model.array_sample_parquet.ArraySampleParquetTrace;
import fs.model.common.PointColumnSpec;

import java.io.File;
import java.sql.Array;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * array_sample_parquet v1 reader다.
 *
 * <p>요청 sample 범위와 겹치는 part parquet만 열어서 필요한 trace row를 읽는다.
 * 이 포맷은 viewer/debugging용 sample-major 포맷이므로 binary shard처럼 feature별
 * block seek를 목표로 하지 않는다.
 */
public class ArraySampleParquetReader implements AutoCloseable {
    private final ArraySampleParquetManifest manifest;
    private Map<String, Long> sampleKeyToId;
    private Map<String, Integer> featureKeyToId;
    private Map<Long, String> sampleKeysById;
    private Map<Integer, String> featureKeysById;

    public ArraySampleParquetReader(ArraySampleParquetManifest manifest) {
        this.manifest = manifest;
    }

    public List<ArraySampleParquetTrace> loadTracesByIds(
            long[] sampleIds,
            int[] featureIds,
            boolean includeMissing,
            boolean decodeCategorical) throws Exception {
        if (sampleIds == null || sampleIds.length == 0) {
            throw new IllegalArgumentException("sampleIds must not be empty");
        }
        HashMap<Long, Boolean> sampleSet = new HashMap<Long, Boolean>();
        long minSample = Long.MAX_VALUE;
        long maxSample = Long.MIN_VALUE;
        for (long sampleId : sampleIds) {
            sampleSet.put(Long.valueOf(sampleId), Boolean.TRUE);
            minSample = Math.min(minSample, sampleId);
            maxSample = Math.max(maxSample, sampleId);
        }
        HashMap<Integer, Boolean> featureSet = null;
        if (featureIds != null) {
            featureSet = new HashMap<Integer, Boolean>();
            for (int featureId : featureIds) {
                featureSet.put(Integer.valueOf(featureId), Boolean.TRUE);
            }
        }

        ArrayList<ArraySampleParquetTrace> out = new ArrayList<ArraySampleParquetTrace>();
        Map<Long, String> sampleKeys = sampleKeysById();
        Map<Integer, String> featureKeys = featureKeysById();
        for (ArraySampleParquetPart part : manifest.parts) {
            if (part.lastSampleId < minSample || part.firstSampleId > maxSample) {
                continue;
            }
            File pointFile = new File(part.path);
            File traceIndexFile = new File(part.traceIndexPath);
            if (!traceIndexFile.exists()) {
                continue;
            }
            try (Connection conn = DuckDBUtils.connect(null);
                 Statement st = conn.createStatement()) {
                ArrayList<TraceIndexRow> traceRows = new ArrayList<TraceIndexRow>();
                HashMap<String, Integer> traceLenByKey = new HashMap<String, Integer>();
                try (ResultSet rs = st.executeQuery(buildTraceIndexSelectSql(traceIndexFile.getAbsolutePath(), sampleIds, featureIds))) {
                    while (rs.next()) {
                        long sampleId = rs.getLong("sample_id");
                        int featureId = rs.getInt("feature_id");
                        int traceLen = rs.getInt("trace_len");
                        TraceIndexRow row = new TraceIndexRow(sampleId, featureId, traceLen);
                        traceRows.add(row);
                        traceLenByKey.put(row.key(), Integer.valueOf(traceLen));
                    }
                }
                HashMap<String, LinkedHashMap<String, Object>> columnsByKey = new HashMap<String, LinkedHashMap<String, Object>>();
                if (!traceRows.isEmpty() && pointFile.exists()) {
                    try (ResultSet rs = st.executeQuery(buildPointGroupSql(pointFile.getAbsolutePath(), sampleIds, featureIds, manifest.pointSchema))) {
                        while (rs.next()) {
                            long sampleId = rs.getLong("sample_id");
                            int featureId = rs.getInt("feature_id");
                            String key = sampleId + ":" + featureId;
                            Integer traceLen = traceLenByKey.get(key);
                            if (traceLen == null) {
                                continue;
                            }
                            LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
                            for (PointColumnSpec spec : manifest.pointSchema) {
                                Object values = readList(rs, spec, traceLen.intValue());
                                columns.put(spec.name, values);
                            }
                            columnsByKey.put(key, columns);
                        }
                    }
                }
                for (TraceIndexRow row : traceRows) {
                    LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
                    LinkedHashMap<String, Object> loaded = columnsByKey.get(row.key());
                    if (loaded == null) {
                        columns.putAll(emptyColumns(decodeCategorical));
                    } else {
                        columns.putAll(loaded);
                    }
                    out.add(new ArraySampleParquetTrace(
                            row.sampleId,
                            sampleKeys.get(Long.valueOf(row.sampleId)),
                            row.featureId,
                            featureKeys.get(Integer.valueOf(row.featureId)),
                            row.traceLen,
                            true,
                            columns));
                }
            }
        }

        if (includeMissing && featureIds != null) {
            HashMap<String, Boolean> seen = new HashMap<String, Boolean>();
            for (ArraySampleParquetTrace trace : out) {
                seen.put(trace.sampleId + ":" + trace.featureId, Boolean.TRUE);
            }
            for (long sampleId : sampleIds) {
                for (int featureId : featureIds) {
                    String key = sampleId + ":" + featureId;
                    if (seen.containsKey(key)) {
                        continue;
                    }
                    out.add(new ArraySampleParquetTrace(
                            sampleId,
                            sampleKeys.get(Long.valueOf(sampleId)),
                            featureId,
                            featureKeys.get(Integer.valueOf(featureId)),
                            0,
                            false,
                            emptyColumns(decodeCategorical)));
                }
            }
        }

        Collections.sort(out, new Comparator<ArraySampleParquetTrace>() {
            @Override
            public int compare(ArraySampleParquetTrace a, ArraySampleParquetTrace b) {
                int sampleCompare = Long.compare(a.sampleId, b.sampleId);
                if (sampleCompare != 0) {
                    return sampleCompare;
                }
                return Integer.compare(a.featureId, b.featureId);
            }
        });
        return out;
    }

    public List<ArraySampleParquetTrace> loadTracesByKeys(
            String[] sampleKeys,
            String[] featureKeys,
            boolean includeMissing,
            boolean decodeCategorical) throws Exception {
        if (sampleKeys == null || sampleKeys.length == 0) {
            throw new IllegalArgumentException("sampleKeys must not be empty");
        }
        long[] sampleIds = new long[sampleKeys.length];
        Map<String, Long> sampleIndex = sampleKeyToId();
        for (int i = 0; i < sampleKeys.length; i++) {
            Long id = sampleIndex.get(sampleKeys[i]);
            if (id == null) {
                throw new IllegalArgumentException("unknown sample key: " + sampleKeys[i]);
            }
            sampleIds[i] = id.longValue();
        }
        int[] featureIds = null;
        if (featureKeys != null) {
            featureIds = new int[featureKeys.length];
            Map<String, Integer> featureIndex = featureKeyToId();
            for (int i = 0; i < featureKeys.length; i++) {
                Integer id = featureIndex.get(featureKeys[i]);
                if (id == null) {
                    throw new IllegalArgumentException("unknown feature key: " + featureKeys[i]);
                }
                featureIds[i] = id.intValue();
            }
        }
        return loadTracesByIds(sampleIds, featureIds, includeMissing, decodeCategorical);
    }

    @Override
    public void close() {
        // ParquetReader instances are opened per call and closed immediately.
    }

    private static String buildTraceIndexSelectSql(String path, long[] sampleIds, int[] featureIds) {
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT * FROM read_parquet(").append(DuckDBUtils.quotePath(path)).append(")");
        sql.append(" WHERE sample_id IN (");
        for (int i = 0; i < sampleIds.length; i++) {
            if (i > 0) {
                sql.append(", ");
            }
            sql.append(sampleIds[i]);
        }
        sql.append(")");
        if (featureIds != null) {
            sql.append(" AND feature_id IN (");
            for (int i = 0; i < featureIds.length; i++) {
                if (i > 0) {
                    sql.append(", ");
                }
                sql.append(featureIds[i]);
            }
            sql.append(")");
        }
        sql.append(" ORDER BY sample_id, feature_id");
        return sql.toString();
    }

    private static String buildPointGroupSql(String path, long[] sampleIds, int[] featureIds, List<PointColumnSpec> pointSchema) {
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT sample_id, feature_id");
        for (PointColumnSpec spec : pointSchema) {
            String name = DuckDBUtils.quoteIdentifier(spec.name);
            sql.append(", list(").append(name).append(" ORDER BY point_idx) AS ").append(name);
        }
        sql.append(" FROM read_parquet(").append(DuckDBUtils.quotePath(path)).append(")");
        sql.append(" WHERE sample_id IN (");
        for (int i = 0; i < sampleIds.length; i++) {
            if (i > 0) {
                sql.append(", ");
            }
            sql.append(sampleIds[i]);
        }
        sql.append(")");
        if (featureIds != null) {
            sql.append(" AND feature_id IN (");
            for (int i = 0; i < featureIds.length; i++) {
                if (i > 0) {
                    sql.append(", ");
                }
                sql.append(featureIds[i]);
            }
            sql.append(")");
        }
        sql.append(" GROUP BY sample_id, feature_id ORDER BY sample_id, feature_id");
        return sql.toString();
    }

    private Object readList(ResultSet rs, PointColumnSpec spec, int traceLen) throws Exception {
        Object raw = null;
        Array sqlArray = null;
        try {
            sqlArray = rs.getArray(spec.name);
        } catch (Exception ignored) {
            sqlArray = null;
        }
        if (sqlArray != null) {
            raw = sqlArray.getArray();
        } else {
            raw = rs.getObject(spec.name);
        }
        if (raw == null) {
            return ArrayUtils.emptyPointColumn(spec, false);
        }
        switch (spec.storageType) {
            case FLOAT64: {
                double[] out = toDoubleArray(raw, spec.name);
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case INT32: {
                int[] out = toIntArray(raw, spec.name);
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case INT64: {
                long[] out = toLongArray(raw, spec.name);
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case STRING: {
                String[] out = toStringArray(raw, spec.name);
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case UINT8: {
                long[] out = toLongArray(raw, spec.name);
                for (int i = 0; i < out.length; i++) {
                    out[i] = out[i] & 0xFFL;
                }
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case UINT16: {
                long[] out = toLongArray(raw, spec.name);
                for (int i = 0; i < out.length; i++) {
                    out[i] = out[i] & 0xFFFFL;
                }
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case UINT32: {
                long[] out = toLongArray(raw, spec.name);
                for (int i = 0; i < out.length; i++) {
                    out[i] = out[i] & 0xFFFFFFFFL;
                }
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            case UINT64: {
                long[] out = toLongArray(raw, spec.name);
                requireTraceLen(spec.name, traceLen, out.length);
                return out;
            }
            default:
                throw new IllegalArgumentException("unsupported storage_type: " + spec.storageType.value);
        }
    }

    private static void requireTraceLen(String columnName, int expected, int actual) {
        if (expected != actual) {
            throw new IllegalArgumentException("trace_len/list length mismatch for " + columnName + ": " + expected + " != " + actual);
        }
    }

    private static double[] toDoubleArray(Object raw, String columnName) {
        if (raw instanceof double[]) {
            return (double[]) raw;
        }
        if (raw instanceof Object[]) {
            Object[] source = (Object[]) raw;
            double[] out = new double[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = ((Number) source[i]).doubleValue();
            }
            return out;
        }
        return ArrayUtils.toDoubleArray(raw, columnName);
    }

    private static int[] toIntArray(Object raw, String columnName) {
        if (raw instanceof int[]) {
            return (int[]) raw;
        }
        if (raw instanceof Object[]) {
            Object[] source = (Object[]) raw;
            int[] out = new int[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = ((Number) source[i]).intValue();
            }
            return out;
        }
        return ArrayUtils.toIntArray(raw, columnName);
    }

    private static long[] toLongArray(Object raw, String columnName) {
        if (raw instanceof long[]) {
            return (long[]) raw;
        }
        if (raw instanceof int[]) {
            int[] source = (int[]) raw;
            long[] out = new long[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i];
            }
            return out;
        }
        if (raw instanceof Object[]) {
            Object[] source = (Object[]) raw;
            long[] out = new long[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = ((Number) source[i]).longValue();
            }
            return out;
        }
        return ArrayUtils.toLongArray(raw, columnName);
    }

    private static String[] toStringArray(Object raw, String columnName) {
        return ArrayUtils.toStringArray(raw, columnName);
    }

    private LinkedHashMap<String, Object> emptyColumns(boolean decodeCategorical) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (PointColumnSpec spec : manifest.pointSchema) {
            out.put(spec.name, ArrayUtils.emptyPointColumn(spec, decodeCategorical));
        }
        return out;
    }

    private Map<String, Long> sampleKeyToId() throws Exception {
        if (sampleKeyToId == null) {
            sampleKeyToId = buildKeyToLongId(manifest.sampleMetaPath, manifest.sampleKeyCol, "sample_id");
        }
        return sampleKeyToId;
    }

    private Map<String, Integer> featureKeyToId() throws Exception {
        if (featureKeyToId == null) {
            featureKeyToId = buildKeyToIntId(manifest.featureMetaPath, manifest.featureKeyCol, "feature_id");
        }
        return featureKeyToId;
    }

    private Map<Long, String> sampleKeysById() throws Exception {
        if (sampleKeysById == null) {
            sampleKeysById = buildLongIdToKey(manifest.sampleMetaPath, manifest.sampleKeyCol, "sample_id");
        }
        return sampleKeysById;
    }

    private Map<Integer, String> featureKeysById() throws Exception {
        if (featureKeysById == null) {
            featureKeysById = buildIntIdToKey(manifest.featureMetaPath, manifest.featureKeyCol, "feature_id");
        }
        return featureKeysById;
    }

    private static Map<String, Long> buildKeyToLongId(String path, String keyCol, String idCol) throws Exception {
        LinkedHashMap<String, Long> out = new LinkedHashMap<String, Long>();
        List<LinkedHashMap<String, Object>> rows = ArrayMetadataWriter.readRows(path);
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> row = rows.get(i);
            Object key = row.get(keyCol);
            if (key != null) {
                Object id = row.get(idCol);
                out.put(key.toString(), Long.valueOf(id == null ? i : ((Number) id).longValue()));
            }
        }
        return out;
    }

    private static Map<String, Integer> buildKeyToIntId(String path, String keyCol, String idCol) throws Exception {
        LinkedHashMap<String, Integer> out = new LinkedHashMap<String, Integer>();
        List<LinkedHashMap<String, Object>> rows = ArrayMetadataWriter.readRows(path);
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> row = rows.get(i);
            Object key = row.get(keyCol);
            if (key != null) {
                Object id = row.get(idCol);
                out.put(key.toString(), Integer.valueOf(id == null ? i : ((Number) id).intValue()));
            }
        }
        return out;
    }

    private static Map<Long, String> buildLongIdToKey(String path, String keyCol, String idCol) throws Exception {
        LinkedHashMap<Long, String> out = new LinkedHashMap<Long, String>();
        List<LinkedHashMap<String, Object>> rows = ArrayMetadataWriter.readRows(path);
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> row = rows.get(i);
            Object key = row.get(keyCol);
            if (key != null) {
                Object id = row.get(idCol);
                out.put(Long.valueOf(id == null ? i : ((Number) id).longValue()), key.toString());
            }
        }
        return out;
    }

    private static Map<Integer, String> buildIntIdToKey(String path, String keyCol, String idCol) throws Exception {
        LinkedHashMap<Integer, String> out = new LinkedHashMap<Integer, String>();
        List<LinkedHashMap<String, Object>> rows = ArrayMetadataWriter.readRows(path);
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> row = rows.get(i);
            Object key = row.get(keyCol);
            if (key != null) {
                Object id = row.get(idCol);
                out.put(Integer.valueOf(id == null ? i : ((Number) id).intValue()), key.toString());
            }
        }
        return out;
    }

    private static final class TraceIndexRow {
        final long sampleId;
        final int featureId;
        final int traceLen;

        TraceIndexRow(long sampleId, int featureId, int traceLen) {
            this.sampleId = sampleId;
            this.featureId = featureId;
            this.traceLen = traceLen;
        }

        String key() {
            return sampleId + ":" + featureId;
        }
    }
}
