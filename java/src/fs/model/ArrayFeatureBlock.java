package fs.model;

import fs.io.ArrayUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public class ArrayFeatureBlock {
    public final int featureId;
    public final int blockId;
    public final long sampleIdStart;
    public final int sampleCount;
    public final long pointCount;
    public final byte[] sampleFlags;
    public final long[] sampleOffsets;
    public final Map<String, Object> columns;
    public final double[] time;
    public final double[] value;

    public ArrayFeatureBlock(
            int featureId,
            int blockId,
            long sampleIdStart,
            int sampleCount,
            long pointCount,
            byte[] sampleFlags,
            long[] sampleOffsets,
            double[] time,
            double[] value) {
        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
        columns.put("time", (time == null) ? new double[0] : time);
        columns.put("value", (value == null) ? new double[0] : value);
        this.featureId = featureId;
        this.blockId = blockId;
        this.sampleIdStart = sampleIdStart;
        this.sampleCount = sampleCount;
        this.pointCount = pointCount;
        this.sampleFlags = sampleFlags;
        this.sampleOffsets = sampleOffsets;
        this.columns = Collections.unmodifiableMap(columns);
        this.time = (time == null) ? new double[0] : time;
        this.value = (value == null) ? new double[0] : value;
    }

    public ArrayFeatureBlock(
            int featureId,
            int blockId,
            long sampleIdStart,
            int sampleCount,
            long pointCount,
            byte[] sampleFlags,
            long[] sampleOffsets,
            Map<String, Object> columns) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        if (columns != null) {
            out.putAll(columns);
        }
        this.featureId = featureId;
        this.blockId = blockId;
        this.sampleIdStart = sampleIdStart;
        this.sampleCount = sampleCount;
        this.pointCount = pointCount;
        this.sampleFlags = sampleFlags;
        this.sampleOffsets = sampleOffsets;
        this.columns = Collections.unmodifiableMap(out);
        this.time = extractDoubleColumn(out, "time");
        this.value = extractDoubleColumn(out, "value");
    }

    public long sampleIdEnd() {
        return sampleIdStart + sampleCount - 1L;
    }

    public ArrayTrace traceForSampleId(long sampleId) {
        int idx = (int) (sampleId - sampleIdStart);
        if (idx < 0 || idx >= sampleCount) {
            return null;
        }
        byte flags = sampleFlags[idx];
        long start = sampleOffsets[idx];
        long end = sampleOffsets[idx + 1];
        if (start < 0 || end < start) {
            throw new IllegalStateException("Invalid offsets for sample_id=" + sampleId);
        }
        int len = (int) (end - start);
        LinkedHashMap<String, Object> traceColumns = new LinkedHashMap<String, Object>();
        for (Map.Entry<String, Object> entry : columns.entrySet()) {
            Object values = entry.getValue();
            int size = ArrayUtils.pointColumnLength(values);
            if (end > size) {
                throw new IllegalStateException("Invalid offsets for sample_id=" + sampleId + " column=" + entry.getKey());
            }
            traceColumns.put(entry.getKey(), ArrayUtils.slicePointColumn(values, (int) start, (int) end));
        }
        if (len == 0) {
            traceColumns.put("time", traceColumns.containsKey("time") ? traceColumns.get("time") : new double[0]);
            traceColumns.put("value", traceColumns.containsKey("value") ? traceColumns.get("value") : new double[0]);
        }
        return new ArrayTrace(sampleId, flags, traceColumns);
    }

    private static double[] extractDoubleColumn(Map<String, Object> columns, String name) {
        Object value = columns.get(name);
        if (value instanceof double[]) {
            return (double[]) value;
        }
        return new double[0];
    }
}
