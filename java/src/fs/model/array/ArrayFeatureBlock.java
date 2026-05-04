package fs.model.array;

import fs.model.common.PointColumnSpec;

import fs.io.common.ArrayUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Array binary shard에서 feature 하나의 block payload를 메모리 상으로 표현한 객체다.
 */
public class ArrayFeatureBlock {
    public final int featureId;
    public final int blockId;
    public final long sampleIdStart;
    public final int sampleCount;
    public final long pointCount;
    public final byte[] sampleFlags;
    public final long[] sampleOffsets;
    public final Map<String, Object> columns;

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
        return new ArrayTrace(sampleId, flags, traceColumns);
    }
}
