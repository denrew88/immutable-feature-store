package fs.model;

import java.util.Arrays;

public class ArrayFeatureBlock {
    public final int featureId;
    public final int blockId;
    public final long sampleIdStart;
    public final int sampleCount;
    public final long pointCount;
    public final byte[] sampleFlags;
    public final long[] sampleOffsets;
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
        this.featureId = featureId;
        this.blockId = blockId;
        this.sampleIdStart = sampleIdStart;
        this.sampleCount = sampleCount;
        this.pointCount = pointCount;
        this.sampleFlags = sampleFlags;
        this.sampleOffsets = sampleOffsets;
        this.time = time;
        this.value = value;
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
        if (start < 0 || end < start || end > time.length || end > value.length) {
            throw new IllegalStateException("Invalid offsets for sample_id=" + sampleId);
        }
        int len = (int) (end - start);
        double[] traceTime = Arrays.copyOfRange(time, (int) start, (int) end);
        double[] traceValue = Arrays.copyOfRange(value, (int) start, (int) end);
        if (len == 0) {
            traceTime = new double[0];
            traceValue = new double[0];
        }
        return new ArrayTrace(sampleId, flags, traceTime, traceValue);
    }
}
