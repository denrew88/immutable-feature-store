package fs.model;

public class ArrayTrace {
    public final long sampleId;
    public final byte flags;
    public final double[] time;
    public final double[] value;

    public ArrayTrace(long sampleId, byte flags, double[] time, double[] value) {
        this.sampleId = sampleId;
        this.flags = flags;
        this.time = time;
        this.value = value;
    }
}
