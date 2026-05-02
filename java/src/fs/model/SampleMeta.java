package fs.model;

public class SampleMeta {
    public final long[] sampleIds;
    public final double[] y;
    public final byte[] yMask;
    public final String[] samplePaths;

    public SampleMeta(long[] sampleIds, double[] y, byte[] yMask, String[] samplePaths) {
        this.sampleIds = sampleIds;
        this.y = y;
        this.yMask = yMask;
        this.samplePaths = samplePaths;
    }

    public int nSamples() {
        return sampleIds.length;
    }
}
