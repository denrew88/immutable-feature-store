package fs.model.common;

/**
 * Selection과 synthetic 경로에서 쓰는 sample-level metadata 한 행을 표현한다.
 */
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
