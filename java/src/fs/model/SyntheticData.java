package fs.model;

public class SyntheticData {
    public final double[][] X;
    public final byte[][] M;
    public final double[] y;
    public final int[] featureIds;

    public SyntheticData(double[][] X, byte[][] M, double[] y, int[] featureIds) {
        this.X = X;
        this.M = M;
        this.y = y;
        this.featureIds = featureIds;
    }
}
