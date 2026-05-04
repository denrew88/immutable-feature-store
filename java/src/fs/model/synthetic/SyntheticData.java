package fs.model.synthetic;

/**
 * Scalar synthetic 데이터 생성 결과에서 X, Y, mask와 메타를 함께 담는 객체다.
 */
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
