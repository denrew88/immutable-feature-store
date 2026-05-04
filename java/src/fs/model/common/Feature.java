package fs.model.common;

/**
 * Scalar feature row 하나의 값 배열과 valid mask를 담는 기본 모델이다.
 */
public class Feature {
    public final double[] values;
    public final byte[] valid;

    public Feature(double[] values, byte[] valid) {
        this.values = values;
        this.valid = valid;
    }
}
