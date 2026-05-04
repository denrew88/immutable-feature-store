package fs.model.scalar;

/**
 * Sample 하나에 대한 scalar 값과 존재 여부를 나타내는 public 모델이다.
 */
public class ScalarValue {
    public final long sampleId;
    public final String sampleKey;
    public final boolean present;
    public final Double value;

    public ScalarValue(long sampleId, String sampleKey, boolean present, Double value) {
        this.sampleId = sampleId;
        this.sampleKey = sampleKey;
        this.present = present;
        this.value = value;
    }
}
