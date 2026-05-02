package fs.model;

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
