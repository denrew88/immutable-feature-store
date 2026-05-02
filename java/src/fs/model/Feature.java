package fs.model;

public class Feature {
    public final double[] values;
    public final byte[] valid;

    public Feature(double[] values, byte[] valid) {
        this.values = values;
        this.valid = valid;
    }
}
