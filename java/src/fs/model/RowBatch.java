package fs.model;

public class RowBatch {
    public final double[][] values;
    public final byte[][] valid;

    public RowBatch(double[][] values, byte[][] valid) {
        this.values = values;
        this.valid = valid;
    }
}
