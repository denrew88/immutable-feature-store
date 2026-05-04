package fs.model.scalar;

/**
 * Scalar shard에서 여러 feature row를 한 번에 읽었을 때의 중간 결과다.
 */
public class RowBatch {
    public final double[][] values;
    public final byte[][] valid;

    public RowBatch(double[][] values, byte[][] valid) {
        this.values = values;
        this.valid = valid;
    }
}
