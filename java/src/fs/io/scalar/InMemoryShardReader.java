package fs.io.scalar;

import fs.model.common.Feature;
import fs.model.scalar.RowBatch;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

/**
 * 테스트와 검증용으로 메모리 상의 feature 행을 그대로 반환하는 reader 구현이다.
 */
public class InMemoryShardReader implements ShardReader {
    private final List<double[][]> values;
    private final List<byte[][]> valid;
    private final int nSamples;

    public InMemoryShardReader(double[][] X, byte[][] M, int shardSize) {
        int nFeatures = X.length;
        this.nSamples = X[0].length;
        this.values = new ArrayList<>();
        this.valid = new ArrayList<>();
        for (int start = 0; start < nFeatures; start += shardSize) {
            int end = Math.min(start + shardSize, nFeatures);
            int rows = end - start;
            double[][] v = new double[rows][];
            byte[][] m = new byte[rows][];
            for (int i = 0; i < rows; i++) {
                v[i] = X[start + i];
                m[i] = M[start + i];
            }
            values.add(v);
            valid.add(m);
        }
    }

    @Override
    public int nSamples() {
        return nSamples;
    }

    @Override
    public RowBatch loadRows(int shardId, int[] offsets) throws SQLException {
        if (offsets == null || offsets.length == 0) {
            return new RowBatch(new double[0][], new byte[0][]);
        }
        double[][] v = new double[offsets.length][];
        byte[][] m = new byte[offsets.length][];
        double[][] shardValues = values.get(shardId);
        byte[][] shardValid = valid.get(shardId);
        for (int i = 0; i < offsets.length; i++) {
            int off = offsets[i];
            v[i] = shardValues[off];
            m[i] = shardValid[off];
        }
        return new RowBatch(v, m);
    }

    @Override
    public Feature loadFeatureByOffset(int shardId, int offset) throws SQLException {
        return new Feature(values.get(shardId)[offset], valid.get(shardId)[offset]);
    }
}
