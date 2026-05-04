package fs.io.scalar;

import fs.model.common.Feature;
import fs.model.scalar.RowBatch;

import java.sql.SQLException;

/**
 * Scalar shard에서 feature row를 읽는 최소 reader 인터페이스다.
 */
public interface ShardReader {
    int nSamples();

    RowBatch loadRows(int shardId, int[] offsets) throws SQLException;

    Feature loadFeatureByOffset(int shardId, int offset) throws SQLException;
}
