package fs.io;

import fs.model.Feature;
import fs.model.RowBatch;

import java.sql.SQLException;

public interface ShardReader {
    int nSamples();

    RowBatch loadRows(int shardId, int[] offsets) throws SQLException;

    Feature loadFeatureByOffset(int shardId, int offset) throws SQLException;
}
