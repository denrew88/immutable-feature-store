package fs.io;

import fs.model.Feature;
import fs.model.RowBatch;
import fs.model.ShardManifest;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class DuckDBShardReader implements ShardReader, AutoCloseable {
    private final ShardManifest manifest;
    private final int maxGap;
    private final Connection conn;

    public DuckDBShardReader(ShardManifest manifest, int maxGap) throws SQLException {
        this.manifest = manifest;
        this.maxGap = maxGap;
        this.conn = DuckDBUtils.connect(null);
    }

    public DuckDBShardReader(ShardManifest manifest) throws SQLException {
        this(manifest, 0);
    }

    @Override
    public int nSamples() {
        return manifest.nSamples;
    }

    @Override
    public RowBatch loadRows(int shardId, int[] offsets) throws SQLException {
        if (offsets == null || offsets.length == 0) {
            return new RowBatch(new double[0][], new byte[0][]);
        }

        int[] order = argsort(offsets);
        int[] sorted = new int[offsets.length];
        for (int i = 0; i < offsets.length; i++) {
            sorted[i] = offsets[order[i]];
        }

        List<int[]> ranges = groupOffsets(sorted, maxGap);
        double[][] sortedValues = new double[offsets.length][];
        byte[][] sortedValid = new byte[offsets.length][];

        int pos = 0;
        for (int[] range : ranges) {
            int start = range[0];
            int end = range[1];
            int length = end - start + 1;
            RangeBatch batch = readRange(shardId, start, length);
            while (pos < sorted.length && sorted[pos] <= end) {
                int off = sorted[pos];
                int idx = off - start;
                sortedValues[pos] = batch.values[idx];
                sortedValid[pos] = batch.valid[idx];
                pos++;
            }
        }

        double[][] values = new double[offsets.length][];
        byte[][] valid = new byte[offsets.length][];
        for (int i = 0; i < offsets.length; i++) {
            int orig = order[i];
            values[orig] = sortedValues[i];
            valid[orig] = sortedValid[i];
        }
        return new RowBatch(values, valid);
    }

    @Override
    public Feature loadFeatureByOffset(int shardId, int offset) throws SQLException {
        RowBatch batch = loadRows(shardId, new int[]{offset});
        if (batch.values.length == 0) {
            return new Feature(new double[0], new byte[0]);
        }
        return new Feature(batch.values[0], batch.valid[0]);
    }

    @Override
    public void close() throws SQLException {
        conn.close();
    }

    private RangeBatch readRange(int shardId, int start, int length) throws SQLException {
        String path = manifest.shardFilePath(shardId);
        String sql = "SELECT value_len, values_blob, valid_blob FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") LIMIT " + length + " OFFSET " + start;
        double[][] values = new double[length][];
        byte[][] valid = new byte[length][];
        int i = 0;
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            while (rs.next()) {
                int valueLen = rs.getInt(1);
                if (valueLen != manifest.nSamples) {
                    throw new SQLException("value_len mismatch: expected " + manifest.nSamples + " got " + valueLen);
                }
                byte[] vb = rs.getBytes(2);
                byte[] mb = rs.getBytes(3);
                values[i] = ArrayUtils.decodeDoubleArray(vb, valueLen);
                if (mb.length != valueLen) {
                    throw new SQLException("Invalid valid mask length: " + mb.length);
                }
                valid[i] = mb;
                i++;
            }
        }
        if (i != length) {
            throw new SQLException("Range read length mismatch: expected " + length + " got " + i);
        }
        return new RangeBatch(values, valid);
    }

    private static class RangeBatch {
        final double[][] values;
        final byte[][] valid;

        RangeBatch(double[][] values, byte[][] valid) {
            this.values = values;
            this.valid = valid;
        }
    }

    private static List<int[]> groupOffsets(int[] offsets, int maxGap) {
        List<int[]> ranges = new ArrayList<>();
        if (offsets.length == 0) {
            return ranges;
        }
        int start = offsets[0];
        int prev = start;
        for (int i = 1; i < offsets.length; i++) {
            int off = offsets[i];
            if (off <= prev + 1 + maxGap) {
                prev = off;
                continue;
            }
            ranges.add(new int[]{start, prev});
            start = off;
            prev = off;
        }
        ranges.add(new int[]{start, prev});
        return ranges;
    }

    private static int[] argsort(int[] arr) {
        int n = arr.length;
        Integer[] idxObj = new Integer[n];
        for (int i = 0; i < n; i++) {
            idxObj[i] = i;
        }
        Arrays.sort(idxObj, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return Integer.compare(arr[a], arr[b]);
            }
        });
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = idxObj[i];
        }
        return idx;
    }
}
