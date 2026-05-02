package fs.pipeline;

import fs.config.SelectionConfig;
import fs.io.ArrayUtils;
import fs.io.DuckDBUtils;
import fs.io.SampleMetaLoader;
import fs.math.Pearson;
import fs.model.Candidate;
import fs.model.SampleMeta;
import fs.model.ShardManifest;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class CandidateBuilder {
    public static List<Candidate> buildCandidatesFromShards(
            ShardManifest manifest,
            SelectionConfig config,
            String requestedYCol,
            fs.config.BuildShardConfig metaConfig) throws SQLException {
        if (manifest.statsYCol != null && manifest.statsYCol.equals(requestedYCol) && locatorHasCandidateStats(manifest.featureLocatorPath)) {
            return buildCandidatesFromLocatorStats(manifest, config);
        }

        SampleMeta meta = SampleMetaLoader.load(manifest.sampleMetaPath, metaConfig, true);
        List<Candidate> candidates = new ArrayList<>();
        try (Connection conn = DuckDBUtils.connect(null)) {
            for (int shardId = 0; shardId < manifest.nShards; shardId++) {
                String path = manifest.shardFilePath(shardId);
                int nRows = countRows(conn, path);
                for (int start = 0; start < nRows; start += config.batchSize) {
                    int length = Math.min(config.batchSize, nRows - start);
                    BatchRows batch = readBatch(conn, path, start, length, manifest.nSamples);
                    Pearson.BatchResult r = Pearson.batchR2OneVsMany(meta.y, meta.yMask, batch.values, batch.valid, config.minNonNullY);
                    for (int i = 0; i < length; i++) {
                        if (r.n[i] >= config.minNonNullY && r.r2[i] >= config.yR2Threshold) {
                            candidates.add(new Candidate(batch.featureIds[i], shardId, start + i, r.r2[i], r.n[i]));
                        }
                    }
                }
            }
        }
        Collections.sort(candidates, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate a, Candidate b) {
                int c = Double.compare(b.r2y, a.r2y);
                if (c != 0) {
                    return c;
                }
                return Integer.compare(a.featureId, b.featureId);
            }
        });
        if (config.maxCandidates > 0 && candidates.size() > config.maxCandidates) {
            return new ArrayList<Candidate>(candidates.subList(0, config.maxCandidates));
        }
        return candidates;
    }

    private static List<Candidate> buildCandidatesFromLocatorStats(ShardManifest manifest, SelectionConfig config) throws SQLException {
        List<Candidate> candidates = new ArrayList<Candidate>();
        try (Connection conn = DuckDBUtils.connect(null)) {
            String sql = "SELECT feature_id, shard_id, offset_in_shard, r2y, n_y_overlap"
                    + " FROM read_parquet(" + DuckDBUtils.quotePath(manifest.featureLocatorPath) + ")"
                    + " WHERE n_y_overlap >= " + config.minNonNullY
                    + " AND r2y >= " + config.yR2Threshold
                    + " ORDER BY r2y DESC, feature_id ASC";
            if (config.maxCandidates > 0) {
                sql += " LIMIT " + config.maxCandidates;
            }
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                while (rs.next()) {
                    candidates.add(new Candidate(
                            rs.getInt(1),
                            rs.getInt(2),
                            rs.getInt(3),
                            rs.getDouble(4),
                            rs.getInt(5)
                    ));
                }
            }
        }
        if (config.maxCandidates > 0 && candidates.size() > config.maxCandidates) {
            return new ArrayList<Candidate>(candidates.subList(0, config.maxCandidates));
        }
        return candidates;
    }

    public static List<Candidate> buildCandidatesFromInMemory(double[][] X, byte[][] M, int[] featureIds, int shardSize, SampleMeta meta, SelectionConfig config) {
        List<Candidate> candidates = new ArrayList<>();
        int nFeatures = X.length;
        int shardId = 0;
        int offset = 0;
        for (int i = 0; i < nFeatures; i++) {
            Pearson.PairwiseResult r = Pearson.pairwiseR2(meta.y, meta.yMask, X[i], M[i], config.minNonNullY);
            if (r.n >= config.minNonNullY && r.r2 >= config.yR2Threshold) {
                candidates.add(new Candidate(featureIds[i], shardId, offset, r.r2, r.n));
            }
            offset++;
            if (offset >= shardSize) {
                shardId++;
                offset = 0;
            }
        }
        Collections.sort(candidates, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate a, Candidate b) {
                int c = Double.compare(b.r2y, a.r2y);
                if (c != 0) {
                    return c;
                }
                return Integer.compare(a.featureId, b.featureId);
            }
        });
        if (config.maxCandidates > 0 && candidates.size() > config.maxCandidates) {
            return new ArrayList<Candidate>(candidates.subList(0, config.maxCandidates));
        }
        return candidates;
    }


    private static int countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
    }

    private static boolean locatorHasCandidateStats(String locatorPath) throws SQLException {
        try (Connection conn = DuckDBUtils.connect(null)) {
            String sql = "SELECT * FROM read_parquet(" + DuckDBUtils.quotePath(locatorPath) + ") LIMIT 0";
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                java.sql.ResultSetMetaData meta = rs.getMetaData();
                boolean hasR2y = false;
                boolean hasNYOverlap = false;
                for (int i = 1; i <= meta.getColumnCount(); i++) {
                    String label = meta.getColumnLabel(i);
                    if ("r2y".equalsIgnoreCase(label)) {
                        hasR2y = true;
                    } else if ("n_y_overlap".equalsIgnoreCase(label)) {
                        hasNYOverlap = true;
                    }
                }
                return hasR2y && hasNYOverlap;
            }
        }
    }

    private static BatchRows readBatch(Connection conn, String path, int start, int length, int nSamples) throws SQLException {
        String sql = "SELECT feature_id, value_len, values_blob, valid_blob FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") LIMIT " + length + " OFFSET " + start;
        int[] featureIds = new int[length];
        double[][] values = new double[length][];
        byte[][] valid = new byte[length][];
        int i = 0;
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            while (rs.next()) {
                featureIds[i] = rs.getInt(1);
                int valueLen = rs.getInt(2);
                if (valueLen != nSamples) {
                    throw new SQLException("value_len mismatch: expected " + nSamples + " got " + valueLen);
                }
                byte[] vb = rs.getBytes(3);
                byte[] mb = rs.getBytes(4);
                values[i] = ArrayUtils.decodeDoubleArray(vb, valueLen);
                if (mb.length != valueLen) {
                    throw new SQLException("Invalid valid mask length: " + mb.length);
                }
                valid[i] = mb;
                i++;
            }
        }
        if (i != length) {
            throw new SQLException("Batch read length mismatch: expected " + length + " got " + i);
        }
        return new BatchRows(featureIds, values, valid);
    }

    private static class BatchRows {
        final int[] featureIds;
        final double[][] values;
        final byte[][] valid;

        BatchRows(int[] featureIds, double[][] values, byte[][] valid) {
            this.featureIds = featureIds;
            this.values = values;
            this.valid = valid;
        }
    }
}
