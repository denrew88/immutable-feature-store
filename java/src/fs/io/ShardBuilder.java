package fs.io;

import fs.config.BuildShardConfig;
import fs.math.Pearson;
import fs.model.SampleMeta;
import fs.model.ShardManifest;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

public class ShardBuilder {
    private static final ConcurrentHashMap<String, ReentrantLock> OUT_DIR_LOCKS = new ConcurrentHashMap<String, ReentrantLock>();

    private static class MappedShard {
        final int shardId;
        final int rows;
        final int nSamples;
        final int[] featureIds;
        final MappedByteBuffer valuesBuf;
        final MappedByteBuffer validBuf;
        final FileChannel valuesChannel;
        final FileChannel validChannel;
        final int rowValueBytes;
        final int rowValidBytes;

        MappedShard(int shardId, int rows, int nSamples, int[] featureIds, File valuesFile, File validFile) throws IOException {
            this.shardId = shardId;
            this.rows = rows;
            this.nSamples = nSamples;
            this.featureIds = featureIds;
            this.rowValueBytes = nSamples * 8;
            this.rowValidBytes = nSamples;
            long totalValueBytes = (long) rows * rowValueBytes;
            long totalValidBytes = (long) rows * rowValidBytes;
            this.valuesChannel = new RandomAccessFile(valuesFile, "rw").getChannel();
            this.validChannel = new RandomAccessFile(validFile, "rw").getChannel();
            this.valuesBuf = valuesChannel.map(FileChannel.MapMode.READ_WRITE, 0, totalValueBytes);
            this.valuesBuf.order(ByteOrder.LITTLE_ENDIAN);
            this.validBuf = validChannel.map(FileChannel.MapMode.READ_WRITE, 0, totalValidBytes);
            this.validBuf.order(ByteOrder.LITTLE_ENDIAN);
        }

        void setValue(int row, int col, double value) {
            int idx = row * nSamples + col;
            int bytePos = idx * 8;
            valuesBuf.putDouble(bytePos, value);
        }

        void setValid(int row, int col) {
            int idx = row * nSamples + col;
            validBuf.put(idx, (byte) 1);
        }

        void flush() {
            valuesBuf.force();
            validBuf.force();
        }

        void close() throws IOException {
            IOException closeEx = null;
            try {
                tryUnmap(valuesBuf);
            } catch (Throwable t) {
                // best-effort; GC fallback will eventually unmap
            }
            try {
                tryUnmap(validBuf);
            } catch (Throwable t) {
                // best-effort; GC fallback will eventually unmap
            }
            try {
                valuesChannel.close();
            } catch (IOException e) {
                closeEx = e;
            }
            try {
                validChannel.close();
            } catch (IOException e) {
                if (closeEx == null) {
                    closeEx = e;
                }
            }
            if (closeEx != null) {
                throw closeEx;
            }
        }
    }

    public static String buildShardsFromSampleMajor(String sampleMetaPath, String outDir, BuildShardConfig config) throws Exception {
        File out = new File(outDir);
        String outKey = out.getCanonicalPath();
        ReentrantLock outLock = OUT_DIR_LOCKS.computeIfAbsent(outKey, new java.util.function.Function<String, ReentrantLock>() {
            @Override
            public ReentrantLock apply(String k) {
                return new ReentrantLock();
            }
        });
        outLock.lock();

        File tmpRoot = null;
        File runTmpDir = null;
        List<MappedShard> shards = new ArrayList<MappedShard>();
        try {
            if (!out.exists() && !out.mkdirs()) {
                throw new IOException("Failed to create out dir: " + outDir);
            }
            File shardDir = new File(out, "feature_shards");
            if (!shardDir.exists() && !shardDir.mkdirs()) {
                throw new IOException("Failed to create shard dir: " + shardDir.getAbsolutePath());
            }
            tmpRoot = new File(out, "_tmp");
            if (!tmpRoot.exists() && !tmpRoot.mkdirs()) {
                throw new IOException("Failed to create tmp root dir: " + tmpRoot.getAbsolutePath());
            }
            runTmpDir = createRunTmpDir(tmpRoot);

            SampleMeta meta = SampleMetaLoader.load(sampleMetaPath, config, true);
            int nSamples = meta.nSamples();

            int[] featureIds = collectFeatureIds(meta.samplePaths, config.featureIdCol);
            int nFeatures = featureIds.length;

            HashMap<Integer, Integer> idToIndex = new HashMap<Integer, Integer>(nFeatures * 2);
            for (int i = 0; i < nFeatures; i++) {
                idToIndex.put(featureIds[i], i);
            }

            int shardSize = (nFeatures + config.nShards - 1) / config.nShards;
            int[] shardStarts = new int[config.nShards];
            int[] shardEnds = new int[config.nShards];
            for (int s = 0; s < config.nShards; s++) {
                shardStarts[s] = s * shardSize;
                shardEnds[s] = Math.min((s + 1) * shardSize, nFeatures);
            }

            for (int s = 0; s < config.nShards; s++) {
                int start = shardStarts[s];
                int end = shardEnds[s];
                int rows = Math.max(0, end - start);
                if (rows == 0) {
                    shards.add(null);
                    continue;
                }
                long totalValueBytes = (long) rows * nSamples * 8;
                long totalValidBytes = (long) rows * nSamples;
                if (totalValueBytes > Integer.MAX_VALUE || totalValidBytes > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("Shard too large for memory mapping. Increase nShards. shard=" + s);
                }
                int[] shardFeatureIds = Arrays.copyOfRange(featureIds, start, end);
                File valuesFile = new File(runTmpDir, String.format("shard_%04d_values.bin", s));
                File validFile = new File(runTmpDir, String.format("shard_%04d_valid.bin", s));
                shards.add(new MappedShard(s, rows, nSamples, shardFeatureIds, valuesFile, validFile));
            }

            try (Connection conn = DuckDBUtils.connect(null)) {
                for (int sIdx = 0; sIdx < meta.samplePaths.length; sIdx++) {
                    String path = meta.samplePaths[sIdx];
                    String sql = "SELECT " + config.featureIdCol + ", " + config.valueCol
                            + " FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
                    try (Statement st = conn.createStatement();
                         ResultSet rs = st.executeQuery(sql)) {
                        while (rs.next()) {
                            long fidLong = rs.getLong(1);
                            int fid = toIntFeatureId(fidLong);
                            Object vObj = rs.getObject(2);
                            if (vObj == null) {
                                continue;
                            }
                            double v = rs.getDouble(2);
                            if (Double.isNaN(v)) {
                                continue;
                            }
                            Integer idxObj = idToIndex.get(fid);
                            int idx = (idxObj == null) ? -1 : idxObj.intValue();
                            if (idx < 0) {
                                continue;
                            }
                            int shardId = idx / shardSize;
                            int offset = idx - shardId * shardSize;
                            MappedShard shard = shards.get(shardId);
                            if (shard == null) {
                                continue;
                            }
                            shard.setValue(offset, sIdx, v);
                            shard.setValid(offset, sIdx);
                        }
                    }
                }
            }

            for (MappedShard shard : shards) {
                if (shard != null) {
                    shard.flush();
                }
            }

            double[] r2yByRank = new double[nFeatures];
            int[] nYOverlapByRank = new int[nFeatures];
            Arrays.fill(r2yByRank, Double.NaN);
            Arrays.fill(nYOverlapByRank, -1);

            try (Connection conn = DuckDBUtils.connect(null)) {
                for (int s = 0; s < config.nShards; s++) {
                    int start = shardStarts[s];
                    int end = shardEnds[s];
                    int rows = Math.max(0, end - start);
                    String shardPath = new File(shardDir, String.format("shard_%04d.parquet", s)).getAbsolutePath();
                    if (rows == 0) {
                        try (Statement st = conn.createStatement()) {
                            st.execute("CREATE TEMP TABLE tmp_shard (feature_id INTEGER, value_len INTEGER, values_blob BLOB, valid_blob BLOB)");
                            st.execute("COPY tmp_shard TO " + DuckDBUtils.quotePath(shardPath) + " (FORMAT PARQUET)");
                            st.execute("DROP TABLE tmp_shard");
                        }
                        continue;
                    }
                    MappedShard shard = shards.get(s);
                    if (shard == null) {
                        continue;
                    }
                    writeShardParquet(conn, shard, shardPath, meta.y, meta.yMask, r2yByRank, nYOverlapByRank, start);
                }
            }

            String locatorPath = new File(out, "feature_locator.parquet").getAbsolutePath();
            try (Connection conn = DuckDBUtils.connect(null)) {
                writeFeatureLocatorParquet(conn, featureIds, shardSize, locatorPath, r2yByRank, nYOverlapByRank);
            }

            ShardManifest manifest = new ShardManifest(
                    sampleMetaPath,
                    nSamples,
                    shardDir.getAbsolutePath(),
                    config.nShards,
                    locatorPath,
                    "PARQUET_V1",
                    "INT32",
                    "BLOB_DOUBLE",
                    "BLOB_UINT8_LEN",
                    config.yCol);
            String manifestPath = new File(out, "shard_manifest.json").getAbsolutePath();
            ManifestIO.write(manifest, manifestPath);
            return manifestPath;
        } finally {
            closeShardsQuietly(shards);
            deleteRecursively(runTmpDir);
            deleteIfEmpty(tmpRoot);
            outLock.unlock();
            if (!outLock.hasQueuedThreads()) {
                OUT_DIR_LOCKS.remove(outKey, outLock);
            }
        }
    }

    private static int[] collectFeatureIds(String[] samplePaths, String featureIdCol) throws SQLException {
        if (samplePaths == null || samplePaths.length == 0) {
            return new int[0];
        }
        List<Integer> ids = new ArrayList<Integer>(1 << 20);
        String pathList = buildParquetPathListLiteral(samplePaths);
        String sql = "SELECT DISTINCT " + featureIdCol + " AS feature_id"
                + " FROM read_parquet(" + pathList + ")"
                + " WHERE " + featureIdCol + " IS NOT NULL"
                + " ORDER BY feature_id";
        try (Connection conn = DuckDBUtils.connect(null)) {
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                while (rs.next()) {
                    long fidLong = rs.getLong(1);
                    ids.add(toIntFeatureId(fidLong));
                }
            }
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            out[i] = ids.get(i);
        }
        return out;
    }

    private static String buildParquetPathListLiteral(String[] samplePaths) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < samplePaths.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(DuckDBUtils.quotePath(samplePaths[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private static int toIntFeatureId(long value) {
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("feature_id out of int32 range: " + value);
        }
        return (int) value;
    }

    private static File createRunTmpDir(File tmpRoot) throws IOException {
        for (int attempt = 0; attempt < 32; attempt++) {
            String dirName = "build_" + System.currentTimeMillis() + "_" + Thread.currentThread().getId()
                    + "_" + UUID.randomUUID().toString().replace("-", "");
            File runDir = new File(tmpRoot, dirName);
            if (runDir.mkdir()) {
                return runDir;
            }
        }
        throw new IOException("Failed to create unique run tmp dir under: " + tmpRoot.getAbsolutePath());
    }

    private static void closeShardsQuietly(List<MappedShard> shards) {
        if (shards == null) {
            return;
        }
        for (MappedShard shard : shards) {
            if (shard == null) {
                continue;
            }
            try {
                shard.close();
            } catch (IOException e) {
                // close best-effort on cleanup path
            }
        }
    }

    private static void tryUnmap(MappedByteBuffer buffer) {
        if (buffer == null) {
            return;
        }
        try {
            Method cleanerMethod = buffer.getClass().getMethod("cleaner");
            cleanerMethod.setAccessible(true);
            Object cleaner = cleanerMethod.invoke(buffer);
            if (cleaner != null) {
                Method cleanMethod = cleaner.getClass().getMethod("clean");
                cleanMethod.invoke(cleaner);
            }
        } catch (Throwable t) {
            // best-effort only
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            // best-effort cleanup; leave if OS lock remains
        }
    }

    private static void deleteIfEmpty(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        File[] children = dir.listFiles();
        if (children != null && children.length > 0) {
            return;
        }
        if (!dir.delete()) {
            // best-effort cleanup
        }
    }

    private static void writeShardParquet(
            Connection conn,
            MappedShard shard,
            String shardPath,
            double[] y,
            byte[] yMask,
            double[] r2yByRank,
            int[] nYOverlapByRank,
            int globalStart) throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_shard (feature_id INTEGER, value_len INTEGER, values_blob BLOB, valid_blob BLOB)");
        }
        String insertSql = "INSERT INTO tmp_shard VALUES (?, ?, ?, ?)";
        int batchSize = 64;
        int rowValueBytes = shard.rowValueBytes;
        int rowValidBytes = shard.rowValidBytes;

        try (PreparedStatement ps = conn.prepareStatement(insertSql)) {
            int batch = 0;
            for (int row = 0; row < shard.rows; row++) {
                byte[] valuesBytes = new byte[rowValueBytes];
                byte[] validBytes = new byte[rowValidBytes];
                ByteBuffer vb = shard.valuesBuf.duplicate();
                vb.position(row * rowValueBytes);
                vb.get(valuesBytes, 0, rowValueBytes);
                ByteBuffer mb = shard.validBuf.duplicate();
                mb.position(row * rowValidBytes);
                mb.get(validBytes, 0, rowValidBytes);

                Pearson.PairwiseResult stats = Pearson.pairwiseR2(
                        y,
                        yMask,
                        ArrayUtils.decodeDoubleArray(valuesBytes, shard.nSamples),
                        validBytes,
                        0
                );
                r2yByRank[globalStart + row] = stats.r2;
                nYOverlapByRank[globalStart + row] = stats.n;

                ps.setInt(1, shard.featureIds[row]);
                ps.setInt(2, shard.nSamples);
                ps.setBytes(3, valuesBytes);
                ps.setBytes(4, validBytes);
                ps.addBatch();
                batch++;
                if (batch >= batchSize) {
                    ps.executeBatch();
                    batch = 0;
                }
            }
            if (batch > 0) {
                ps.executeBatch();
            }
        }
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_shard TO " + DuckDBUtils.quotePath(shardPath) + " (FORMAT PARQUET)");
            st.execute("DROP TABLE tmp_shard");
        }
    }

    private static void writeFeatureLocatorParquet(
            Connection conn,
            int[] featureIds,
            int shardSize,
            String locatorPath,
            double[] r2yByRank,
            int[] nYOverlapByRank) throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_feature_locator (feature_id INTEGER, global_rank INTEGER, shard_id INTEGER, offset_in_shard INTEGER, r2y DOUBLE, n_y_overlap INTEGER)");
        }
        String insertSql = "INSERT INTO tmp_feature_locator VALUES (?, ?, ?, ?, ?, ?)";
        int batchSize = 1024;
        try (PreparedStatement ps = conn.prepareStatement(insertSql)) {
            int batch = 0;
            for (int i = 0; i < featureIds.length; i++) {
                int shardId = i / shardSize;
                int offsetInShard = i - shardId * shardSize;
                ps.setInt(1, featureIds[i]);
                ps.setInt(2, i);
                ps.setInt(3, shardId);
                ps.setInt(4, offsetInShard);
                ps.setDouble(5, r2yByRank[i]);
                ps.setInt(6, nYOverlapByRank[i]);
                ps.addBatch();
                batch++;
                if (batch >= batchSize) {
                    ps.executeBatch();
                    batch = 0;
                }
            }
            if (batch > 0) {
                ps.executeBatch();
            }
        }
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_feature_locator TO " + DuckDBUtils.quotePath(locatorPath) + " (FORMAT PARQUET)");
            st.execute("DROP TABLE tmp_feature_locator");
        }
    }
}
