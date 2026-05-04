package fs.io.scalar;

import fs.config.BuildShardConfig;
import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.math.Pearson;
import fs.model.common.SampleMeta;
import fs.model.scalar.ScalarSampleBundleManifest;
import fs.model.scalar.ShardManifest;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Method;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * scalar sample-major 또는 sample bundle stage를 최종 shard dataset으로 변환하는 builder다.
 *
 * <p>핵심 흐름은
 * 1) feature/shard layout 계산
 * 2) 임시 memory-mapped row buffer 채우기
 * 3) shard parquet와 locator/selection stats/manifest 기록
 * 이다.
 */
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
            valuesBuf.putDouble(idx * 8, value);
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
            } catch (Throwable ignored) {
            }
            try {
                tryUnmap(validBuf);
            } catch (Throwable ignored) {
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

    /**
     * sample-major parquet 목록을 읽어 scalar shard dataset을 만든다.
     */
    public static String buildShardsFromSampleMajor(String sampleMetaPath, String outDir, BuildShardConfig config) throws Exception {
        BuildShardConfig cfg = normalizeConfig(config);
        File featureMetaFile = resolveFeatureMetaPath(sampleMetaPath, cfg.featureIdCol, cfg);
        SampleMeta meta = SampleMetaLoader.load(sampleMetaPath, cfg, true);
        if (meta.samplePaths == null) {
            throw new IllegalArgumentException("sample_meta parquet must contain sample paths to build from sample-major files");
        }
        int[] featureIds = loadDenseFeatureIds(featureMetaFile.getAbsolutePath(), cfg.featureIdCol);
        return buildFromSources(sampleMetaPath, meta, featureIds, featureMetaFile, outDir, cfg, new SampleMajorRowFiller(meta.samplePaths));
    }

    /**
     * sample bundle manifest를 읽어 scalar shard dataset을 만든다.
     */
    public static String buildShardsFromSampleBundles(String sampleBundleManifestPath, String outDir, BuildShardConfig config) throws Exception {
        BuildShardConfig cfg = normalizeConfig(config);
        ScalarSampleBundleManifest stage = ScalarSampleBundleManifestIO.read(sampleBundleManifestPath);
        SampleMeta meta = SampleMetaLoader.loadTargets(stage.sampleMetaPath, cfg, true);
        int[] featureIds = loadDenseFeatureIds(stage.featureMetaPath, cfg.featureIdCol);
        cfg.sampleIdCol = stage.sampleIdCol;
        cfg.featureIdCol = stage.featureIdCol;
        cfg.valueCol = stage.valueCol;
        return buildFromSources(stage.sampleMetaPath, meta, featureIds, new File(stage.featureMetaPath), outDir, cfg, new SampleBundleRowFiller(stage.bundlePaths));
    }

    /**
     * sample-major 입력 형식을 추상화한 뒤 공통 shard build 파이프라인을 수행한다.
     *
     * <p>RowFiller가 입력만 다르게 채워 주면, 그 이후의 shard layout 계산,
     * memory-mapped row buffer 구성, parquet/locator/stats/manifest 작성은 동일하다.
     */
    private static String buildFromSources(
            String sourceSampleMetaPath,
            SampleMeta meta,
            int[] featureIds,
            File sourceFeatureMetaFile,
            String outDir,
            BuildShardConfig config,
            RowFiller rowFiller) throws Exception {
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

            int nSamples = meta.nSamples();
            int nFeatures = featureIds.length;
            HashMapIntInt idToIndex = buildIdToIndex(featureIds);
            ShardLayout layout = computeShardLayout(nFeatures, nSamples, config);

            for (int shardId = 0; shardId < layout.nShards; shardId++) {
                int start = layout.shardStarts[shardId];
                int end = layout.shardEnds[shardId];
                int rows = Math.max(0, end - start);
                if (rows == 0) {
                    shards.add(null);
                    continue;
                }
                long totalValueBytes = (long) rows * nSamples * 8L;
                long totalValidBytes = (long) rows * nSamples;
                if (totalValueBytes > Integer.MAX_VALUE || totalValidBytes > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("shard too large for memory mapping; increase shard count");
                }
                int[] shardFeatureIds = Arrays.copyOfRange(featureIds, start, end);
                File valuesFile = new File(runTmpDir, String.format("shard_%04d_values.bin", shardId));
                File validFile = new File(runTmpDir, String.format("shard_%04d_valid.bin", shardId));
                shards.add(new MappedShard(shardId, rows, nSamples, shardFeatureIds, valuesFile, validFile));
            }

            rowFiller.fill(config, idToIndex, layout, shards);
            for (MappedShard shard : shards) {
                if (shard != null) {
                    shard.flush();
                }
            }

            List<String> statsYCols = resolveStatsYCols(config);
            LinkedHashMap<String, double[]> r2yByY = new LinkedHashMap<String, double[]>();
            LinkedHashMap<String, int[]> nOverlapByY = new LinkedHashMap<String, int[]>();
            LinkedHashMap<String, SampleMeta> targetsByY = new LinkedHashMap<String, SampleMeta>();
            for (String statsYCol : statsYCols) {
                targetsByY.put(statsYCol, SampleMetaLoader.loadTargets(sourceSampleMetaPath, statsYCol, config.sampleIdCol));
                r2yByY.put(statsYCol, new double[nFeatures]);
                nOverlapByY.put(statsYCol, new int[nFeatures]);
                Arrays.fill(r2yByY.get(statsYCol), Double.NaN);
                Arrays.fill(nOverlapByY.get(statsYCol), -1);
            }

            try (Connection conn = DuckDBUtils.connect(null)) {
                for (int shardId = 0; shardId < layout.nShards; shardId++) {
                    int start = layout.shardStarts[shardId];
                    int end = layout.shardEnds[shardId];
                    int rows = Math.max(0, end - start);
                    String shardPath = new File(shardDir, String.format("shard_%04d.parquet", shardId)).getAbsolutePath();
                    if (rows == 0) {
                        writeEmptyShard(conn, shardPath);
                        continue;
                    }
                    MappedShard shard = shards.get(shardId);
                    writeShardParquet(conn, shard, shardPath, targetsByY, r2yByY, nOverlapByY, start);
                }
            }

            String sampleMetaOut = new File(out, "sample_meta.parquet").getAbsolutePath();
            String featureMetaOut = new File(out, "feature_meta.parquet").getAbsolutePath();
            materializeMetadataFile(sourceSampleMetaPath, sampleMetaOut);
            materializeMetadataFile(sourceFeatureMetaFile.getAbsolutePath(), featureMetaOut);

            String locatorPath = new File(out, "feature_locator.parquet").getAbsolutePath();
            try (Connection conn = DuckDBUtils.connect(null)) {
                writeFeatureLocatorParquet(conn, featureIds, layout.shardSize, locatorPath);
            }

            LinkedHashMap<String, String> selectionStats = new LinkedHashMap<String, String>();
            File selectionStatsDir = new File(out, "selection_stats");
            if (!selectionStatsDir.exists() && !selectionStatsDir.mkdirs()) {
                throw new IOException("failed to create selection_stats dir: " + selectionStatsDir.getAbsolutePath());
            }
            try (Connection conn = DuckDBUtils.connect(null)) {
                for (String statsYCol : statsYCols) {
                    String fileName = encodeStatsFilename(statsYCol) + ".parquet";
                    File statsFile = new File(selectionStatsDir, fileName);
                    writeSelectionStatsParquet(conn, statsFile.getAbsolutePath(), featureIds, layout.shardSize, r2yByY.get(statsYCol), nOverlapByY.get(statsYCol));
                    selectionStats.put(statsYCol, statsFile.getAbsolutePath());
                }
            }

            ShardManifest manifest = new ShardManifest(
                    sampleMetaOut,
                    featureMetaOut,
                    nSamples,
                    nFeatures,
                    shardDir.getAbsolutePath(),
                    layout.nShards,
                    locatorPath,
                    "parquet_v1",
                    "INT32",
                    "blob_float64_le_len",
                    "blob_uint8_len",
                    "dense_row_ids",
                    config.sampleKeyCol,
                    config.featureKeyCol,
                    (config.nShards > 0) ? null : Long.valueOf(config.targetShardBytes),
                    selectionStats,
                    null
            );
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

    private static void materializeMetadataFile(String sourcePath, String outPath) throws IOException {
        if (sourcePath == null || sourcePath.isEmpty()) {
            return;
        }
        File source = new File(sourcePath).getAbsoluteFile();
        File target = new File(outPath).getAbsoluteFile();
        if (source.getCanonicalPath().equals(target.getCanonicalPath())) {
            return;
        }
        Files.copy(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }

    /**
     * build 설정을 기본값과 제약 조건에 맞게 정규화한다.
     */
    private static BuildShardConfig normalizeConfig(BuildShardConfig config) {
        BuildShardConfig cfg = (config == null) ? new BuildShardConfig() : config;
        if (cfg.nShards < 0) {
            throw new IllegalArgumentException("nShards must be >= 0");
        }
        if (cfg.nShards == 0 && cfg.targetShardBytes <= 0L) {
            throw new IllegalArgumentException("targetShardBytes must be > 0 when nShards is not set");
        }
        return cfg;
    }

    private static File resolveFeatureMetaPath(String sampleMetaPath, String featureIdCol, BuildShardConfig config) {
        if (config.featureMetaPath != null && !config.featureMetaPath.isEmpty()) {
            File explicit = new File(config.featureMetaPath).getAbsoluteFile();
            if (!explicit.exists()) {
                throw new IllegalArgumentException("feature_meta not found: " + explicit.getAbsolutePath());
            }
            return explicit;
        }
        if (config.featureKeyCol == null || config.featureKeyCol.isEmpty()) {
            config.featureKeyCol = "feature_key";
        }
        File sampleMetaFile = new File(sampleMetaPath).getAbsoluteFile();
        File candidate = new File(sampleMetaFile.getParentFile(), "feature_meta.parquet");
        if (!candidate.exists()) {
            throw new IllegalArgumentException("feature_meta.parquet not found next to sample_meta: " + candidate.getAbsolutePath());
        }
        return candidate;
    }

    private static int[] loadDenseFeatureIds(String featureMetaPath, String featureIdCol) throws SQLException {
        try (Connection conn = DuckDBUtils.connect(null)) {
            int count = countRows(conn, featureMetaPath);
            int[] featureIds = new int[count];
            String sql = "SELECT " + featureIdCol + " FROM read_parquet(" + DuckDBUtils.quotePath(featureMetaPath) + ") ORDER BY " + featureIdCol;
            int i = 0;
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                while (rs.next()) {
                    long value = rs.getLong(1);
                    if (value != i) {
                        throw new SQLException("feature_meta " + featureIdCol + " must equal dense row order 0..n-1; row=" + i + " value=" + value);
                    }
                    featureIds[i] = toIntFeatureId(value);
                    i++;
                }
            }
            if (i != count) {
                throw new SQLException("feature_meta row count mismatch");
            }
            return featureIds;
        }
    }

    private static List<String> resolveStatsYCols(BuildShardConfig config) {
        ArrayList<String> out = new ArrayList<String>();
        if (config.statsYCols != null) {
            for (String value : config.statsYCols) {
                if (value != null && !value.isEmpty() && !out.contains(value)) {
                    out.add(value);
                }
            }
        }
        if (out.isEmpty()) {
            out.add(config.yCol);
        }
        return out;
    }

    /**
     * feature 행을 shard별 contiguous 구간으로 나누는 layout을 계산한다.
     *
     * <p>설정에 따라 shard 개수를 직접 고정하거나, feature 하나의 예상 바이트 수를 이용해
     * target shard 크기에서 적절한 shard size를 역산한다.
     */
    private static ShardLayout computeShardLayout(int nFeatures, int nSamples, BuildShardConfig config) {
        if (nFeatures <= 0) {
            return new ShardLayout(1, 1, new int[]{0}, new int[]{0});
        }
        if (config.nShards > 0) {
            int shardSize = (nFeatures + config.nShards - 1) / config.nShards;
            int[] starts = new int[config.nShards];
            int[] ends = new int[config.nShards];
            for (int i = 0; i < config.nShards; i++) {
                starts[i] = i * shardSize;
                ends[i] = Math.min((i + 1) * shardSize, nFeatures);
            }
            return new ShardLayout(config.nShards, shardSize, starts, ends);
        }
        long featureBytes = Math.max(1L, (long) nSamples * 9L + 64L);
        int shardSize = Math.max(1, (int) (config.targetShardBytes / featureBytes));
        int nShards = Math.max(1, (nFeatures + shardSize - 1) / shardSize);
        int[] starts = new int[nShards];
        int[] ends = new int[nShards];
        for (int i = 0; i < nShards; i++) {
            starts[i] = i * shardSize;
            ends[i] = Math.min((i + 1) * shardSize, nFeatures);
        }
        return new ShardLayout(nShards, shardSize, starts, ends);
    }

    private static void writeEmptyShard(Connection conn, String shardPath) throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_shard (feature_id INTEGER, value_len INTEGER, values_blob BLOB, valid_blob BLOB)");
            st.execute("COPY tmp_shard TO " + DuckDBUtils.quotePath(shardPath) + " (FORMAT PARQUET)");
            st.execute("DROP TABLE tmp_shard");
        }
    }

    private static void writeShardParquet(
            Connection conn,
            MappedShard shard,
            String shardPath,
            Map<String, SampleMeta> targetsByY,
            Map<String, double[]> r2yByY,
            Map<String, int[]> nOverlapByY,
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

                double[] values = ArrayUtils.decodeDoubleArray(valuesBytes, shard.nSamples);
                for (Map.Entry<String, SampleMeta> entry : targetsByY.entrySet()) {
                    Pearson.PairwiseResult stats = Pearson.pairwiseR2(entry.getValue().y, entry.getValue().yMask, values, validBytes, 0);
                    r2yByY.get(entry.getKey())[globalStart + row] = stats.r2;
                    nOverlapByY.get(entry.getKey())[globalStart + row] = stats.n;
                }

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

    private static void writeFeatureLocatorParquet(Connection conn, int[] featureIds, int shardSize, String locatorPath) throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_feature_locator (feature_id INTEGER, global_rank INTEGER, shard_id INTEGER, offset_in_shard INTEGER)");
        }
        String insertSql = "INSERT INTO tmp_feature_locator VALUES (?, ?, ?, ?)";
        try (PreparedStatement ps = conn.prepareStatement(insertSql)) {
            int batch = 0;
            for (int i = 0; i < featureIds.length; i++) {
                int shardId = i / shardSize;
                int offsetInShard = i - shardId * shardSize;
                ps.setInt(1, featureIds[i]);
                ps.setInt(2, i);
                ps.setInt(3, shardId);
                ps.setInt(4, offsetInShard);
                ps.addBatch();
                batch++;
                if (batch >= 1024) {
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

    private static void writeSelectionStatsParquet(
            Connection conn,
            String path,
            int[] featureIds,
            int shardSize,
            double[] r2y,
            int[] nYOverlap) throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_selection_stats (feature_id INTEGER, shard_id INTEGER, offset_in_shard INTEGER, r2y DOUBLE, n_y_overlap INTEGER)");
        }
        String insertSql = "INSERT INTO tmp_selection_stats VALUES (?, ?, ?, ?, ?)";
        try (PreparedStatement ps = conn.prepareStatement(insertSql)) {
            int batch = 0;
            for (int i = 0; i < featureIds.length; i++) {
                int shardId = i / shardSize;
                int offsetInShard = i - shardId * shardSize;
                ps.setInt(1, featureIds[i]);
                ps.setInt(2, shardId);
                ps.setInt(3, offsetInShard);
                ps.setDouble(4, r2y[i]);
                ps.setInt(5, nYOverlap[i]);
                ps.addBatch();
                batch++;
                if (batch >= 1024) {
                    ps.executeBatch();
                    batch = 0;
                }
            }
            if (batch > 0) {
                ps.executeBatch();
            }
        }
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_selection_stats TO " + DuckDBUtils.quotePath(path) + " (FORMAT PARQUET)");
            st.execute("DROP TABLE tmp_selection_stats");
        }
    }

    private static String encodeStatsFilename(String yCol) {
        try {
            return URLEncoder.encode(yCol, StandardCharsets.UTF_8.name()).replace("+", "%20");
        } catch (java.io.UnsupportedEncodingException e) {
            throw new IllegalStateException("UTF-8 encoding is unavailable", e);
        }
    }

    private static HashMapIntInt buildIdToIndex(int[] featureIds) {
        HashMapIntInt out = new HashMapIntInt(Math.max(16, featureIds.length * 2));
        for (int i = 0; i < featureIds.length; i++) {
            out.put(featureIds[i], i);
        }
        return out;
    }

    private static int countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
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
            } catch (IOException ignored) {
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
        } catch (Throwable ignored) {
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
            // best-effort cleanup
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

    /**
     * 입력 소스가 다르더라도 공통 memory-mapped shard buffer를 채우게 하는 추상화다.
     */
    private interface RowFiller {
        void fill(BuildShardConfig config, HashMapIntInt idToIndex, ShardLayout layout, List<MappedShard> shards) throws SQLException;

    }

    /**
     * sample-major parquet 파일들을 순서대로 읽어 memory-mapped shard row를 채운다.
     */
    private static final class SampleMajorRowFiller implements RowFiller {
        private final String[] samplePaths;

        SampleMajorRowFiller(String[] samplePaths) {
            this.samplePaths = samplePaths;
        }

        @Override
        public void fill(BuildShardConfig config, HashMapIntInt idToIndex, ShardLayout layout, List<MappedShard> shards) throws SQLException {
            try (Connection conn = DuckDBUtils.connect(null)) {
                for (int sampleIdx = 0; sampleIdx < samplePaths.length; sampleIdx++) {
                    String path = samplePaths[sampleIdx];
                    String sql = "SELECT " + config.featureIdCol + ", " + config.valueCol
                            + " FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
                    try (Statement st = conn.createStatement();
                         ResultSet rs = st.executeQuery(sql)) {
                        while (rs.next()) {
                            int fid = toIntFeatureId(rs.getLong(1));
                            Object vObj = rs.getObject(2);
                            if (vObj == null) {
                                continue;
                            }
                            double value = rs.getDouble(2);
                            if (Double.isNaN(value)) {
                                continue;
                            }
                            int idx = idToIndex.getOrDefault(fid, -1);
                            if (idx < 0) {
                                continue;
                            }
                            int shardId = idx / layout.shardSize;
                            int offset = idx - shardId * layout.shardSize;
                            MappedShard shard = shards.get(shardId);
                            if (shard == null) {
                                continue;
                            }
                            shard.setValue(offset, sampleIdx, value);
                            shard.setValid(offset, sampleIdx);
                        }
                    }
                }
            }
        }
    }

    /**
     * sample bundle parquet 파일들을 순서대로 읽어 memory-mapped shard row를 채운다.
     */
    private static final class SampleBundleRowFiller implements RowFiller {
        private final List<String> bundlePaths;

        SampleBundleRowFiller(List<String> bundlePaths) {
            this.bundlePaths = bundlePaths;
        }

        @Override
        public void fill(BuildShardConfig config, HashMapIntInt idToIndex, ShardLayout layout, List<MappedShard> shards) throws SQLException {
            try (Connection conn = DuckDBUtils.connect(null)) {
                for (String path : bundlePaths) {
                    String sql = "SELECT " + config.sampleIdCol + ", " + config.featureIdCol + ", " + config.valueCol
                            + " FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
                    try (Statement st = conn.createStatement();
                         ResultSet rs = st.executeQuery(sql)) {
                        while (rs.next()) {
                            int sampleId = (int) rs.getLong(1);
                            int fid = toIntFeatureId(rs.getLong(2));
                            Object vObj = rs.getObject(3);
                            if (vObj == null) {
                                continue;
                            }
                            double value = rs.getDouble(3);
                            if (Double.isNaN(value)) {
                                continue;
                            }
                            int idx = idToIndex.getOrDefault(fid, -1);
                            if (idx < 0) {
                                continue;
                            }
                            int shardId = idx / layout.shardSize;
                            int offset = idx - shardId * layout.shardSize;
                            MappedShard shard = shards.get(shardId);
                            if (shard == null) {
                                continue;
                            }
                            shard.setValue(offset, sampleId, value);
                            shard.setValid(offset, sampleId);
                        }
                    }
                }
            }
        }
    }

    private static final class ShardLayout {
        final int nShards;
        final int shardSize;
        final int[] shardStarts;
        final int[] shardEnds;

        ShardLayout(int nShards, int shardSize, int[] shardStarts, int[] shardEnds) {
            this.nShards = nShards;
            this.shardSize = shardSize;
            this.shardStarts = shardStarts;
            this.shardEnds = shardEnds;
        }
    }

    private static final class HashMapIntInt {
        private final java.util.HashMap<Integer, Integer> delegate;

        HashMapIntInt(int capacity) {
            this.delegate = new java.util.HashMap<Integer, Integer>(capacity);
        }

        void put(int key, int value) {
            delegate.put(Integer.valueOf(key), Integer.valueOf(value));
        }

        int getOrDefault(int key, int fallback) {
            Integer out = delegate.get(Integer.valueOf(key));
            return (out == null) ? fallback : out.intValue();
        }
    }
}
