package fs.io;

import fs.config.ArrayShardConfig;
import fs.model.ArrayBinaryShardInfo;
import fs.model.ArrayBundleManifest;
import fs.model.ArrayShardManifest;
import fs.model.PointColumnSpec;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ArrayShardBuilder {
    public static String buildFromBundles(String bundleManifestPath, String outDir, ArrayShardConfig config) throws Exception {
        ArrayBundleManifest bundleManifest = ArrayBundleManifestIO.read(bundleManifestPath);
        ArrayShardConfig cfg = (config == null) ? new ArrayShardConfig() : config;
        if (cfg.samplesPerBlock <= 0) {
            throw new IllegalArgumentException("samplesPerBlock must be > 0");
        }
        if (cfg.nShards <= 0 && cfg.targetShardBytes <= 0L) {
            throw new IllegalArgumentException("either nShards or targetShardBytes must be > 0");
        }

        File out = new File(outDir);
        if (!out.exists() && !out.mkdirs()) {
            throw new IOException("Failed to create out dir: " + out.getAbsolutePath());
        }
        File shardDir = new File(out, "array_binary_feature_shards");
        if (!shardDir.exists() && !shardDir.mkdirs()) {
            throw new IOException("Failed to create shard dir: " + shardDir.getAbsolutePath());
        }
        String manifestPath = new File(out, "array_binary_shard_manifest.json").getAbsolutePath();

        String sourceFeatureMetaPath = resolveFeatureMetaPath(bundleManifestPath, bundleManifest);
        String artifactSampleMetaPath = materializeMetadataFile(bundleManifest.sampleMetaPath, out, "sample_meta.parquet");
        String artifactFeatureMetaPath = materializeMetadataFile(sourceFeatureMetaPath, out, "feature_meta.parquet");
        List<PointColumnSpec> pointSchema = copyCategoricalDictionaries(bundleManifest.pointSchema, out);

        try (Connection conn = DuckDBUtils.connect(null)) {
            validateDenseIds(conn, bundleManifest.sampleMetaPath, "sample_id");
            validateDenseIds(conn, sourceFeatureMetaPath, "feature_id");
            int nFeatures = countRows(conn, sourceFeatureMetaPath);
            int blocksPerFeature = blocksPerFeature(bundleManifest.nSamples, cfg.samplesPerBlock);
            ArrayFeatureStats featureStats = collectFeatureStats(conn, bundleManifest, nFeatures, cfg.samplesPerBlock, pointSchema);
            int[][] shardPartitions = buildShardPartitions(featureStats, cfg);
            HashMap<Integer, Integer> featureToShard = buildFeatureToShard(shardPartitions);

            BinaryShardSink sink = new BinaryShardSink(shardDir, bundleManifest.nSamples, cfg.samplesPerBlock, blocksPerFeature, shardPartitions, pointSchema);
            try {
                processInputRows(conn, bundleManifest, cfg, featureToShard, sink, pointSchema);
            } finally {
                sink.close();
            }

            ArrayBinaryShardInfo[] shardInfos = sink.shardInfos();
            ArrayShardManifest manifest = new ArrayShardManifest(
                    ArrayBinaryFormat.FILE_VERSION,
                    artifactSampleMetaPath,
                    artifactFeatureMetaPath,
                    bundleManifest.nSamples,
                    nFeatures,
                    shardDir.getAbsolutePath(),
                    shardInfos.length,
                    cfg.samplesPerBlock,
                    blocksPerFeature,
                    "INT32",
                    "UINT8",
                    "INT64",
                    hasColumn(pointSchema, "time") ? "FLOAT64_LE" : "",
                    hasColumn(pointSchema, "value") ? "FLOAT64_LE" : "",
                    ArrayBinaryFormat.DEFAULT_CODEC_NAME,
                    ArrayBinaryFormat.FILE_ENDIANNESS,
                    ArrayBinaryFormat.DEFAULT_SAMPLE_KEY_COL,
                    ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL,
                    shardInfos,
                    pointSchema);
            ArrayShardManifestIO.write(manifest, manifestPath);
        }

        return manifestPath;
    }

    private static List<PointColumnSpec> copyCategoricalDictionaries(List<PointColumnSpec> pointSchema, File outDir) throws IOException {
        List<PointColumnSpec> normalized = PointColumnSpec.normalizeList(pointSchema);
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(normalized.size());
        File dictDir = new File(outDir, "categorical_dictionaries");
        for (PointColumnSpec spec : normalized) {
            if (spec.dictionaryPath == null || spec.dictionaryPath.isEmpty()) {
                out.add(spec);
                continue;
            }
            if (!dictDir.exists() && !dictDir.mkdirs()) {
                throw new IOException("failed to create categorical dictionary dir: " + dictDir.getAbsolutePath());
            }
            File src = new File(spec.dictionaryPath).getAbsoluteFile();
            File dst = new File(dictDir, src.getName()).getAbsoluteFile();
            Files.copy(src.toPath(), dst.toPath(), StandardCopyOption.REPLACE_EXISTING);
            out.add(spec.withDictionaryPath(dst.getAbsolutePath()));
        }
        return out;
    }

    private static String resolveFeatureMetaPath(String bundleManifestPath, ArrayBundleManifest bundleManifest) {
        if (bundleManifest.featureMetaPath != null && !bundleManifest.featureMetaPath.isEmpty()) {
            return bundleManifest.featureMetaPath;
        }
        File manifestDir = new File(bundleManifestPath).getAbsoluteFile().getParentFile();
        File fallback = new File(manifestDir, "array_feature_meta.parquet");
        if (fallback.exists()) {
            return fallback.getAbsolutePath();
        }
        throw new IllegalArgumentException("feature_meta_path is required in bundle manifest");
    }

    private static String materializeMetadataFile(String sourcePath, File outDir, String targetName) throws IOException {
        if (sourcePath == null || sourcePath.isEmpty()) {
            throw new IllegalArgumentException("metadata source path must not be empty");
        }
        File src = new File(sourcePath).getAbsoluteFile();
        if (!src.exists()) {
            throw new IOException("metadata source does not exist: " + src.getAbsolutePath());
        }
        File dst = new File(outDir, targetName).getAbsoluteFile();
        Files.copy(src.toPath(), dst.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return dst.getAbsolutePath();
    }

    private static int countRows(Connection conn, String parquetPath) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(parquetPath) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
    }

    private static void validateDenseIds(Connection conn, String parquetPath, String idCol) throws SQLException {
        String sql = "SELECT COUNT(*) FROM ("
                + "SELECT " + idCol + ", row_number() OVER () - 1 AS dense_id "
                + "FROM read_parquet(" + DuckDBUtils.quotePath(parquetPath) + ")"
                + ") t WHERE " + idCol + " <> dense_id";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            long mismatches = rs.getLong(1);
            if (mismatches != 0L) {
                throw new SQLException("metadata ids are not dense row ids for " + parquetPath);
            }
        }
    }

    private static int blocksPerFeature(int nSamples, int samplesPerBlock) {
        return Math.max(1, (nSamples + samplesPerBlock - 1) / samplesPerBlock);
    }

    private static ArrayFeatureStats collectFeatureStats(
            Connection conn,
            ArrayBundleManifest manifest,
            int nFeatures,
            int samplesPerBlock,
            List<PointColumnSpec> pointSchema) throws SQLException {
        long[] estimatedBytes = new long[nFeatures];
        Arrays.fill(estimatedBytes, 1L);
        if (manifest.nBundles <= 0) {
            return new ArrayFeatureStats(buildDenseFeatureIds(nFeatures), estimatedBytes);
        }
        long blockOverhead = estimateBlockOverheadBytes(samplesPerBlock);
        long bytesPerPoint = 0L;
        for (PointColumnSpec spec : pointSchema) {
            bytesPerPoint += spec.storageType.itemSize;
        }
        String sql = "SELECT feature_id, SUM(trace_len) AS total_trace_len, COUNT(DISTINCT sample_id / " + samplesPerBlock + ") AS block_count "
                + "FROM read_parquet(" + buildParquetPathListLiteral(manifest) + ") "
                + "GROUP BY feature_id ORDER BY feature_id";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            while (rs.next()) {
                int featureId = rs.getInt(1);
                if (featureId < 0 || featureId >= nFeatures) {
                    throw new SQLException("feature_id out of dense feature range: " + featureId);
                }
                long totalTraceLen = rs.getLong(2);
                long blockCount = rs.getLong(3);
                long estBytes = Math.max(1L, totalTraceLen * bytesPerPoint + blockCount * blockOverhead);
                estimatedBytes[featureId] = estBytes;
            }
        }
        return new ArrayFeatureStats(buildDenseFeatureIds(nFeatures), estimatedBytes);
    }

    private static int[] buildDenseFeatureIds(int nFeatures) {
        int[] ids = new int[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            ids[i] = i;
        }
        return ids;
    }

    private static long estimateBlockOverheadBytes(int samplesPerBlock) {
        return 64L + 9L * (long) samplesPerBlock;
    }

    private static int[][] buildShardPartitions(ArrayFeatureStats featureStats, ArrayShardConfig cfg) {
        if (cfg.nShards > 0) {
            return partitionFeatureIdsByCount(featureStats.featureIds, cfg.nShards);
        }
        return partitionFeatureIdsByTargetBytes(featureStats.featureIds, featureStats.estimatedBytes, cfg.targetShardBytes);
    }

    private static int[][] partitionFeatureIdsByCount(int[] featureIds, int nShards) {
        if (nShards <= 0) {
            throw new IllegalArgumentException("nShards must be > 0");
        }
        int[][] partitions = new int[nShards][];
        if (featureIds.length == 0) {
            for (int i = 0; i < nShards; i++) {
                partitions[i] = new int[0];
            }
            return partitions;
        }
        int shardSize = Math.max(1, (featureIds.length + nShards - 1) / nShards);
        for (int shardId = 0; shardId < nShards; shardId++) {
            int start = shardId * shardSize;
            int end = Math.min((shardId + 1) * shardSize, featureIds.length);
            if (start >= featureIds.length) {
                partitions[shardId] = new int[0];
            } else {
                partitions[shardId] = Arrays.copyOfRange(featureIds, start, end);
            }
        }
        return partitions;
    }

    private static int[][] partitionFeatureIdsByTargetBytes(int[] featureIds, long[] estimatedBytes, long targetShardBytes) {
        if (targetShardBytes <= 0L) {
            throw new IllegalArgumentException("targetShardBytes must be > 0");
        }
        if (featureIds.length == 0) {
            return new int[][]{new int[0]};
        }
        List<int[]> partitions = new ArrayList<int[]>();
        List<Integer> current = new ArrayList<Integer>();
        long currentBytes = 0L;
        for (int i = 0; i < featureIds.length; i++) {
            long estBytes = Math.max(1L, estimatedBytes[i]);
            if (!current.isEmpty() && currentBytes + estBytes > targetShardBytes) {
                partitions.add(toIntArray(current));
                current.clear();
                currentBytes = 0L;
            }
            current.add(featureIds[i]);
            currentBytes += estBytes;
        }
        if (!current.isEmpty()) {
            partitions.add(toIntArray(current));
        }
        return partitions.toArray(new int[partitions.size()][]);
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static HashMap<Integer, Integer> buildFeatureToShard(int[][] shardPartitions) {
        int estimatedSize = 16;
        for (int[] partition : shardPartitions) {
            estimatedSize += partition.length;
        }
        HashMap<Integer, Integer> out = new HashMap<Integer, Integer>(Math.max(16, estimatedSize * 2));
        for (int shardId = 0; shardId < shardPartitions.length; shardId++) {
            for (int featureId : shardPartitions[shardId]) {
                out.put(featureId, shardId);
            }
        }
        return out;
    }

    private static void processInputRows(
            Connection conn,
            ArrayBundleManifest manifest,
            ArrayShardConfig config,
            HashMap<Integer, Integer> featureToShard,
            BinaryShardSink sink,
            List<PointColumnSpec> pointSchema) throws SQLException {
        if (manifest.nBundles <= 0) {
            return;
        }
        String sql = buildSelectSql(manifest, config.samplesPerBlock, pointSchema);

        int currentFeatureId = Integer.MIN_VALUE;
        Integer currentFeatureShardId = null;
        BlockAccumulator block = null;

        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            while (rs.next()) {
                int featureId = rs.getInt(1);
                long sampleId = rs.getLong(2);
                byte flags = rs.getByte(3);
                int traceLen = rs.getInt(4);

                if (sampleId < 0 || sampleId >= manifest.nSamples) {
                    throw new SQLException("sample_id out of range: " + sampleId);
                }
                if (traceLen < 0) {
                    throw new SQLException("trace_len must be >= 0");
                }

                LinkedHashMap<String, byte[]> columnBlobs = new LinkedHashMap<String, byte[]>();
                int columnIndex = 5;
                for (PointColumnSpec spec : pointSchema) {
                    byte[] blob = rs.getBytes(columnIndex++);
                    validatePointBlobLength(blob, traceLen, spec);
                    columnBlobs.put(spec.name, (blob == null) ? new byte[0] : blob);
                }

                if (featureId != currentFeatureId) {
                    finalizeBlock(block, currentFeatureId, currentFeatureShardId, sink);
                    block = null;
                    currentFeatureId = featureId;
                    currentFeatureShardId = featureToShard.get(featureId);
                    if (currentFeatureShardId == null) {
                        throw new SQLException("missing shard assignment for feature_id=" + featureId);
                    }
                }

                int blockId = (int) (sampleId / config.samplesPerBlock);
                if (block == null || block.blockId != blockId) {
                    finalizeBlock(block, currentFeatureId, currentFeatureShardId, sink);
                    block = new BlockAccumulator(featureId, blockId, config.samplesPerBlock, manifest.nSamples, pointSchema);
                }
                block.append(sampleId, flags, traceLen, columnBlobs);
            }
        }

        finalizeBlock(block, currentFeatureId, currentFeatureShardId, sink);
    }

    private static String buildSelectSql(ArrayBundleManifest manifest, int samplesPerBlock, List<PointColumnSpec> pointSchema) {
        StringBuilder sb = new StringBuilder();
        sb.append("SELECT feature_id, sample_id, flags, trace_len");
        for (PointColumnSpec spec : pointSchema) {
            sb.append(", ").append(blobColumnName(spec.name));
        }
        sb.append(" FROM read_parquet(").append(buildParquetPathListLiteral(manifest)).append(")");
        sb.append(" ORDER BY feature_id, sample_id");
        return sb.toString();
    }

    private static void validatePointBlobLength(byte[] blob, int traceLen, PointColumnSpec spec) throws SQLException {
        int expectedBytes = ArrayUtils.pointColumnBytes(spec, traceLen);
        int actual = (blob == null) ? 0 : blob.length;
        if (actual != expectedBytes) {
            throw new SQLException(spec.name + "_blob length mismatch: expected " + expectedBytes + " got " + actual);
        }
    }

    private static void finalizeBlock(
            BlockAccumulator block,
            int currentFeatureId,
            Integer shardIdObj,
            BinaryShardSink sink) throws SQLException {
        if (block == null || shardIdObj == null) {
            return;
        }
        block.finish();
        if (!block.hasPresentRows()) {
            return;
        }

        int shardId = shardIdObj.intValue();
        byte[] sampleOffsetsBlob = ArrayUtils.encodeLongArray(block.sampleOffsets);
        LinkedHashMap<String, byte[]> columnBlobs = new LinkedHashMap<String, byte[]>();
        for (Map.Entry<String, ByteArrayOutputStream> entry : block.columnOuts.entrySet()) {
            columnBlobs.put(entry.getKey(), entry.getValue().toByteArray());
        }
        ArrayShardRow row = new ArrayShardRow(
                currentFeatureId,
                block.blockId,
                block.sampleIdStart,
                block.sampleCount,
                block.pointCount,
                Arrays.copyOf(block.sampleFlags, block.sampleFlags.length),
                sampleOffsetsBlob,
                columnBlobs);
        try {
            sink.writeRow(shardId, row);
        } catch (IOException e) {
            throw new SQLException("failed to write binary shard row", e);
        }
    }

    private static String buildParquetPathListLiteral(ArrayBundleManifest manifest) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < manifest.nBundles; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(DuckDBUtils.quotePath(manifest.bundleFilePath(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    private static boolean hasColumn(List<PointColumnSpec> pointSchema, String name) {
        for (PointColumnSpec spec : pointSchema) {
            if (spec.name.equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static String blobColumnName(String name) {
        return name + "_blob";
    }

    private static final class ArrayFeatureStats {
        final int[] featureIds;
        final long[] estimatedBytes;

        ArrayFeatureStats(int[] featureIds, long[] estimatedBytes) {
            this.featureIds = featureIds;
            this.estimatedBytes = estimatedBytes;
        }
    }

    private static final class BlockAccumulator {
        final int featureId;
        final int blockId;
        final long sampleIdStart;
        final int sampleCount;
        final byte[] sampleFlags;
        final long[] sampleOffsets;
        final LinkedHashMap<String, ByteArrayOutputStream> columnOuts;
        long pointCount;
        int nextRelativeSample;
        int presentCount;

        BlockAccumulator(int featureId, int blockId, int samplesPerBlock, int nSamples, List<PointColumnSpec> pointSchema) {
            this.featureId = featureId;
            this.blockId = blockId;
            this.sampleIdStart = ((long) blockId) * samplesPerBlock;
            long remaining = nSamples - sampleIdStart;
            this.sampleCount = (int) Math.min(samplesPerBlock, remaining);
            if (sampleCount <= 0) {
                throw new IllegalArgumentException("invalid block sample_count for block_id=" + blockId);
            }
            this.sampleFlags = new byte[sampleCount];
            this.sampleOffsets = new long[sampleCount + 1];
            this.columnOuts = new LinkedHashMap<String, ByteArrayOutputStream>();
            for (PointColumnSpec spec : pointSchema) {
                this.columnOuts.put(spec.name, new ByteArrayOutputStream());
            }
            this.pointCount = 0L;
            this.nextRelativeSample = 0;
            this.presentCount = 0;
        }

        void append(long sampleId, byte flags, int traceLen, Map<String, byte[]> columnBlobs) throws SQLException {
            int relativeSample = (int) (sampleId - sampleIdStart);
            if (relativeSample < 0 || relativeSample >= sampleCount) {
                throw new SQLException("sample_id out of block range: " + sampleId);
            }
            if (relativeSample < nextRelativeSample) {
                throw new SQLException("duplicate or unsorted (feature_id=" + featureId + ", sample_id=" + sampleId + ")");
            }

            while (nextRelativeSample < relativeSample) {
                sampleOffsets[nextRelativeSample + 1] = pointCount;
                nextRelativeSample++;
            }

            byte outFlags = flags;
            if (!ArrayFeatureFlags.isPresent(outFlags)) {
                outFlags |= ArrayFeatureFlags.PRESENT;
            }
            if (traceLen == 0 && !ArrayFeatureFlags.isEmpty(outFlags)) {
                outFlags |= ArrayFeatureFlags.EMPTY;
            }
            sampleFlags[relativeSample] = outFlags;

            if (traceLen > 0) {
                for (Map.Entry<String, byte[]> entry : columnBlobs.entrySet()) {
                    columnOuts.get(entry.getKey()).write(entry.getValue(), 0, entry.getValue().length);
                }
                pointCount += traceLen;
            }
            sampleOffsets[relativeSample + 1] = pointCount;
            nextRelativeSample = relativeSample + 1;
            presentCount++;
        }

        void finish() {
            while (nextRelativeSample < sampleCount) {
                sampleOffsets[nextRelativeSample + 1] = pointCount;
                nextRelativeSample++;
            }
        }

        boolean hasPresentRows() {
            return presentCount > 0;
        }
    }

    private static final class ArrayShardRow {
        final int featureId;
        final int blockId;
        final long sampleIdStart;
        final int sampleCount;
        final long pointCount;
        final byte[] sampleFlags;
        final byte[] sampleOffsetsBlob;
        final LinkedHashMap<String, byte[]> columnBlobs;

        ArrayShardRow(
                int featureId,
                int blockId,
                long sampleIdStart,
                int sampleCount,
                long pointCount,
                byte[] sampleFlags,
                byte[] sampleOffsetsBlob,
                LinkedHashMap<String, byte[]> columnBlobs) {
            this.featureId = featureId;
            this.blockId = blockId;
            this.sampleIdStart = sampleIdStart;
            this.sampleCount = sampleCount;
            this.pointCount = pointCount;
            this.sampleFlags = sampleFlags;
            this.sampleOffsetsBlob = sampleOffsetsBlob;
            this.columnBlobs = columnBlobs;
        }
    }

    private static final class BinaryShardSink implements AutoCloseable {
        private final File shardDir;
        private final int nShards;
        private final int nSamples;
        private final int samplesPerBlock;
        private final int blocksPerFeature;
        private final int[][] shardPartitions;
        private final List<PointColumnSpec> pointSchema;
        private final ArrayBinaryShardInfo[] shardInfos;

        private int currentShardId;
        private BinaryShardWriter currentWriter;

        BinaryShardSink(
                File shardDir,
                int nSamples,
                int samplesPerBlock,
                int blocksPerFeature,
                int[][] shardPartitions,
                List<PointColumnSpec> pointSchema) {
            this.shardDir = shardDir;
            this.nShards = shardPartitions.length;
            this.nSamples = nSamples;
            this.samplesPerBlock = samplesPerBlock;
            this.blocksPerFeature = blocksPerFeature;
            this.shardPartitions = shardPartitions;
            this.pointSchema = pointSchema;
            this.shardInfos = new ArrayBinaryShardInfo[nShards];
            this.currentShardId = -1;
            this.currentWriter = null;
        }

        void writeRow(int shardId, ArrayShardRow row) throws IOException {
            if (currentWriter == null || shardId != currentShardId) {
                rotateToShard(shardId);
            }
            currentWriter.writeRow(row);
        }

        ArrayBinaryShardInfo[] shardInfos() {
            return shardInfos;
        }

        private void rotateToShard(int shardId) throws IOException {
            if (currentWriter != null) {
                shardInfos[currentShardId] = currentWriter.closeAndBuildInfo();
            }
            if (currentShardId >= 0 && shardId < currentShardId) {
                throw new IOException("non-monotonic shard order: current=" + currentShardId + " next=" + shardId);
            }
            currentShardId = shardId;
            currentWriter = new BinaryShardWriter(
                    shardDir,
                    shardId,
                    shardPartitions[shardId],
                    nSamples,
                    samplesPerBlock,
                    blocksPerFeature,
                    pointSchema);
        }

        @Override
        public void close() throws IOException {
            IOException first = null;
            if (currentWriter != null) {
                try {
                    shardInfos[currentShardId] = currentWriter.closeAndBuildInfo();
                } catch (IOException e) {
                    first = e;
                } finally {
                    currentWriter = null;
                }
            }
            for (int shardId = 0; shardId < nShards; shardId++) {
                if (shardInfos[shardId] != null) {
                    continue;
                }
                try {
                    shardInfos[shardId] = new BinaryShardWriter(
                            shardDir,
                            shardId,
                            shardPartitions[shardId],
                            nSamples,
                            samplesPerBlock,
                            blocksPerFeature,
                            pointSchema).closeAndBuildInfo();
                } catch (IOException e) {
                    if (first == null) {
                        first = e;
                    }
                }
            }
            if (first != null) {
                throw first;
            }
        }
    }

    private static final class BinaryShardWriter {
        private final int shardId;
        private final int[] featureIds;
        private final int blocksPerFeature;
        private final List<PointColumnSpec> pointSchema;
        private final File blocksIndexFile;
        private final File blocksDataFile;
        private final RandomAccessFile blocksIndex;
        private final RandomAccessFile blocksData;
        private final int featureCount;
        private final int featureIdStart;
        private final int featureIdEnd;
        private final int blockCount;

        private long nextRecordIndex;
        private boolean closed;

        BinaryShardWriter(
                File shardDir,
                int shardId,
                int[] featureIds,
                int nSamples,
                int samplesPerBlock,
                int blocksPerFeature,
                List<PointColumnSpec> pointSchema) throws IOException {
            this.shardId = shardId;
            this.featureIds = (featureIds == null) ? new int[0] : featureIds;
            this.blocksPerFeature = blocksPerFeature;
            this.pointSchema = pointSchema;
            this.featureCount = this.featureIds.length;
            this.featureIdStart = (featureCount == 0) ? 0 : this.featureIds[0];
            this.featureIdEnd = (featureCount == 0) ? -1 : this.featureIds[featureCount - 1];
            this.blockCount = featureCount * blocksPerFeature;
            this.blocksIndexFile = new File(shardDir, ArrayBinaryFormat.blocksIndexName(shardId));
            this.blocksDataFile = new File(shardDir, ArrayBinaryFormat.blocksDataName(shardId));
            this.blocksIndex = new RandomAccessFile(blocksIndexFile, "rw");
            this.blocksData = new RandomAccessFile(blocksDataFile, "rw");
            ArrayBinaryFormat.writePlaceholderHeader(blocksIndex);
            ArrayBinaryFormat.writePlaceholderHeader(blocksData);
            this.blocksData.seek(ArrayBinaryFormat.FILE_HEADER_BYTES);
            this.nextRecordIndex = 0L;
            this.closed = false;
        }

        void writeRow(ArrayShardRow row) throws IOException {
            if (closed) {
                throw new IOException("writer already closed");
            }
            if (featureCount == 0) {
                throw new IOException("cannot write row into empty shard");
            }
            int localFeature = row.featureId - featureIdStart;
            if (localFeature < 0 || localFeature >= featureCount) {
                throw new IOException("feature_id out of shard range: " + row.featureId);
            }
            long targetRecordIndex = ((long) localFeature) * (long) blocksPerFeature + (long) row.blockId;
            if (targetRecordIndex < nextRecordIndex) {
                throw new IOException("non-monotonic block order in shard " + shardId);
            }
            while (nextRecordIndex < targetRecordIndex) {
                writeEmptyRecord();
            }
            ByteArrayOutputStream encodedColumns = new ByteArrayOutputStream();
            for (PointColumnSpec spec : pointSchema) {
                byte[] blob = row.columnBlobs.get(spec.name);
                encodedColumns.write(blob, 0, (blob == null) ? 0 : blob.length);
            }
            long dataOffset = blocksData.getFilePointer();
            long dataLength = ArrayBinaryFormat.writeBlockPayload(
                    blocksData,
                    row.featureId,
                    row.blockId,
                    row.sampleIdStart,
                    row.sampleCount,
                    row.pointCount,
                    row.sampleFlags,
                    row.sampleOffsetsBlob,
                    encodedColumns.toByteArray(),
                    pointSchema.size(),
                    ArrayBinaryFormat.CODEC_NONE);
            ArrayBinaryFormat.writeBlockRecord(
                    blocksIndex,
                    dataOffset,
                    dataLength,
                    row.pointCount,
                    ArrayBinaryFormat.CODEC_NONE);
            nextRecordIndex++;
        }

        ArrayBinaryShardInfo closeAndBuildInfo() throws IOException {
            if (closed) {
                return buildInfo();
            }
            IOException first = null;
            try {
                while (nextRecordIndex < (long) blockCount) {
                    writeEmptyRecord();
                }
                ArrayBinaryFormat.writeFileHeader(
                        blocksIndex,
                        ArrayBinaryFormat.BLOCKS_INDEX_MAGIC,
                        ArrayBinaryFormat.FILE_VERSION,
                        ArrayBinaryFormat.BLOCK_RECORD_BYTES,
                        blockCount,
                        featureCount,
                        shardId);
                long dataBytes = Math.max(0L, blocksData.length() - ArrayBinaryFormat.FILE_HEADER_BYTES);
                ArrayBinaryFormat.writeFileHeader(
                        blocksData,
                        ArrayBinaryFormat.BLOCKS_DATA_MAGIC,
                        ArrayBinaryFormat.FILE_VERSION,
                        0,
                        blockCount,
                        dataBytes,
                        shardId);
            } catch (IOException e) {
                first = e;
            }
            try {
                blocksIndex.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                }
            }
            try {
                blocksData.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                }
            }
            closed = true;
            if (first != null) {
                throw first;
            }
            return buildInfo();
        }

        private void writeEmptyRecord() throws IOException {
            ArrayBinaryFormat.writeBlockRecord(blocksIndex, 0L, 0L, 0L, ArrayBinaryFormat.CODEC_NONE);
            nextRecordIndex++;
        }

        private ArrayBinaryShardInfo buildInfo() {
            return new ArrayBinaryShardInfo(
                    shardId,
                    featureIdStart,
                    featureIdEnd,
                    featureCount,
                    blockCount,
                    blocksIndexFile.getName(),
                    blocksDataFile.getName());
        }
    }
}
