package fs.io.array;

import fs.config.ArrayShardConfig;
import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.model.array.ArrayBinaryShardInfo;
import fs.model.array.ArrayBundleManifest;
import fs.model.array.ArrayShardManifest;
import fs.model.common.PointColumnSpec;

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

/**
 * array sample-major bundle stage를 최종 array binary shard artifact로 변환하는 builder이다.
 *
 * <p>이 클래스가 담당하는 build 파이프라인은 다음 세 단계로 나뉜다.
 * 1) bundle row를 집계해서 feature별 예상 바이트 수를 계산한다.
 * 2) 예상 바이트 수를 기준으로 feature를 shard별 연속 구간으로 나눈다.
 * 3) bundle row를 다시 읽어 block 단위로 조립한 뒤 각 shard의 {@code blocks.idx}/{@code blocks.bin}에 기록한다.
 *
 * <p>즉 이 클래스는 "bundle parquet를 읽고 shard partition을 계산한 뒤 binary block 파일로 내리는"
 * array binary 전용 build 엔진이다.
 */
public class ArrayShardBuilder {
    private static final long BLOCK_HEADER_ESTIMATE_BYTES = 64L;
    private static final long BLOCK_PER_SAMPLE_CONTROL_BYTES = 9L;
    private static final String BUILD_DUCKDB_MEMORY_LIMIT = "4GB";
    private static final int BUILD_DUCKDB_THREADS = 4;

    /**
     * bundle manifest를 읽어 최종 array binary shard dataset을 만든다.
     *
     * <p>이 overload는 codec과 metadata key column 이름에 기본값을 사용한다.
     *
     * @param bundleManifestPath bundle manifest 경로
     * @param outDir 최종 artifact 출력 디렉터리
     * @param config shard 크기와 block 크기를 정하는 설정
     * @return 생성된 shard manifest 경로
     */
    public static String buildFromBundles(String bundleManifestPath, String outDir, ArrayShardConfig config) throws Exception {
        return buildFromBundles(
                bundleManifestPath,
                outDir,
                config,
                ArrayBinaryFormat.DEFAULT_CODEC_NAME,
                ArrayBinaryFormat.DEFAULT_SAMPLE_KEY_COL,
                ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL);
    }

    /**
     * bundle manifest를 읽어 최종 array binary shard dataset을 만든다.
     *
     * <p>이 함수는 실제 build 전체를 수행하는 진입점이다. 내부 동작 순서는 다음과 같다.
     * 1) manifest와 metadata 위치를 정규화한다.
     * 2) sample/feature metadata와 categorical dictionary를 최종 artifact 위치로 복사한다.
     * 3) feature별 예상 크기를 계산하고 shard partition을 만든다.
     * 4) bundle row를 다시 읽어 block 단위로 조립한다.
     * 5) 조립된 block을 shard별 {@code blocks.idx}/{@code blocks.bin} 파일에 기록한다.
     * 6) 최종 shard manifest를 작성한다.
     *
     * @param bundleManifestPath bundle manifest 경로
     * @param outDir 최종 artifact 출력 디렉터리
     * @param config shard 크기와 block 크기를 정하는 설정
     * @param codec 최종 manifest에 기록할 codec 이름. 현재는 {@code none}만 지원한다.
     * @param sampleKeyCol sample metadata에서 외부 sample key로 사용할 컬럼 이름
     * @param featureKeyCol feature metadata에서 외부 feature key로 사용할 컬럼 이름
     * @return 생성된 shard manifest 경로
     */
    public static String buildFromBundles(
            String bundleManifestPath,
            String outDir,
            ArrayShardConfig config,
            String codec,
            String sampleKeyCol,
            String featureKeyCol) throws Exception {
        ArrayBundleManifest bundleManifest = ArrayBundleManifestIO.read(bundleManifestPath);
        ArrayShardConfig cfg = (config == null) ? new ArrayShardConfig() : config;
        String codecName = (codec == null || codec.isEmpty()) ? ArrayBinaryFormat.DEFAULT_CODEC_NAME : codec.trim().toLowerCase();
        if (!ArrayBinaryFormat.DEFAULT_CODEC_NAME.equals(codecName)) {
            throw new IllegalArgumentException("java array shard builder currently supports only codec='none'");
        }
        String manifestSampleKeyCol = (sampleKeyCol == null || sampleKeyCol.isEmpty())
                ? ArrayBinaryFormat.DEFAULT_SAMPLE_KEY_COL
                : sampleKeyCol;
        String manifestFeatureKeyCol = (featureKeyCol == null || featureKeyCol.isEmpty())
                ? ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL
                : featureKeyCol;
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

        // 최종 artifact 안에서도 self-contained하게 다시 열 수 있도록 metadata와 dictionary를 복사한다.
        String sourceFeatureMetaPath = resolveFeatureMetaPath(bundleManifestPath, bundleManifest);
        String artifactSampleMetaPath = materializeMetadataFile(bundleManifest.sampleMetaPath, out, "sample_meta.parquet");
        String artifactFeatureMetaPath = materializeMetadataFile(sourceFeatureMetaPath, out, "feature_meta.parquet");
        List<PointColumnSpec> pointSchema = copyCategoricalDictionaries(bundleManifest.pointSchema, out);

        try (Connection conn = DuckDBUtils.connect(null)) {
            applyBuildDuckDbSettings(conn, out);

            // metadata의 sample_id/feature_id가 0..N-1 dense 규칙을 만족하는지 먼저 검증한다.
            validateDenseIds(conn, bundleManifest.sampleMetaPath, "sample_id");
            validateDenseIds(conn, sourceFeatureMetaPath, "feature_id");

            int nFeatures = countRows(conn, sourceFeatureMetaPath);
            int blocksPerFeature = blocksPerFeature(bundleManifest.nSamples, cfg.samplesPerBlock);

            // feature별 예상 크기를 계산하고, 이를 기준으로 shard 단위의 feature 구간을 만든다.
            ArrayFeatureStats featureStats = collectFeatureStats(conn, bundleManifest, nFeatures, cfg.samplesPerBlock, pointSchema);
            int[][] shardPartitions = buildShardPartitions(featureStats, cfg);
            HashMap<Integer, Integer> featureToShard = buildFeatureToShard(shardPartitions);

            // 정렬된 bundle row를 block으로 다시 묶어 각 shard의 binary 파일로 밀어 넣는다.
            BinaryShardSink sink = new BinaryShardSink(
                    shardDir,
                    bundleManifest.nSamples,
                    cfg.samplesPerBlock,
                    blocksPerFeature,
                    shardPartitions,
                    pointSchema);
            try {
                processInputRows(conn, bundleManifest, cfg, featureToShard, sink, pointSchema);
            } finally {
                sink.close();
            }

            // shard별 출력 파일 이름과 feature 범위 정보를 manifest에 기록한다.
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
                    codecName,
                    ArrayBinaryFormat.FILE_ENDIANNESS,
                    manifestSampleKeyCol,
                    manifestFeatureKeyCol,
                    shardInfos,
                    pointSchema);
            ArrayShardManifestIO.write(manifest, manifestPath);
        }

        return manifestPath;
    }

    /**
     * point schema 안의 categorical dictionary 파일을 최종 artifact 위치로 복사하고,
     * schema의 dictionary path도 새 위치로 갱신한다.
     *
     * <p>array binary shard는 categorical dictionary를 self-contained artifact 안에 같이 보관한다.
     * 따라서 builder 입력 단계에서 참조하던 외부 dictionary 파일을 그대로 두지 않고,
     * 출력 디렉터리 아래로 복사한 뒤 그 경로를 manifest에 반영한다.
     */
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

    /**
     * bundle manifest가 가리키는 feature metadata 경로를 결정한다.
     *
     * <p>manifest에 {@code featureMetaPath}가 있으면 그것을 우선 사용한다.
     * 없으면 bundle manifest 옆의 {@code array_feature_meta.parquet}를 관례적 기본값으로 본다.
     */
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

    /**
     * sample/feature metadata parquet를 최종 artifact 디렉터리로 복사한다.
     *
     * <p>최종 shard 디렉터리만 받아도 다시 열 수 있도록 metadata를 함께 materialize한다.
     *
     * @param sourcePath 원본 metadata parquet 경로
     * @param outDir 출력 디렉터리
     * @param targetName 출력 디렉터리 안에서 사용할 파일 이름
     * @return 복사된 metadata 파일의 절대 경로
     */
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

    /**
     * parquet 파일의 row 수를 센다.
     *
     * <p>feature 개수, sample 개수 같은 메타 수량을 확인할 때 사용하는 작은 helper이다.
     */
    private static int countRows(Connection conn, String parquetPath) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(parquetPath) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
    }

    /**
     * metadata parquet의 id 컬럼이 dense row id 규칙(0..N-1)을 만족하는지 검증한다.
     *
     * <p>이 builder는 sample_id와 feature_id가 parquet row 순서와 같은 dense 정수라고 가정한다.
     * 이 가정이 깨지면 shard 내부에서 sample/feature를 O(1) 위치 계산으로 찾을 수 없으므로
     * build 초반에 바로 실패시킨다.
     */
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

    /**
     * sample 축을 고정 block 크기로 자를 때 feature 하나가 가지는 block 수를 계산한다.
     */
    private static int blocksPerFeature(int nSamples, int samplesPerBlock) {
        return Math.max(1, (nSamples + samplesPerBlock - 1) / samplesPerBlock);
    }

    /**
     * feature별 예상 shard 바이트 수를 계산한다.
     *
     * <p>이 단계의 목적은 "어떤 feature를 같은 shard에 묶을지" 결정하기 위한 대략적인 크기 추정이다.
     * 현재 구현은 bundle row 전체를 집계해서 다음 두 값을 구한다.
     * 1) feature별 total trace length
     * 2) feature별 distinct block count
     *
     * <p>그 뒤 {@code total_trace_len * bytes_per_point + block_count * block_overhead}로
     * feature 하나가 차지할 대략적인 바이트 수를 계산한다.
     * 이 값은 정확한 on-disk 크기가 아니라 shard partition을 위한 휴리스틱이다.
     */
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

    /**
     * 0..N-1 dense feature id 배열을 만든다.
     */
    private static int[] buildDenseFeatureIds(int nFeatures) {
        int[] ids = new int[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            ids[i] = i;
        }
        return ids;
    }

    /**
     * block 하나가 payload 외에 추가로 가지는 제어 정보 바이트 수를 추정한다.
     *
     * <p>block overhead 추정에는 다음이 들어간다.
     * 1) 고정 binary payload header
     * 2) sample마다 필요한 flags + offsets 제어 영역
     */
    private static long estimateBlockOverheadBytes(int samplesPerBlock) {
        return BLOCK_HEADER_ESTIMATE_BYTES + BLOCK_PER_SAMPLE_CONTROL_BYTES * (long) samplesPerBlock;
    }

    /**
     * feature 목록을 shard 단위 partition으로 자른다.
     *
     * <p>설정에 따라 두 가지 분기가 있다.
     * 1) {@code nShards > 0}: shard 개수를 고정하고 feature를 균등 분할한다.
     * 2) 그렇지 않으면 {@code targetShardBytes} 이하가 되도록 feature 예상 크기를 누적하며 나눈다.
     */
    private static int[][] buildShardPartitions(ArrayFeatureStats featureStats, ArrayShardConfig cfg) {
        if (cfg.nShards > 0) {
            return partitionFeatureIdsByCount(featureStats.featureIds, cfg.nShards);
        }
        return partitionFeatureIdsByTargetBytes(featureStats.featureIds, featureStats.estimatedBytes, cfg.targetShardBytes);
    }

    /**
     * feature id를 shard 개수 기준으로 거의 균등하게 나눈다.
     *
     * <p>각 shard는 contiguous feature 구간을 가진다.
     */
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

    /**
     * feature id를 target shard bytes 기준으로 누적 분할한다.
     *
     * <p>현재 shard에 feature를 계속 넣다가 목표 크기를 넘기기 직전에 끊는 단순 greedy 전략을 사용한다.
     * 각 shard는 여전히 contiguous feature 구간을 유지한다.
     */
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

    /**
     * {@code List<Integer>}를 primitive int 배열로 바꾼다.
     */
    private static int[] toIntArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    /**
     * feature_id에서 shard_id를 찾는 lookup map을 만든다.
     *
     * <p>입력 row를 읽을 때 각 feature가 어느 shard로 가야 하는지 O(1)로 찾기 위해 사용한다.
     */
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

    /**
     * bundle row를 feature/block 기준으로 다시 묶어 binary shard sink로 보낸다.
     *
     * <p>이 단계가 실제 "row 기반 bundle 표현"을 "block 기반 binary 표현"으로 바꾸는 핵심이다.
     * 먼저 SQL로 {@code feature_id, sample_id} 순서로 정렬하여 조회한다. 
     * 각 feature마다 여러 block으로 쪼개서 저장하는데, feature_id로 먼저 정렬되어 있으므로 
     * 한 개 feature의 한 개 block 단위로 진행이 가능하다. 한 block이 끝나면 다음 block으로 넘어가고, 
     * 쭉 진행하여 한 개 feature가 끝나면 다음 feature로 진행하는 식이다.
     * 현재 block 데이터는 메모리에 들고 있다가 경계가 바뀌는 시점(즉 block이나 feature가 바뀔 때)에 finalize해서 sink로 넘기면 된다.
     *
     * <p>요약하면 다음 순서이다.
     * 1) buildSelectSql에서 전체 bundle데이터를 feature_id, sample_id순으로 정렬하여 sql로 조회한다.
     * 2) 정렬된 bundle row를 한 줄씩 읽는다.
     * 3) 같은 {@code (feature_id, block_id)}에 속하는 row를 {@link BlockAccumulator}에 누적한다.
     * 4) feature 또는 block 경계가 바뀌면 누적 중인 block을 {@link ArrayShardRow}로 고정한다.
     * 5) 완성된 block row를 {@link BinaryShardSink}가 shard별 writer에 전달한다.
     */
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

                // feature 경계가 바뀌면 이전 feature의 마지막 block을 먼저 마감한다.
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
                // 같은 feature 안에서 block 경계가 바뀌면 현재 block을 shard row로 고정하고 새 block을 연다.
                if (block == null || block.blockId != blockId) {
                    finalizeBlock(block, currentFeatureId, currentFeatureShardId, sink);
                    block = new BlockAccumulator(featureId, blockId, config.samplesPerBlock, manifest.nSamples, pointSchema);
                }
                block.append(sampleId, flags, traceLen, columnBlobs);
            }
        }

        finalizeBlock(block, currentFeatureId, currentFeatureShardId, sink);
    }

    /**
     * bundle parquet를 읽는 SQL을 만든다.
     *
     * <p>중요한 점은 마지막의 {@code ORDER BY feature_id, sample_id}이다.
     * 이 정렬 순서를 전제로 {@link BlockAccumulator}가 현재 feature/current block 하나만 들고
     * streaming 방식으로 block row를 만들 수 있다.
     */
    private static String buildSelectSql(ArrayBundleManifest manifest, int samplesPerBlock, List<PointColumnSpec> pointSchema) {
        StringBuilder sb = new StringBuilder();
        sb.append("SELECT feature_id, sample_id, flags, trace_len");
        for (PointColumnSpec spec : pointSchema) {
            sb.append(", ").append(blobColumnName(spec.name));
        }
        sb.append(" FROM read_parquet(").append(buildParquetPathListLiteral(manifest) + ")");
        sb.append(" ORDER BY feature_id, sample_id");
        return sb.toString();
    }

    /**
     * point column blob 길이가 trace length와 schema에 맞는지 검증한다.
     *
     * <p>예를 들어 trace_len이 10이고 storage type이 float64면 blob 길이는 80바이트여야 한다.
     * 이 검증이 실패하면 bundle 단계에서 이미 잘못된 데이터가 들어온 것이다.
     */
    private static void validatePointBlobLength(byte[] blob, int traceLen, PointColumnSpec spec) throws SQLException {
        int expectedBytes = ArrayUtils.pointColumnBytes(spec, traceLen);
        int actual = (blob == null) ? 0 : blob.length;
        if (actual != expectedBytes) {
            throw new SQLException(spec.name + "_blob length mismatch: expected " + expectedBytes + " got " + actual);
        }
    }

    /**
     * 누적 중인 block을 최종 {@link ArrayShardRow}로 고정한 뒤 shard sink로 넘긴다.
     *
     * <p>이 함수가 하는 일은 다음과 같다.
     * 1) block 내부 offset 배열을 마감한다.
     * 2) present sample이 하나도 없는 block은 버린다.
     * 3) sample offsets와 point column 바이트를 immutable row 형태로 고정한다.
     * 4) 이 row를 올바른 shard writer로 전달한다.
     */
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

    /**
     * bundle parquet 경로 목록을 DuckDB의 {@code read_parquet([...])} literal 형식으로 만든다.
     */
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

    /**
     * point schema에 특정 컬럼 이름이 있는지 확인한다.
     */
    private static boolean hasColumn(List<PointColumnSpec> pointSchema, String name) {
        for (PointColumnSpec spec : pointSchema) {
            if (spec.name.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /**
     * point column 이름으로부터 bundle parquet 안의 blob 컬럼 이름을 만든다.
     */
    private static String blobColumnName(String name) {
        return name + "_blob";
    }

    /**
     * feature shard partition을 계산할 때 쓰는 중간 통계이다.
     *
     * <p>{@code featureIds[i]}와 {@code estimatedBytes[i]}는 같은 feature를 가리킨다.
     */
    private static final class ArrayFeatureStats {
        final int[] featureIds;
        final long[] estimatedBytes;

        ArrayFeatureStats(int[] featureIds, long[] estimatedBytes) {
            this.featureIds = featureIds;
            this.estimatedBytes = estimatedBytes;
        }
    }

    /**
     * 하나의 {@code (feature_id, block_id)}에 해당하는 block payload를 메모리에서 조립하는 누적기이다.
     *
     * <p>이 객체는 다음 상태를 유지한다.
     * 1) sample별 flags
     * 2) sample별 trace 시작/끝 offset
     * 3) point schema 각 컬럼의 blob byte stream
     *
     * <p>정렬된 bundle row를 sample 순서대로 append하다가, block 경계가 바뀌면
     * 이 객체를 {@link ArrayShardRow}로 고정해서 writer에 넘긴다.
     */
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

        /**
         * sample 하나의 trace를 현재 block에 붙인다.
         *
         * <p>정렬된 입력을 전제로 하기 때문에 이미 지난 sample index가 다시 오면 에러로 본다.
         * 중간에 비어 있는 sample이 있으면 그 위치의 offset만 그대로 이어준다.
         */
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

        /**
         * 아직 한 번도 등장하지 않은 tail sample들의 offset을 채워 block을 마감한다.
         */
        void finish() {
            while (nextRelativeSample < sampleCount) {
                sampleOffsets[nextRelativeSample + 1] = pointCount;
                nextRelativeSample++;
            }
        }

        /**
         * 이 block 안에 실제 present sample이 하나라도 있었는지 반환한다.
         */
        boolean hasPresentRows() {
            return presentCount > 0;
        }
    }

    /**
     * 완성된 block 하나를 shard writer에 넘기기 위한 immutable 전달 객체이다.
     *
     * <p>{@link BlockAccumulator}가 가변 상태로 조립하던 값을
     * 최종 파일 쓰기 직전에 고정한 형태라고 보면 된다.
     */
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

    /**
     * shard별 writer lifecycle을 관리하는 상위 조정자이다.
     *
     * <p>입력 row는 shard 순서가 단조증가한다고 가정한다.
     * 그래서 현재 shard용 writer 하나만 열어 두고, shard가 바뀌는 시점에 이전 writer를 닫고
     * 다음 writer로 넘어가는 방식으로 파일 수를 관리한다.
     *
     * <p>또한 끝날 때 한 번도 row가 쓰이지 않은 shard도 빈 {@code blocks.idx}/{@code blocks.bin}으로
     * materialize해서 manifest의 shard 배열을 완성한다.
     */
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

        /**
         * shard row 하나를 적절한 shard writer로 전달한다.
         */
        void writeRow(int shardId, ArrayShardRow row) throws IOException {
            if (currentWriter == null || shardId != currentShardId) {
                rotateToShard(shardId);
            }
            currentWriter.writeRow(row);
        }

        /**
         * 최종 manifest에 넣을 shard 정보 배열을 반환한다.
         */
        ArrayBinaryShardInfo[] shardInfos() {
            return shardInfos;
        }

        /**
         * 현재 writer를 닫고 다음 shard용 writer로 회전한다.
         *
         * <p>입력 shard 순서가 뒤로 되돌아가는 경우는 build 로직 가정이 깨진 것이므로 에러로 본다.
         */
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

        /**
         * 열려 있는 writer를 마감하고, 아직 materialize되지 않은 빈 shard도 모두 생성한다.
         */
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

    /**
     * shard 하나의 {@code blocks.idx}/{@code blocks.bin} 파일을 실제로 쓰는 저수준 writer이다.
     *
     * <p>writer가 관리하는 핵심 규칙은 다음과 같다.
     * 1) shard 안의 block record 순서는 {@code local_feature_index * blocksPerFeature + blockId}로 고정된다.
     * 2) 없는 block도 빈 record를 써서 slot을 채운다.
     * 3) 실제 payload는 {@code blocks.bin}에, offset/length 메타는 {@code blocks.idx}에 기록한다.
     */
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

        /**
         * shard row 하나를 현재 shard 파일에 기록한다.
         *
         * <p>핵심 단계는 다음과 같다.
         * 1) 이 row가 shard 내부에서 몇 번째 record slot인지 계산한다.
         * 2) 아직 안 채운 앞쪽 slot이 있으면 빈 record로 채운다.
         * 3) point schema 순서대로 column blob을 하나의 payload로 이어붙인다.
         * 4) payload를 {@code blocks.bin}에 쓰고, offset/length 메타를 {@code blocks.idx}에 쓴다.
         */
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

        /**
         * 남은 빈 slot을 채우고 파일 header를 완성한 뒤 shard info를 반환한다.
         */
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

        /**
         * 실제 payload가 없는 빈 block record를 하나 쓴다.
         *
         * <p>reader가 고정된 record index 계산으로 block을 찾을 수 있도록
         * 존재하지 않는 block도 slot 자체는 유지한다.
         */
        private void writeEmptyRecord() throws IOException {
            ArrayBinaryFormat.writeBlockRecord(blocksIndex, 0L, 0L, 0L, ArrayBinaryFormat.CODEC_NONE);
            nextRecordIndex++;
        }

        /**
         * 최종 manifest에 기록할 shard 요약 정보를 만든다.
         */
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

    /**
     * array binary shard build처럼 큰 정렬과 집계를 수행하는 DuckDB 연결에만
     * 국소적으로 안전 설정을 적용한다.
     *
     * <p>이 build 경로는 bundle parquet 전체를 대상으로
     * {@code GROUP BY + COUNT DISTINCT}와
     * {@code ORDER BY feature_id, sample_id}를 수행한다.
     * 따라서 일반 lookup용 연결보다 메모리 압박이 크고,
     * spill 위치와 worker 수를 명시하는 편이 안전하다.
     *
     * <p>적용하는 설정은 다음과 같다.
     * 1) temp_directory: 큰 sort/aggregate가 spill할 디렉터리
     * 2) memory_limit: DuckDB가 쓰려는 작업 메모리 상한
     * 3) threads: 병렬 worker 수 상한
     * 4) preserve_insertion_order=false: 입력 순서 보존 비용 완화
     */
    private static void applyBuildDuckDbSettings(Connection conn, File outDir) throws SQLException, IOException {
        File tempDir = new File(outDir, "_duckdb_tmp");
        if (!tempDir.exists() && !tempDir.mkdirs()) {
            throw new IOException("Failed to create DuckDB temp dir: " + tempDir.getAbsolutePath());
        }
        int threads = Math.max(1, Math.min(BUILD_DUCKDB_THREADS, Runtime.getRuntime().availableProcessors()));
        try (Statement st = conn.createStatement()) {
            st.execute("SET temp_directory = " + DuckDBUtils.quotePath(tempDir.getAbsolutePath()));
            st.execute("SET memory_limit = '" + BUILD_DUCKDB_MEMORY_LIMIT + "'");
            st.execute("SET threads = " + threads);
            st.execute("SET preserve_insertion_order = false");
        }
    }
}
