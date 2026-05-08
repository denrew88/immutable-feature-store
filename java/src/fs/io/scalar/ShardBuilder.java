package fs.io.scalar;

import fs.config.BuildShardConfig;
import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.math.Pearson;
import fs.model.common.SampleMeta;
import fs.model.scalar.ScalarSampleBundleManifest;
import fs.model.scalar.ShardManifest;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;

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
 * scalar sample-major stage 또는 sample bundle stage를 최종 shard dataset으로 변환하는 builder 유틸리티이다.
 *
 * <p>이 클래스는 다음 순서로 동작한다.
 * 1) 전체 feature를 shard별 연속 구간으로 나누는 layout을 계산한다.
 * 2) shard마다 feature-major dense row buffer를 memory-mapped 파일로 만든다.
 * 3) 입력 parquet를 읽어 각 feature x sample 위치에 값을 채운다.
 * 4) 채워진 buffer를 shard parquet, feature locator, selection stats, manifest로 직렬화한다.
 *
 * <p>중요한 점은 입력 형식이 sample-major이든 sample bundle이든, 중간 표현은 항상
 * {@link MappedShard}라는 동일한 dense matrix 구조로 수렴한다는 점이다.
 */
public class ShardBuilder {
    private static final ConcurrentHashMap<String, ReentrantLock> OUT_DIR_LOCKS = new ConcurrentHashMap<String, ReentrantLock>();

    /**
     * shard 하나를 쓰기 전에 잠깐 보관하는 dense feature-major 중간 버퍼이다.
     *
     * <p>행은 feature, 열은 sample에 대응한다.
     * 값 버퍼는 {@code double[row, sample]}를 little-endian 바이트로 저장하고,
     * valid 버퍼는 {@code byte[row, sample]} 형태로 값 존재 여부를 저장한다.
     *
     * <p>빌드 과정은 먼저 이 버퍼를 전부 채운 뒤, 나중에 row 단위로 다시 읽어 parquet row
     * {@code (feature_id, value_len, values_blob, valid_blob)} 로 직렬화한다.
     */
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
     * sample-major parquet 목록을 읽어 최종 scalar shard dataset을 만든다.
     *
     * <p>입력 sample metadata 안에는 sample별 parquet 경로가 들어 있어야 한다.
     * 이 함수는 그 경로들을 순회하면서 값을 읽고, 내부적으로는 {@link SampleMajorRowFiller}를 사용해
     * 공통 shard build 파이프라인으로 넘긴다.
     *
     * <p>즉 이 경로의 입력 stage는 "sample마다 별도 parquet가 있는 sample-major 형식"이다.
     * 바깥 루프가 sample이 되고, 각 sample 파일 안의 {@code (feature_id, value)} row를 읽어
     * dense shard buffer의 현재 sample column을 채운다.
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
     * sample bundle manifest를 읽어 최종 scalar shard dataset을 만든다.
     *
     * <p>sample bundle stage는 이미 {@code (sample_id, feature_id, value)} row를 모아 둔 상태이므로,
     * 이 함수는 bundle parquet들을 순회하는 {@link SampleBundleRowFiller}를 사용해 공통 빌드 경로로 넘긴다.
     *
     * <p>즉 이 경로의 입력 stage는 "여러 sample row를 bundle parquet 몇 개에 묶어 둔 형식"이다.
     * 개별 sample 파일 경로를 다시 따라갈 필요 없이, bundle row 하나를 읽을 때마다
     * sample id와 feature id를 바로 해석해 dense shard buffer 위치를 채운다.
     *
     * <p>두 함수가 최종적으로 하는 일은 같지만, 하나는 sample-major 입력을 위한 경로이고
     * 다른 하나는 builder가 만들어 둔 bundled sample-major 입력을 위한 경로라는 점이 다르다.
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
     * 입력 형식을 추상화한 공통 scalar shard build 파이프라인을 수행한다.
     *
     * <p>이 함수가 실제로 하는 일은 다음과 같다.
     * 1) 출력 디렉터리별 lock을 잡아 같은 위치로 동시에 build하지 못하게 한다.
     * 2) shard 개수와 shard별 feature 구간을 계산한다.
     * 3) shard마다 {@link MappedShard}를 만들고, {@link RowFiller}가 dense matrix를 채운다.
     * 4) 채워진 matrix를 shard parquet 파일들로 내린다.
     * 5) 같은 과정에서 selection용 r2 통계, feature locator, manifest도 같이 기록한다.
     *
     * <p>입력이 sample-major이든 sample bundle이든, 차이는 {@link RowFiller}가 값을 어떻게 읽어 오느냐뿐이고
     * 그 이후 단계는 모두 동일하다.
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

            // shard별로 feature-major dense buffer를 준비한다.
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

            // 입력 parquet를 읽어 각 shard buffer의 (feature row, sample col) 위치를 채운다.
            rowFiller.fill(config, idToIndex, layout, shards);
            for (MappedShard shard : shards) {
                if (shard != null) {
                    shard.flush();
                }
            }

            // shard를 parquet로 내릴 때 같이 계산할 selection 통계 배열을 미리 준비한다.
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

            // 채워 둔 dense buffer를 DuckDB temp table을 거치지 않고 바로 shard parquet로 직렬화한다.
            for (int shardId = 0; shardId < layout.nShards; shardId++) {
                int start = layout.shardStarts[shardId];
                int end = layout.shardEnds[shardId];
                int rows = Math.max(0, end - start);
                String shardPath = new File(shardDir, String.format("shard_%04d.parquet", shardId)).getAbsolutePath();
                if (rows == 0) {
                    writeEmptyShard(shardPath);
                    continue;
                }
                MappedShard shard = shards.get(shardId);
                writeShardParquet(shard, shardPath, targetsByY, r2yByY, nOverlapByY, start);
            }

            String sampleMetaOut = new File(out, "sample_meta.parquet").getAbsolutePath();
            String featureMetaOut = new File(out, "feature_meta.parquet").getAbsolutePath();
            materializeMetadataFile(sourceSampleMetaPath, sampleMetaOut);
            materializeMetadataFile(sourceFeatureMetaFile.getAbsolutePath(), featureMetaOut);

            // feature가 어느 shard와 row offset에 있는지 찾는 lookup table을 기록한다.
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

            // 최종 dataset을 다시 열기 위한 manifest를 마지막에 기록한다.
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

    /**
     * source metadata parquet를 출력 디렉터리로 복사한다.
     *
     * <p>최종 shard dataset이 self-contained directory가 되도록 sample/feature metadata를
     * 출력 디렉터리 안에 다시 materialize한다. 이미 같은 파일이면 복사는 생략한다.
     */
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
     * build 설정에 기본값과 제약 조건을 반영해 정규화한다.
     *
     * <p>특히 shard 개수를 직접 지정하지 않을 때는 target shard size가 반드시 있어야 하므로,
     * 여기서 기본 검증을 먼저 수행한다.
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

    /**
     * feature metadata parquet 경로를 결정한다.
     *
     * <p>명시 경로가 있으면 그것을 사용하고, 없으면 sample metadata 옆의
     * {@code feature_meta.parquet}를 기본 위치로 가정한다.
     */
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

    /**
     * feature metadata에서 dense feature id 배열을 읽는다.
     *
     * <p>이 builder는 feature id가 row 순서와 같은 dense 0..N-1 규칙이라고 가정한다.
     * 그래서 parquet를 읽으면서 각 row의 feature id가 정확히 현재 row index와 일치하는지 검증한다.
     */
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

    /**
     * selection stats를 계산할 y 컬럼 목록을 정리한다.
     *
     * <p>{@code statsYCols}가 명시되면 중복을 제거해 그 목록을 사용하고,
     * 없으면 기본 target 컬럼인 {@code yCol} 하나만 사용한다.
     */
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
     * feature 전체를 shard별 contiguous 구간으로 나누는 layout을 계산한다.
     *
     * <p>이 builder에서 shard는 "연속된 feature row 구간"이다.
     * 그래서 layout 계산 결과는 각 shard가 담당하는 feature 시작/끝 index와 shard 크기로 표현된다.
     *
     * <p>규칙은 두 가지이다.
     * 1) {@code nShards > 0}이면 shard 개수를 고정하고 feature를 균등 분할한다.
     * 2) 아니면 feature 하나당 예상 바이트 수를 기준으로 shard 크기를 추정해 필요한 shard 개수를 계산한다.
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

    /**
     * row가 하나도 없는 빈 shard parquet 파일을 만든다.
     *
     * <p>feature 수가 shard 수보다 적으면 일부 shard는 비어 있을 수 있다.
     * 그래도 locator와 manifest 상 shard 번호 체계를 유지하기 위해, 빈 shard도 실제 parquet 파일로 남긴다.
     */
    private static void writeEmptyShard(String shardPath) throws IOException {
        try (ScalarShardParquetWriter writer = ScalarShardParquetWriter.open(shardPath)) {
            // row를 하나도 쓰지 않고 닫으면 빈 shard parquet만 남는다.
        }
    }

    /**
     * 메모리에 채워 둔 {@link MappedShard} 하나를 최종 shard parquet 파일로 직렬화한다.
     *
     * <p>입력 {@link MappedShard}는 이미 feature-major dense matrix로 채워져 있다.
     * 이 함수는 row를 하나씩 다시 읽어 parquet row
     * {@code (feature_id, value_len, values_blob, valid_blob)} 로 바꾼다.
     *
     * <p>동시에 selection용 통계도 같이 계산한다.
     * 각 feature row를 y 컬럼들과 pairwise 비교해 {@code r2}와 유효 sample 개수를 구하고,
     * 그 값을 전역 결과 배열 {@code r2yByY}, {@code nOverlapByY}에 기록한다.
     *
     * @param shard 이미 값이 채워진 dense shard buffer
     * @param shardPath 최종 shard parquet 출력 경로
     * @param targetsByY y 컬럼별 target 값과 valid mask
     * @param r2yByY feature별 r2 결과를 채울 배열 맵
     * @param nOverlapByY feature별 overlap 개수를 채울 배열 맵
     * @param globalStart 현재 shard의 첫 feature가 전체 feature 순서에서 시작하는 global index
     */
    private static void writeShardParquet(
            MappedShard shard,
            String shardPath,
            Map<String, SampleMeta> targetsByY,
            Map<String, double[]> r2yByY,
            Map<String, int[]> nOverlapByY,
            int globalStart) throws IOException {
        int rowValueBytes = shard.rowValueBytes;
        int rowValidBytes = shard.rowValidBytes;

        try (ScalarShardParquetWriter writer = ScalarShardParquetWriter.open(shardPath)) {
            for (int row = 0; row < shard.rows; row++) {
                byte[] valuesBytes = new byte[rowValueBytes];
                byte[] validBytes = new byte[rowValidBytes];
                ByteBuffer vb = shard.valuesBuf.duplicate();
                vb.position(row * rowValueBytes);
                vb.get(valuesBytes, 0, rowValueBytes);
                ByteBuffer mb = shard.validBuf.duplicate();
                mb.position(row * rowValidBytes);
                mb.get(validBytes, 0, rowValidBytes);

                // parquet로 쓰는 원시 blob을 그대로 사용하되, selection 통계를 위해 한 번 decode한다.
                double[] values = ArrayUtils.decodeDoubleArray(valuesBytes, shard.nSamples);
                for (Map.Entry<String, SampleMeta> entry : targetsByY.entrySet()) {
                    Pearson.PairwiseResult stats = Pearson.pairwiseR2(entry.getValue().y, entry.getValue().yMask, values, validBytes, 0);
                    r2yByY.get(entry.getKey())[globalStart + row] = stats.r2;
                    nOverlapByY.get(entry.getKey())[globalStart + row] = stats.n;
                }

                // shard parquet의 한 row는 feature 하나의 전체 sample 값을 blob 두 개로 담는다.
                writer.writeRow(shard.featureIds[row], shard.nSamples, valuesBytes, validBytes);
            }
        }
    }

    /**
     * feature와 shard 위치의 대응 관계를 담은 lookup parquet를 기록한다.
     *
     * <p>reader는 feature id 하나를 받으면 먼저 이 locator를 통해
     * 1) 어느 shard 파일에 들어 있는지
     * 2) 그 shard 안에서 몇 번째 row인지
     * 를 찾는다. 여기서는 feature가 shard 안에서 contiguous하게 배치된다는 사실을 이용해
     * {@code shard_id}와 {@code offset_in_shard}를 계산한다.
     */
    private static void writeFeatureLocatorParquet(Connection conn, int[] featureIds, int shardSize, String locatorPath) throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_feature_locator (feature_id INTEGER, global_rank INTEGER, shard_id INTEGER, offset_in_shard INTEGER)");
        }
        try (DuckDBAppender appender = ((DuckDBConnection) conn).createAppender(DuckDBConnection.DEFAULT_SCHEMA, "tmp_feature_locator")) {
            for (int i = 0; i < featureIds.length; i++) {
                int shardId = i / shardSize;
                int offsetInShard = i - shardId * shardSize;
                appender.beginRow();
                appender.append(featureIds[i]);
                appender.append(i);
                appender.append(shardId);
                appender.append(offsetInShard);
                appender.endRow();
            }
            appender.flush();
        }
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_feature_locator TO " + DuckDBUtils.quotePath(locatorPath) + " (FORMAT PARQUET)");
            st.execute("DROP TABLE tmp_feature_locator");
        }
    }

    /**
     * selection에 바로 쓸 feature별 통계 parquet를 기록한다.
     *
     * <p>각 row는 feature 하나에 대한 {@code r2y}, {@code n_y_overlap}와
     * shard 위치 정보를 함께 담는다. 나중에 selection 단계는 이 파일만 읽어
     * 후보 feature를 빠르게 고를 수 있다.
     */
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
        try (DuckDBAppender appender = ((DuckDBConnection) conn).createAppender(DuckDBConnection.DEFAULT_SCHEMA, "tmp_selection_stats")) {
            for (int i = 0; i < featureIds.length; i++) {
                int shardId = i / shardSize;
                int offsetInShard = i - shardId * shardSize;
                appender.beginRow();
                appender.append(featureIds[i]);
                appender.append(shardId);
                appender.append(offsetInShard);
                appender.append(r2y[i]);
                appender.append(nYOverlap[i]);
                appender.endRow();
            }
            appender.flush();
        }
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_selection_stats TO " + DuckDBUtils.quotePath(path) + " (FORMAT PARQUET)");
            st.execute("DROP TABLE tmp_selection_stats");
        }
    }

    /**
     * selection stats 파일명을 안전한 경로 문자열로 바꾼다.
     *
     * <p>y 컬럼명에 공백이나 특수문자가 있을 수 있으므로 URL encoding을 적용한 뒤 parquet 파일명으로 사용한다.
     */
    private static String encodeStatsFilename(String yCol) {
        try {
            return URLEncoder.encode(yCol, StandardCharsets.UTF_8.name()).replace("+", "%20");
        } catch (java.io.UnsupportedEncodingException e) {
            throw new IllegalStateException("UTF-8 encoding is unavailable", e);
        }
    }

    /**
     * feature id를 dense row index로 바꾸는 빠른 lookup 맵을 만든다.
     *
     * <p>입력 parquet 안의 feature id를 읽었을 때, 그것이 전체 feature 순서에서 몇 번째인지
     * 즉시 찾아 shard 번호와 shard 내부 row offset을 계산하는 데 사용한다.
     */
    private static HashMapIntInt buildIdToIndex(int[] featureIds) {
        HashMapIntInt out = new HashMapIntInt(Math.max(16, featureIds.length * 2));
        for (int i = 0; i < featureIds.length; i++) {
            out.put(featureIds[i], i);
        }
        return out;
    }

    /**
     * parquet 파일 row 수를 센다.
     *
     * <p>feature metadata의 전체 feature 개수를 구할 때 사용한다.
     */
    private static int countRows(Connection conn, String path) throws SQLException {
        String sql = "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(path) + ")";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getInt(1);
        }
    }

    /**
     * long으로 읽은 feature id를 int 범위인지 검증한 뒤 변환한다.
     */
    private static int toIntFeatureId(long value) {
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("feature_id out of int32 range: " + value);
        }
        return (int) value;
    }

    /**
     * 현재 build 실행만을 위한 고유한 임시 디렉터리를 만든다.
     *
     * <p>같은 outDir 아래에서 build를 여러 번 수행할 수 있으므로, tmp 루트 아래에는
     * 실행별 하위 디렉터리를 따로 만든다.
     */
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

    /**
     * 생성했던 memory-mapped shard들을 best-effort로 닫는다.
     *
     * <p>빌드가 중간에 실패하더라도 temp 파일을 정리할 수 있도록 finally 블록에서 호출한다.
     */
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

    /**
     * MappedByteBuffer unmap을 JVM이 허용하는 범위에서 시도한다.
     *
     * <p>Windows에서는 버퍼가 열린 상태면 backing file 삭제가 실패할 수 있으므로,
     * reflection 기반 cleaner 호출로 file lock을 빨리 풀려고 시도한다.
     */
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

    /**
     * 파일 또는 디렉터리를 재귀적으로 best-effort 삭제한다.
     */
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

    /**
     * 디렉터리가 비어 있으면 삭제한다.
     *
     * <p>build 실행 전용 임시 디렉터리를 정리한 뒤, 상위 tmp 루트도 비었으면 같이 치운다.
     */
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
     * 입력 형식 차이를 추상화한 shard buffer 채우기 전략이다.
     *
     * <p>sample-major parquet와 sample bundle parquet는 읽는 방법은 다르지만,
     * 최종적으로는 {@link MappedShard}의 특정 feature row / sample column 위치를 채운다는 점이 같다.
     * 그 공통 인터페이스가 {@code RowFiller}이다.
     */
    private interface RowFiller {
        void fill(BuildShardConfig config, HashMapIntInt idToIndex, ShardLayout layout, List<MappedShard> shards) throws SQLException;
    }

    /**
     * sample-major parquet 파일들을 순서대로 읽어 shard buffer를 채운다.
     *
     * <p>sample-major 입력은 "sample 하나당 feature row 여러 개" 구조이므로,
     * 바깥 루프는 sample, 안쪽 루프는 해당 sample에 존재하는 feature 값들이 된다.
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
     * sample bundle parquet 파일들을 순서대로 읽어 shard buffer를 채운다.
     *
     * <p>sample bundle 입력은 이미 {@code (sample_id, feature_id, value)} row를 펼쳐 놓은 형태이므로,
     * 각 row를 읽을 때마다 바로 해당 shard row / sample column 위치를 찾을 수 있다.
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

    /**
     * feature 전체를 shard별 contiguous 구간으로 나눈 결과이다.
     */
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

    /**
     * primitive int에서 int로 가는 lookup 용도의 얇은 wrapper이다.
     *
     * <p>성능을 극단적으로 튜닝한 자료구조는 아니고, 코드에서 "feature id에서 dense index로 간다"는 의미를
     * 드러내기 위한 작은 도우미로 사용한다.
     */
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
