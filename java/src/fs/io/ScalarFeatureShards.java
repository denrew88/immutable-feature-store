package fs.io;

import fs.config.BuildShardConfig;
import fs.config.SelectionConfig;
import fs.io.scalar.DuckDBShardReader;
import fs.io.scalar.ManifestIO;
import fs.io.scalar.ScalarMetadataWriter;
import fs.model.selection.Candidate;
import fs.model.scalar.ShardManifest;
import fs.pipeline.CandidateBuilder;
import fs.pipeline.Selector;

import java.util.List;
import java.util.Map;

/**
 * scalar shard 포맷을 다루는 자바용 facade다.
 *
 * <p>manifest 로드, reader 열기, metadata 작성, direct-ingestion builder 생성,
 * selection helper 호출을 한곳에서 제공한다.
 */
public final class ScalarFeatureShards {
    private ScalarFeatureShards() {
    }

    /**
     * scalar shard manifest를 읽는다.
     */
    public static ShardManifest loadManifest(String manifestPath) throws Exception {
        return ManifestIO.read(manifestPath);
    }

    /**
     * scalar shard dataset facade를 연다.
     */
    public static ScalarShardDataset open(String manifestPath) throws Exception {
        return new ScalarShardDataset(manifestPath);
    }

    /**
     * sample metadata parquet를 작성한다.
     */
    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ScalarMetadataWriter.writeSampleMeta(records, path);
    }

    /**
     * feature metadata parquet를 작성한다.
     */
    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ScalarMetadataWriter.writeFeatureMeta(records, path);
    }

    /**
     * 기본 설정의 scalar direct-ingestion builder를 생성한다.
     */
    public static ScalarDatasetBuilder newBuilder(String outDir, String sampleMetaPath) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath);
    }

    /**
     * 모든 옵션을 지정해 scalar direct-ingestion builder를 생성한다.
     */
    public static ScalarDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig,
            String sampleMajorOutDir) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig, sampleMajorOutDir);
    }

    /**
     * selection 입력으로 쓸 candidate 목록만 생성한다.
     */
    public static List<Candidate> buildCandidates(String manifestPath, String yCol, SelectionConfig config) throws Exception {
        ShardManifest manifest = ManifestIO.read(manifestPath);
        BuildShardConfig metaConfig = new BuildShardConfig();
        metaConfig.sampleIdCol = "sample_id";
        metaConfig.yCol = (yCol == null || yCol.isEmpty()) ? "y" : yCol;
        return CandidateBuilder.buildCandidatesFromShards(manifest, config, metaConfig.yCol, metaConfig);
    }

    /**
     * shard dataset에서 candidate를 만든 뒤 incremental selector를 실행한다.
     */
    public static List<Candidate> selectFeatures(String manifestPath, String yCol, SelectionConfig config) throws Exception {
        ShardManifest manifest = ManifestIO.read(manifestPath);
        BuildShardConfig metaConfig = new BuildShardConfig();
        metaConfig.sampleIdCol = "sample_id";
        metaConfig.yCol = (yCol == null || yCol.isEmpty()) ? "y" : yCol;
        List<Candidate> candidates = CandidateBuilder.buildCandidatesFromShards(manifest, config, metaConfig.yCol, metaConfig);
        try (DuckDBShardReader reader = new DuckDBShardReader(manifest, config.maxGap)) {
            return Selector.selectFeaturesIncremental(candidates, reader, config);
        }
    }
}
