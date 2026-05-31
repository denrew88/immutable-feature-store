package fs.io;

import fs.config.BuildShardConfig;
import fs.config.SelectionConfig;
import fs.io.scalar.DuckDBShardReader;
import fs.io.scalar.ManifestIO;
import fs.io.scalar.ScalarDenseLongShardBuilder;
import fs.io.scalar.ScalarMetadataWriter;
import fs.model.selection.Candidate;
import fs.model.scalar.ShardManifest;
import fs.pipeline.CandidateBuilder;
import fs.pipeline.Selector;

import java.util.List;
import java.util.Map;

/**
 * scalar shard 작업을 한곳에서 시작할 수 있게 묶어 둔 자바 facade이다.
 *
 * <p>manifest 로드, reader 열기, metadata 작성, direct-ingestion builder 생성,
 * selection helper 호출을 상위 API에서 한 번에 제공한다.
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
     * dense-long scalar shard dataset을 연다.
     */
    public static ScalarDenseLongDataset openDenseLong(String manifestPath) throws Exception {
        return new ScalarDenseLongDataset(manifestPath);
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
     * 기본 설정의 scalar direct-ingestion builder를 만든다.
     */
    public static ScalarDatasetBuilder newBuilder(String outDir, String sampleMetaPath) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath);
    }

    /**
     * resumable scalar build session을 열거나 같은 stage 디렉터리에서 자동으로 이어받는다.
     */
    public static ScalarDatasetBuilder openSession(String outDir, String sampleMetaPath) throws Exception {
        return ScalarDatasetBuilder.openSession(outDir, sampleMetaPath);
    }

    /**
     * sample별 raw parquet를 쓰는 random-order scalar builder를 연다.
     */
    public static ScalarRawDatasetBuilder openRawSession(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        return ScalarRawDatasetBuilder.openSession(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig);
    }

    /**
     * 모든 옵션을 지정해 scalar direct-ingestion builder를 만든다.
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
     * 모든 옵션을 지정해 resumable scalar build session을 열거나 같은 stage 디렉터리에서 이어받는다.
     *
     * <p>새 session을 시작할 수도 있고, 기존 {@code state.json}과 committed bundle 로그가 있으면
     * 마지막 committed sample 다음부터 다시 쓸 수 있는 상태로 복원한다.
     */
    public static ScalarDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig,
            String sampleMajorOutDir) throws Exception {
        return ScalarDatasetBuilder.openSession(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig, sampleMajorOutDir);
    }

    /**
     * scalar sample-bundle/raw-sample manifest에서 dense-long parquet shard를 만든다.
     */
    public static String buildDenseLongShardsFromSampleBundles(
            String sampleBundleManifestPath,
            String outDir,
            BuildShardConfig buildConfig) throws Exception {
        return ScalarDenseLongShardBuilder.buildFromSampleBundles(sampleBundleManifestPath, outDir, buildConfig);
    }

    /**
     * selection 후보 목록만 생성한다.
     *
     * <p>이 함수는 selection stats나 shard를 읽어 y와 관련 있는 feature 후보를 점수순으로 정리하지만,
     * 후보들 사이의 상관관계를 다시 제거하는 최종 선택 단계는 수행하지 않는다.
     * 즉 결과는 "최종 선택 직전의 후보 리스트"이다.
     */
    public static List<Candidate> buildCandidates(String manifestPath, String yCol, SelectionConfig config) throws Exception {
        ShardManifest manifest = ManifestIO.read(manifestPath);
        BuildShardConfig metaConfig = new BuildShardConfig();
        metaConfig.sampleIdCol = "sample_id";
        metaConfig.yCol = (yCol == null || yCol.isEmpty()) ? "y" : yCol;
        return CandidateBuilder.buildCandidatesFromShards(manifest, config, metaConfig.yCol, metaConfig);
    }

    /**
     * shard dataset에서 후보를 만든 뒤 incremental selector까지 실행한다.
     *
     * <p>즉 내부적으로는 먼저 {@link #buildCandidates(String, String, SelectionConfig)}와 같은 단계로
     * 후보를 만든 다음, 그 후보들끼리의 중복성과 상관관계를 다시 계산해 최종 선택 집합만 남긴다.
     * {@code buildCandidates(...)}가 "후보 생성"에 멈춘다면, 이 함수는 "최종 선택"까지 수행한다.
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
