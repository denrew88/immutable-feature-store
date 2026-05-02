package fs.io;

import fs.config.BuildShardConfig;
import fs.config.SelectionConfig;
import fs.model.Candidate;
import fs.model.ShardManifest;
import fs.pipeline.CandidateBuilder;
import fs.pipeline.Selector;

import java.util.List;
import java.util.Map;

public final class ScalarFeatureShards {
    private ScalarFeatureShards() {
    }

    public static ShardManifest loadManifest(String manifestPath) throws Exception {
        return ManifestIO.read(manifestPath);
    }

    public static ScalarShardDataset open(String manifestPath) throws Exception {
        return new ScalarShardDataset(manifestPath);
    }

    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ScalarMetadataWriter.writeSampleMeta(records, path);
    }

    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ScalarMetadataWriter.writeFeatureMeta(records, path);
    }

    public static ScalarDatasetBuilder newBuilder(String outDir, String sampleMetaPath) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath);
    }

    public static ScalarDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig,
            String sampleMajorOutDir) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig, sampleMajorOutDir);
    }

    public static List<Candidate> buildCandidates(String manifestPath, String yCol, SelectionConfig config) throws Exception {
        ShardManifest manifest = ManifestIO.read(manifestPath);
        BuildShardConfig metaConfig = new BuildShardConfig();
        metaConfig.sampleIdCol = "sample_id";
        metaConfig.yCol = (yCol == null || yCol.isEmpty()) ? "y" : yCol;
        return CandidateBuilder.buildCandidatesFromShards(manifest, config, metaConfig.yCol, metaConfig);
    }

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
