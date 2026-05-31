package fs.io;

import fs.config.BuildShardConfig;
import fs.io.scalar.ScalarDenseLongManifestIO;
import fs.io.scalar.ScalarDenseLongShardBuilder;
import fs.io.scalar.ScalarMetadataWriter;
import fs.model.scalar.ScalarDenseLongManifest;

import java.util.List;
import java.util.Map;

/**
 * dense-long scalar shard 작업을 한곳에서 시작할 수 있게 묶은 Java facade다.
 */
public final class ScalarFeatureShards {
    private ScalarFeatureShards() {
    }

    /**
     * dense-long scalar manifest를 읽는다.
     */
    public static ScalarDenseLongManifest loadManifest(String manifestPath) throws Exception {
        return ScalarDenseLongManifestIO.read(manifestPath);
    }

    /**
     * dense-long scalar shard dataset을 연다.
     */
    public static ScalarDenseLongDataset open(String manifestPath) throws Exception {
        return new ScalarDenseLongDataset(manifestPath);
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
     * 순차 sample-ingestion builder를 만든다. 최종 산출물은 dense-long shard다.
     */
    public static ScalarDatasetBuilder newBuilder(String outDir, String sampleMetaPath) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath);
    }

    /**
     * 순차 sample-ingestion session을 열거나 재개한다. 최종 산출물은 dense-long shard다.
     */
    public static ScalarDatasetBuilder openSession(String outDir, String sampleMetaPath) throws Exception {
        return ScalarDatasetBuilder.openSession(outDir, sampleMetaPath);
    }

    /**
     * 모든 옵션을 지정해 순차 sample-ingestion builder를 만든다.
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
     * 모든 옵션을 지정해 순차 sample-ingestion session을 열거나 재개한다.
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
     * scalar sample-bundle/raw-sample manifest에서 dense-long parquet shard를 만든다.
     */
    public static String buildDenseLongShardsFromSampleBundles(
            String sampleBundleManifestPath,
            String outDir,
            BuildShardConfig buildConfig) throws Exception {
        return ScalarDenseLongShardBuilder.buildFromSampleBundles(sampleBundleManifestPath, outDir, buildConfig);
    }
}
