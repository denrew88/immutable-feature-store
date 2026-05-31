package fs.io;

import fs.config.BuildShardConfig;
import fs.io.scalar.ScalarDenseLongManifestIO;
import fs.io.scalar.ScalarDenseLongShardBuilder;
import fs.io.scalar.ScalarMetadataWriter;
import fs.model.scalar.ScalarDenseLongManifest;

import java.util.List;
import java.util.Map;

/**
 * scalar dense-long shard 작업의 public facade입니다.
 *
 * <p>현재 표준 builder는 sample별 raw parquet stage를 먼저 만들고, 마지막에
 * dense-long parquet shard로 materialize합니다. 순차 ingest가 필요하면
 * {@link ScalarDatasetBuilder#status()}의 pending sample 목록을 앞에서부터
 * 처리하면 되므로 별도 sequential builder는 제공하지 않습니다.</p>
 */
public final class ScalarFeatureShards {
    private ScalarFeatureShards() {
    }

    /** dense-long scalar manifest를 읽습니다. */
    public static ScalarDenseLongManifest loadManifest(String manifestPath) throws Exception {
        return ScalarDenseLongManifestIO.read(manifestPath);
    }

    /** dense-long scalar shard dataset을 엽니다. */
    public static ScalarDenseLongDataset open(String manifestPath) throws Exception {
        return new ScalarDenseLongDataset(manifestPath);
    }

    /** dense-long scalar shard dataset을 엽니다. */
    public static ScalarDenseLongDataset openDenseLong(String manifestPath) throws Exception {
        return new ScalarDenseLongDataset(manifestPath);
    }

    /** sample metadata parquet를 작성합니다. */
    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ScalarMetadataWriter.writeSampleMeta(records, path);
    }

    /** feature metadata parquet를 작성합니다. */
    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ScalarMetadataWriter.writeFeatureMeta(records, path);
    }

    /** sample-file 기반 standard builder를 새로 만듭니다. */
    public static ScalarDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig);
    }

    /** sample-file 기반 standard builder session을 열거나 재개합니다. */
    public static ScalarDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        return ScalarDatasetBuilder.openSession(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig);
    }

    /** scalar sample-major manifest에서 dense-long parquet shard를 만듭니다. */
    public static String buildDenseLongShardsFromSampleMajorManifest(
            String sampleMajorManifestPath,
            String outDir,
            BuildShardConfig buildConfig) throws Exception {
        return ScalarDenseLongShardBuilder.buildFromSampleMajorManifest(sampleMajorManifestPath, outDir, buildConfig);
    }
}
