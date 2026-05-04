package fs.io;

import fs.config.ArrayShardConfig;
import fs.io.array.ArrayFeatureIdIndex;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.io.array.ArraySampleIdIndex;
import fs.io.array.ArrayShardBuilder;
import fs.io.array.ArrayShardManifestIO;
import fs.io.common.ArrayMetadataWriter;
import fs.model.array.ArrayShardManifest;
import fs.model.common.PointColumnSpec;

import java.util.List;
import java.util.Map;

/**
 * array binary shard 포맷을 다루는 자바용 진입점이다.
 *
 * <p>이 클래스는 manifest 로드, locator/index 준비, direct-ingestion builder 생성처럼
 * 외부에서 바로 쓰는 얇은 facade만 모아 둔다. 실제 shard build와 payload decode는
 * {@link ArrayShardBuilder}, {@link ArrayBinaryShardReader}가 맡는다.
 */
public final class ArrayBinaryShards {
    private ArrayBinaryShards() {
    }

    /**
     * sample-major bundle artifact를 읽어 최종 array binary shard dataset을 만든다.
     *
     * @param bundleManifestPath bundle stage manifest 경로
     * @param outDir 최종 artifact 출력 디렉터리
     * @param config shard 크기와 block 크기를 정하는 설정
     * @return 생성된 array shard manifest 경로
     */
    public static String buildFromBundles(String bundleManifestPath, String outDir, ArrayShardConfig config) throws Exception {
        return ArrayShardBuilder.buildFromBundles(bundleManifestPath, outDir, config);
    }

    /**
     * array binary shard manifest를 메모리 모델로 로드한다.
     *
     * @param manifestPath shard manifest JSON 경로
     * @return 파싱된 manifest 객체
     */
    public static ArrayShardManifest loadManifest(String manifestPath) throws Exception {
        return ArrayShardManifestIO.read(manifestPath);
    }

    /**
     * manifest 경로를 받아 low-level reader를 연다.
     *
     * @param manifestPath shard manifest JSON 경로
     * @return 열린 array binary shard reader
     */
    public static ArrayBinaryShardReader open(String manifestPath) throws Exception {
        return new ArrayBinaryShardReader(loadManifest(manifestPath));
    }

    /**
     * feature locator index를 로드한다.
     *
     * @param manifest shard manifest
     * @return feature locator index
     */
    public static ArrayFeatureLocatorIndex loadLocator(ArrayShardManifest manifest) throws Exception {
        return ArrayFeatureLocatorIndex.load(manifest);
    }

    /**
     * sample id/key lookup index를 로드한다.
     *
     * @param manifest shard manifest
     * @return sample id index
     */
    public static ArraySampleIdIndex loadSampleIds(ArrayShardManifest manifest) throws Exception {
        return ArraySampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
    }

    /**
     * feature id/key lookup index를 로드한다.
     *
     * @param manifest shard manifest
     * @return feature id index
     */
    public static ArrayFeatureIdIndex loadFeatureIds(ArrayShardManifest manifest) throws Exception {
        return ArrayFeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
    }

    /**
     * dense sample metadata parquet를 작성한다.
     *
     * @param records metadata row 목록
     * @param path 출력 parquet 경로
     * @return 절대 경로
     */
    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeSampleMeta(records, path);
    }

    /**
     * dense feature metadata parquet를 작성한다.
     *
     * @param records metadata row 목록
     * @param path 출력 parquet 경로
     * @return 절대 경로
     */
    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeFeatureMeta(records, path);
    }

    /**
     * direct-ingestion builder를 기본 설정으로 생성한다.
     *
     * @param outDir 최종 출력 디렉터리
     * @param sampleMetaPath sample metadata parquet 경로
     * @param pointSchema point column schema
     * @return 열린 builder
     */
    public static ArrayDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema);
    }
}
