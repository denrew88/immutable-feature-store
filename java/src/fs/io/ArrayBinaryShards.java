package fs.io;

import fs.config.ArrayBinaryBuildOptions;
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
 * array binary shard 작업의 진입점을 모아 둔 facade이다.
 *
 * <p>manifest 로드, locator/index 준비, direct-ingestion builder 생성처럼
 * 상위 사용자가 바로 호출하는 얇은 API만 제공한다.
 * 실제 payload decode와 shard build는 {@link ArrayBinaryShardReader}, {@link ArrayShardBuilder}가 맡는다.
 */
public final class ArrayBinaryShards {
    private ArrayBinaryShards() {
    }

    /**
     * bundle stage를 읽어 기본 설정으로 최종 array binary shard를 만든다.
     *
     * <p>이 오버로드는 shard 크기와 block 크기 같은 핵심 설정만 {@link ArrayShardConfig}로 받고,
     * codec과 metadata key 컬럼명은 표준 기본값을 사용한다.
     */
    public static String buildFromBundles(String bundleManifestPath, String outDir, ArrayShardConfig config) throws Exception {
        return ArrayShardBuilder.buildFromBundles(bundleManifestPath, outDir, config);
    }

    /**
     * bundle stage를 읽어 최종 array binary shard를 만들되, codec과 metadata key 컬럼명까지 직접 지정한다.
     *
     * <p>첫 번째 오버로드와 최종 결과는 같지만, 이 버전은
     * {@code codec}, {@code sampleKeyCol}, {@code featureKeyCol}까지 호출자가 직접 넘겨
     * manifest 기록 방식과 압축 정책을 더 세밀하게 제어할 수 있다.
     */
    public static String buildFromBundles(
            String bundleManifestPath,
            String outDir,
            ArrayShardConfig config,
            String codec,
            String sampleKeyCol,
            String featureKeyCol) throws Exception {
        return ArrayShardBuilder.buildFromBundles(bundleManifestPath, outDir, config, codec, sampleKeyCol, featureKeyCol);
    }

    /**
     * array binary shard manifest를 메모리 모델로 로드한다.
     */
    public static ArrayShardManifest loadManifest(String manifestPath) throws Exception {
        return ArrayShardManifestIO.read(manifestPath);
    }

    /**
     * manifest 경로를 받아 low-level reader를 연다.
     */
    public static ArrayBinaryShardReader open(String manifestPath) throws Exception {
        return new ArrayBinaryShardReader(loadManifest(manifestPath));
    }

    /**
     * feature locator index를 로드한다.
     */
    public static ArrayFeatureLocatorIndex loadLocator(ArrayShardManifest manifest) throws Exception {
        return ArrayFeatureLocatorIndex.load(manifest);
    }

    /**
     * sample id/key lookup index를 로드한다.
     */
    public static ArraySampleIdIndex loadSampleIds(ArrayShardManifest manifest) throws Exception {
        return ArraySampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
    }

    /**
     * feature id/key lookup index를 로드한다.
     */
    public static ArrayFeatureIdIndex loadFeatureIds(ArrayShardManifest manifest) throws Exception {
        return ArrayFeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
    }

    /**
     * dense sample metadata parquet를 작성한다.
     */
    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeSampleMeta(records, path);
    }

    /**
     * dense feature metadata parquet를 작성한다.
     */
    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeFeatureMeta(records, path);
    }

    /**
     * direct-ingestion builder를 가장 단순한 형태로 만든다.
     *
     * <p>이 버전은 point schema만 받고 build 옵션은 내부 기본값을 사용한다.
     * discovered-feature mode가 기본이며, feature metadata는 나중에 별도로 보강할 수 있다.
     */
    public static ArrayDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema);
    }

    /**
     * resumable array build session을 열거나 같은 stage 디렉터리에서 자동으로 이어받는다.
     */
    public static ArrayDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        return ArrayDatasetBuilder.openSession(outDir, sampleMetaPath, pointSchema);
    }

    /**
     * direct-ingestion builder를 만들되 build 옵션을 직접 지정한다.
     *
     * <p>위 오버로드와 달리 shard 크기, block 크기, key 컬럼명 같은
     * {@link ArrayBinaryBuildOptions}를 함께 넘길 수 있다.
     * 다만 feature metadata 파일은 아직 주지 않으므로, 기본적으로는 discovered-feature mode이다.
     */
    public static ArrayDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema, buildOptions);
    }

    /**
     * build 옵션을 지정해 resumable array build session을 열거나 기존 stage를 이어받는다.
     */
    public static ArrayDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        return ArrayDatasetBuilder.openSession(outDir, sampleMetaPath, pointSchema, buildOptions);
    }

    /**
     * known-feature metadata까지 같이 넘겨 direct-ingestion builder를 만든다.
     *
     * <p>세 오버로드 중 가장 구체적인 버전이다.
     * feature metadata parquet를 바로 주기 때문에 builder는 known-feature mode로 시작하고,
     * {@link ArrayBinaryBuildOptions}로 build 세부 설정도 함께 제어할 수 있다.
     */
    public static ArrayDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema, featureMetaPath, buildOptions);
    }

    /**
     * known-feature metadata와 build 옵션을 같이 주고 resumable array build session을 연다.
     */
    public static ArrayDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        return ArrayDatasetBuilder.openSession(outDir, sampleMetaPath, pointSchema, featureMetaPath, buildOptions);
    }
}
