package fs.model.array;

/**
 * Array synthetic 생성 결과에서 bundle manifest와 shard manifest 경로를 함께 돌려주는 객체다.
 */
public class ArraySyntheticArtifacts {
    public final String bundleManifestPath;
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final String shardManifestPath;

    public ArraySyntheticArtifacts(
            String bundleManifestPath,
            String sampleMetaPath,
            String featureMetaPath,
            String shardManifestPath) {
        this.bundleManifestPath = bundleManifestPath;
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.shardManifestPath = shardManifestPath;
    }
}
