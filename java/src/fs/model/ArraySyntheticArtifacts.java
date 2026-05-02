package fs.model;

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
