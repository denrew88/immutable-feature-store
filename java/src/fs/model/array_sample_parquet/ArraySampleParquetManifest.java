package fs.model.array_sample_parquet;

import fs.model.common.PointColumnSpec;

import java.util.List;

/**
 * array_sample_parquet v1 manifest model이다.
 */
public class ArraySampleParquetManifest {
    public final int version;
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final int nSamples;
    public final int nFeatures;
    public final String samplePartsPath;
    public final String sampleKeyCol;
    public final String featureKeyCol;
    public final List<PointColumnSpec> pointSchema;
    public final List<ArraySampleParquetPart> parts;

    public ArraySampleParquetManifest(
            int version,
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            int nFeatures,
            String samplePartsPath,
            String sampleKeyCol,
            String featureKeyCol,
            List<PointColumnSpec> pointSchema,
            List<ArraySampleParquetPart> parts) {
        this.version = version;
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.nSamples = nSamples;
        this.nFeatures = nFeatures;
        this.samplePartsPath = samplePartsPath;
        this.sampleKeyCol = sampleKeyCol;
        this.featureKeyCol = featureKeyCol;
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        this.parts = parts;
    }
}
