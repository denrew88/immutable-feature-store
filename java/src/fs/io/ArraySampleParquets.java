package fs.io;

import fs.io.array_sample_parquet.ArraySampleParquetDatasetBuilder;
import fs.io.array_sample_parquet.ArraySampleParquetManifestIO;
import fs.io.array_sample_parquet.ArraySampleParquetReader;
import fs.io.common.ArrayMetadataWriter;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.array_sample_parquet.ArraySampleParquetManifest;
import fs.model.common.PointColumnSpec;

import java.util.List;
import java.util.Map;

/**
 * array_sample_parquet public facade.
 *
 * <p>{@link #openSession(String, String, List, String, ArraySampleParquetBuildOptions)}는
 * sample별 raw parquet를 먼저 만들고 {@code finish()/compact()}에서 최종 part를 만드는
 * resume-safe builder를 연다.</p>
 */
public final class ArraySampleParquets {
    private ArraySampleParquets() {
    }

    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeSampleMeta(records, path);
    }

    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeFeatureMeta(records, path);
    }

    public static ArraySampleParquetDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            ArraySampleParquetBuildOptions options) throws Exception {
        return ArraySampleParquetDatasetBuilder.openSession(outDir, sampleMetaPath, pointSchema, featureMetaPath, null, options);
    }

    public static ArraySampleParquetDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            List<String> featureKeys,
            ArraySampleParquetBuildOptions options) throws Exception {
        return ArraySampleParquetDatasetBuilder.openSession(outDir, sampleMetaPath, pointSchema, "", featureKeys, options);
    }

    public static ArraySampleParquetManifest loadManifest(String manifestPath) throws Exception {
        return ArraySampleParquetManifestIO.read(manifestPath);
    }

    public static ArraySampleParquetReader open(String manifestPath) throws Exception {
        return new ArraySampleParquetReader(loadManifest(manifestPath));
    }
}
