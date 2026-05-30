package fs.model.array_sample_parquet;

/**
 * sample-major Parquet dataset의 part parquet 하나를 설명한다.
 */
public class ArraySampleParquetPart {
    public final int partId;
    public final String path;
    public final String traceIndexPath;
    public final long firstSampleId;
    public final long lastSampleId;
    public final int sampleCount;
    public final int traceCount;
    public final int rowCount;
    public final long byteSize;
    public final long traceIndexByteSize;

    public ArraySampleParquetPart(
            int partId,
            String path,
            String traceIndexPath,
            long firstSampleId,
            long lastSampleId,
            int sampleCount,
            int traceCount,
            int rowCount,
            long byteSize,
            long traceIndexByteSize) {
        this.partId = partId;
        this.path = path;
        this.traceIndexPath = traceIndexPath;
        this.firstSampleId = firstSampleId;
        this.lastSampleId = lastSampleId;
        this.sampleCount = sampleCount;
        this.traceCount = traceCount;
        this.rowCount = rowCount;
        this.byteSize = byteSize;
        this.traceIndexByteSize = traceIndexByteSize;
    }
}
