package fs.io.array_sample_parquet;

import java.util.Map;

/**
 * sample 하나에 속한 trace들을 명시적으로 묶는 context다.
 */
public class ArraySampleParquetSampleContext implements AutoCloseable {
    private final ArraySampleParquetDatasetBuilder builder;
    private final long sampleId;
    private boolean closed;

    ArraySampleParquetSampleContext(ArraySampleParquetDatasetBuilder builder, long sampleId) {
        this.builder = builder;
        this.sampleId = sampleId;
        this.builder.beginSample(sampleId);
    }

    public void addTrace(Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
        builder.addTrace(sampleId, featureId, featureKey, columns);
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        closed = true;
        builder.endSample(false);
    }
}
