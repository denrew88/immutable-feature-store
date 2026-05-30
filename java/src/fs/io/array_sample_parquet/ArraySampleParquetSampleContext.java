package fs.io.array_sample_parquet;

import java.util.Map;

/**
 * sample 하나에 속한 trace들을 raw parquet 파일 하나로 묶는 context.
 */
public class ArraySampleParquetSampleContext implements AutoCloseable {
    private final ArraySampleParquetDatasetBuilder builder;
    private final long sampleId;
    public final boolean skipped;
    private boolean closed;
    private boolean aborted;

    ArraySampleParquetSampleContext(ArraySampleParquetDatasetBuilder builder, long sampleId) {
        this(builder, sampleId, false);
    }

    ArraySampleParquetSampleContext(ArraySampleParquetDatasetBuilder builder, long sampleId, boolean skipIfCompleted) {
        this.builder = builder;
        this.sampleId = sampleId;
        this.skipped = this.builder.beginSample(sampleId, skipIfCompleted);
    }

    public void addTrace(Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
        if (skipped) {
            return;
        }
        try {
            builder.addTrace(sampleId, featureId, featureKey, columns);
        } catch (Exception e) {
            aborted = true;
            throw e;
        }
    }

    public void abort() {
        aborted = true;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        closed = true;
        if (!skipped) {
            builder.endSample(aborted);
        }
    }
}
