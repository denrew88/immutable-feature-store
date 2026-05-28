package fs.model.array_sample_parquet;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * sample-major Parquet에서 읽은 trace 하나다.
 */
public class ArraySampleParquetTrace {
    public final long sampleId;
    public final String sampleKey;
    public final int featureId;
    public final String featureKey;
    public final int traceLen;
    public final boolean present;
    public final Map<String, Object> columns;

    public ArraySampleParquetTrace(
            long sampleId,
            String sampleKey,
            int featureId,
            String featureKey,
            int traceLen,
            boolean present,
            Map<String, Object> columns) {
        this.sampleId = sampleId;
        this.sampleKey = sampleKey;
        this.featureId = featureId;
        this.featureKey = featureKey;
        this.traceLen = traceLen;
        this.present = present;
        this.columns = new LinkedHashMap<String, Object>(columns);
    }
}
