package fs.model.array;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Sample 하나에 대한 array trace 컬럼 묶음을 나타내는 public 모델이다.
 */
public class ArrayTrace {
    public final long sampleId;
    public final byte flags;
    public final Map<String, Object> columns;

    public ArrayTrace(long sampleId, byte flags, Map<String, Object> columns) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        if (columns != null) {
            out.putAll(columns);
        }
        this.sampleId = sampleId;
        this.flags = flags;
        this.columns = Collections.unmodifiableMap(out);
    }
}
