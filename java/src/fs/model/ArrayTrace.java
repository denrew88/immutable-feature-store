package fs.model;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

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
