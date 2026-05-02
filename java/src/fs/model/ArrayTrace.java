package fs.model;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public class ArrayTrace {
    public final long sampleId;
    public final byte flags;
    public final Map<String, Object> columns;
    public final double[] time;
    public final double[] value;

    public ArrayTrace(long sampleId, byte flags, double[] time, double[] value) {
        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
        columns.put("time", (time == null) ? new double[0] : time);
        columns.put("value", (value == null) ? new double[0] : value);
        this.sampleId = sampleId;
        this.flags = flags;
        this.columns = Collections.unmodifiableMap(columns);
        this.time = (time == null) ? new double[0] : time;
        this.value = (value == null) ? new double[0] : value;
    }

    public ArrayTrace(long sampleId, byte flags, Map<String, Object> columns) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        if (columns != null) {
            out.putAll(columns);
        }
        this.sampleId = sampleId;
        this.flags = flags;
        this.columns = Collections.unmodifiableMap(out);
        this.time = extractDoubleColumn(out, "time");
        this.value = extractDoubleColumn(out, "value");
    }

    private static double[] extractDoubleColumn(Map<String, Object> columns, String name) {
        Object value = columns.get(name);
        if (value instanceof double[]) {
            return (double[]) value;
        }
        return new double[0];
    }
}
