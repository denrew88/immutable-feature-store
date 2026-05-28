package fs.io.array_sample_parquet;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * categorical point column의 label을 uint32 code로 매핑한다.
 *
 * <p>code 0은 missing/unknown 예약값이다. 실제 label은 1부터 배정한다.
 */
final class CategoricalRegistry {
    private final LinkedHashMap<String, Long> labelToCode = new LinkedHashMap<String, Long>();
    private final ArrayList<String> labels = new ArrayList<String>();

    long[] encode(Object values, String columnName) {
        if (values == null) {
            return new long[0];
        }
        Object[] source = toObjectArray(values, columnName);
        long[] out = new long[source.length];
        for (int i = 0; i < source.length; i++) {
            Object value = source[i];
            if (value == null) {
                out[i] = 0L;
                continue;
            }
            String label = value.toString();
            Long code = labelToCode.get(label);
            if (code == null) {
                code = Long.valueOf(labels.size() + 1L);
                labelToCode.put(label, code);
                labels.add(label);
            }
            out[i] = code.longValue();
        }
        return out;
    }

    List<String> labels() {
        return new ArrayList<String>(labels);
    }

    void loadLabels(List<String> storedLabels) {
        labelToCode.clear();
        labels.clear();
        if (storedLabels == null) {
            return;
        }
        for (String label : storedLabels) {
            String text = String.valueOf(label);
            if (!labelToCode.containsKey(text)) {
                labelToCode.put(text, Long.valueOf(labels.size() + 1L));
                labels.add(text);
            }
        }
    }

    private static Object[] toObjectArray(Object values, String columnName) {
        if (values instanceof Object[]) {
            return (Object[]) values;
        }
        if (values instanceof List<?>) {
            List<?> list = (List<?>) values;
            return list.toArray(new Object[list.size()]);
        }
        throw new IllegalArgumentException("categorical point column " + columnName + " must be Object[] or List");
    }
}
