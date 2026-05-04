package fs.io.array;

import fs.io.common.ArrayUtils;

import java.util.Map;

/**
 * Array trace row에 기록하는 flag 비트와 해석 helper를 모아둔 클래스다.
 */
public final class ArrayFeatureFlags {
    public static final byte PRESENT = 0x01;
    public static final byte EMPTY = 0x02;

    private ArrayFeatureFlags() {
    }

    public static byte compute(Map<String, Object> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("columns must not be null");
        }
        int traceLen = 0;
        for (Object values : columns.values()) {
            traceLen = ArrayUtils.pointColumnLength(values);
            break;
        }
        byte flags = PRESENT;
        if (traceLen == 0) {
            flags |= EMPTY;
        }
        return flags;
    }

    public static boolean isPresent(byte flags) {
        return (flags & PRESENT) != 0;
    }

    public static boolean isEmpty(byte flags) {
        return (flags & EMPTY) != 0;
    }
}
