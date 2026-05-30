package fs.io.common;

import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.util.Arrays;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.LongBuffer;

/**
 * Array binary/scalar shard 경로에서 공통으로 쓰는 dtype, 인코딩, 배열 변환 helper를 모아둔 클래스다.
 */
public class ArrayUtils {
    public static double[] decodeDoubleArray(byte[] bytes, int nSamples) {
        if (bytes == null) {
            return new double[0];
        }
        if (bytes.length != nSamples * 8) {
            throw new IllegalArgumentException("Invalid double array byte length: " + bytes.length);
        }
        double[] out = new double[nSamples];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer db = bb.asDoubleBuffer();
        db.get(out);
        return out;
    }

    public static byte[] encodeDoubleArray(double[] values) {
        if (values == null || values.length == 0) {
            return new byte[0];
        }
        ByteBuffer bb = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : values) {
            bb.putDouble(v);
        }
        return bb.array();
    }

    public static double[] decodeDoubleArray(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            return new double[0];
        }
        if ((bytes.length % 8) != 0) {
            throw new IllegalArgumentException("Invalid double array byte length: " + bytes.length);
        }
        return decodeDoubleArray(bytes, bytes.length / 8);
    }

    public static byte[] encodeLongArray(long[] values) {
        if (values == null || values.length == 0) {
            return new byte[0];
        }
        ByteBuffer bb = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (long v : values) {
            bb.putLong(v);
        }
        return bb.array();
    }

    public static long[] decodeLongArray(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            return new long[0];
        }
        if ((bytes.length % 8) != 0) {
            throw new IllegalArgumentException("Invalid long array byte length: " + bytes.length);
        }
        long[] out = new long[bytes.length / 8];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        LongBuffer lb = bb.asLongBuffer();
        lb.get(out);
        return out;
    }

    public static byte[] encodePointColumn(Object values, PointColumnSpec spec) {
        if (spec == null) {
            throw new IllegalArgumentException("point column spec must not be null");
        }
        switch (spec.storageType) {
            case FLOAT64:
                return encodeDoubleArray(toDoubleArray(values, spec.name));
            case INT32: {
                int[] ints = toIntArray(values, spec.name);
                ByteBuffer bb = ByteBuffer.allocate(ints.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                for (int value : ints) {
                    bb.putInt(value);
                }
                return bb.array();
            }
            case INT64: {
                long[] longs = toLongArray(values, spec.name);
                ByteBuffer bb = ByteBuffer.allocate(longs.length * 8).order(ByteOrder.LITTLE_ENDIAN);
                for (long value : longs) {
                    bb.putLong(value);
                }
                return bb.array();
            }
            case UINT8: {
                long[] longs = toLongArray(values, spec.name);
                ByteBuffer bb = ByteBuffer.allocate(longs.length);
                for (long value : longs) {
                    bb.put((byte) (value & 0xFFL));
                }
                return bb.array();
            }
            case UINT16: {
                long[] longs = toLongArray(values, spec.name);
                ByteBuffer bb = ByteBuffer.allocate(longs.length * 2).order(ByteOrder.LITTLE_ENDIAN);
                for (long value : longs) {
                    bb.putShort((short) (value & 0xFFFFL));
                }
                return bb.array();
            }
            case UINT32: {
                long[] longs = toLongArray(values, spec.name);
                ByteBuffer bb = ByteBuffer.allocate(longs.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                for (long value : longs) {
                    bb.putInt((int) (value & 0xFFFFFFFFL));
                }
                return bb.array();
            }
            case UINT64: {
                long[] longs = toLongArray(values, spec.name);
                ByteBuffer bb = ByteBuffer.allocate(longs.length * 8).order(ByteOrder.LITTLE_ENDIAN);
                for (long value : longs) {
                    bb.putLong(value);
                }
                return bb.array();
            }
            default:
                throw new IllegalArgumentException("unsupported point storage type: " + spec.storageType.value);
        }
    }

    public static Object decodePointColumn(byte[] bytes, long pointCount, PointColumnSpec spec) {
        if (spec == null) {
            throw new IllegalArgumentException("point column spec must not be null");
        }
        int count = toIntCount(pointCount, spec.name);
        switch (spec.storageType) {
            case FLOAT64:
                return decodeDoubleArray(bytes, count);
            case INT32: {
                if ((bytes == null ? 0 : bytes.length) != count * 4) {
                    throw new IllegalArgumentException("Invalid int32 byte length: " + ((bytes == null) ? 0 : bytes.length));
                }
                int[] out = new int[count];
                ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < count; i++) {
                    out[i] = bb.getInt();
                }
                return out;
            }
            case INT64:
                return decodeLongArray(bytes);
            case UINT8: {
                if ((bytes == null ? 0 : bytes.length) != count) {
                    throw new IllegalArgumentException("Invalid uint8 byte length: " + ((bytes == null) ? 0 : bytes.length));
                }
                long[] out = new long[count];
                for (int i = 0; i < count; i++) {
                    out[i] = bytes[i] & 0xFFL;
                }
                return out;
            }
            case UINT16: {
                if ((bytes == null ? 0 : bytes.length) != count * 2) {
                    throw new IllegalArgumentException("Invalid uint16 byte length: " + ((bytes == null) ? 0 : bytes.length));
                }
                long[] out = new long[count];
                ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < count; i++) {
                    out[i] = bb.getShort() & 0xFFFFL;
                }
                return out;
            }
            case UINT32: {
                if ((bytes == null ? 0 : bytes.length) != count * 4) {
                    throw new IllegalArgumentException("Invalid uint32 byte length: " + ((bytes == null) ? 0 : bytes.length));
                }
                long[] out = new long[count];
                ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < count; i++) {
                    out[i] = bb.getInt() & 0xFFFFFFFFL;
                }
                return out;
            }
            case UINT64:
                return decodeLongArray(bytes);
            default:
                throw new IllegalArgumentException("unsupported point storage type: " + spec.storageType.value);
        }
    }

    public static int pointColumnLength(Object values) {
        if (values == null) {
            return 0;
        }
        if (values instanceof double[]) {
            return ((double[]) values).length;
        }
        if (values instanceof int[]) {
            return ((int[]) values).length;
        }
        if (values instanceof long[]) {
            return ((long[]) values).length;
        }
        if (values instanceof String[]) {
            return ((String[]) values).length;
        }
        if (values instanceof Object[]) {
            return ((Object[]) values).length;
        }
        if (values instanceof java.util.List<?>) {
            return ((java.util.List<?>) values).size();
        }
        throw new IllegalArgumentException("unsupported point column array type: " + values.getClass().getName());
    }

    public static Object slicePointColumn(Object values, int start, int end) {
        if (values instanceof double[]) {
            return Arrays.copyOfRange((double[]) values, start, end);
        }
        if (values instanceof int[]) {
            return Arrays.copyOfRange((int[]) values, start, end);
        }
        if (values instanceof long[]) {
            return Arrays.copyOfRange((long[]) values, start, end);
        }
        if (values instanceof String[]) {
            return Arrays.copyOfRange((String[]) values, start, end);
        }
        if (values instanceof Object[]) {
            return Arrays.copyOfRange((Object[]) values, start, end);
        }
        if (values instanceof java.util.List<?>) {
            java.util.List<?> source = (java.util.List<?>) values;
            return source.subList(start, end).toArray(new Object[end - start]);
        }
        throw new IllegalArgumentException("unsupported point column array type: " + values.getClass().getName());
    }

    public static Object emptyPointColumn(PointColumnSpec spec) {
        return emptyPointColumn(spec, false);
    }

    public static Object emptyPointColumn(PointColumnSpec spec, boolean decodeCategorical) {
        if (decodeCategorical && spec.dictionaryPath != null && !spec.dictionaryPath.isEmpty()) {
            return new String[0];
        }
        switch (spec.storageType) {
            case FLOAT64:
                return new double[0];
            case INT32:
                return new int[0];
            case INT64:
            case UINT8:
            case UINT16:
            case UINT32:
            case UINT64:
                return new long[0];
            case STRING:
                return new String[0];
            default:
                throw new IllegalArgumentException("unsupported point storage type: " + spec.storageType.value);
        }
    }

    public static String[] decodeCategoricalLabels(Object values, java.util.Map<Long, String> dictionary) {
        if (values == null) {
            return new String[0];
        }
        long[] codes = toLongArray(values, "categorical");
        String[] out = new String[codes.length];
        for (int i = 0; i < codes.length; i++) {
            long code = codes[i];
            if (code == 0L) {
                out[i] = null;
            } else if (dictionary != null && dictionary.containsKey(code)) {
                out[i] = dictionary.get(code);
            } else {
                out[i] = Long.toUnsignedString(code);
            }
        }
        return out;
    }

    public static double[] toDoubleArray(Object values, String columnName) {
        if (values == null) {
            return new double[0];
        }
        if (values instanceof double[]) {
            return (double[]) values;
        }
        if (values instanceof float[]) {
            float[] source = (float[]) values;
            double[] out = new double[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i];
            }
            return out;
        }
        if (values instanceof int[]) {
            int[] source = (int[]) values;
            double[] out = new double[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i];
            }
            return out;
        }
        if (values instanceof long[]) {
            long[] source = (long[]) values;
            double[] out = new double[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i];
            }
            return out;
        }
        if (values instanceof Number[]) {
            Number[] source = (Number[]) values;
            double[] out = new double[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i].doubleValue();
            }
            return out;
        }
        if (values instanceof java.util.List<?>) {
            java.util.List<?> source = (java.util.List<?>) values;
            double[] out = new double[source.size()];
            for (int i = 0; i < source.size(); i++) {
                Object value = source.get(i);
                if (!(value instanceof Number)) {
                    throw new IllegalArgumentException("point column " + columnName + " must contain numeric values");
                }
                out[i] = ((Number) value).doubleValue();
            }
            return out;
        }
        throw new IllegalArgumentException("unsupported values for column " + columnName + ": " + values.getClass().getName());
    }

    public static int[] toIntArray(Object values, String columnName) {
        if (values == null) {
            return new int[0];
        }
        if (values instanceof int[]) {
            return (int[]) values;
        }
        long[] longs = toLongArray(values, columnName);
        int[] out = new int[longs.length];
        for (int i = 0; i < longs.length; i++) {
            long value = longs[i];
            if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("point column " + columnName + " value out of int32 range: " + value);
            }
            out[i] = (int) value;
        }
        return out;
    }

    public static long[] toLongArray(Object values, String columnName) {
        if (values == null) {
            return new long[0];
        }
        if (values instanceof long[]) {
            return (long[]) values;
        }
        if (values instanceof int[]) {
            int[] source = (int[]) values;
            long[] out = new long[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i];
            }
            return out;
        }
        if (values instanceof Number[]) {
            Number[] source = (Number[]) values;
            long[] out = new long[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i].longValue();
            }
            return out;
        }
        if (values instanceof java.util.List<?>) {
            java.util.List<?> source = (java.util.List<?>) values;
            long[] out = new long[source.size()];
            for (int i = 0; i < source.size(); i++) {
                Object value = source.get(i);
                if (!(value instanceof Number)) {
                    throw new IllegalArgumentException("point column " + columnName + " must contain numeric values");
                }
                out[i] = ((Number) value).longValue();
            }
            return out;
        }
        throw new IllegalArgumentException("unsupported values for column " + columnName + ": " + values.getClass().getName());
    }

    public static String[] toStringArray(Object values, String columnName) {
        if (values == null) {
            return new String[0];
        }
        if (values instanceof String[]) {
            return (String[]) values;
        }
        if (values instanceof Object[]) {
            Object[] source = (Object[]) values;
            String[] out = new String[source.length];
            for (int i = 0; i < source.length; i++) {
                Object value = source[i];
                if (value == null) {
                    throw new IllegalArgumentException("point column " + columnName + " must not contain null string values");
                }
                out[i] = value.toString();
            }
            return out;
        }
        if (values instanceof java.util.List<?>) {
            java.util.List<?> source = (java.util.List<?>) values;
            String[] out = new String[source.size()];
            for (int i = 0; i < source.size(); i++) {
                Object value = source.get(i);
                if (value == null) {
                    throw new IllegalArgumentException("point column " + columnName + " must not contain null string values");
                }
                out[i] = value.toString();
            }
            return out;
        }
        throw new IllegalArgumentException("unsupported string values for column " + columnName + ": " + values.getClass().getName());
    }

    public static int pointColumnBytes(PointColumnSpec spec, int traceLen) {
        return spec.storageType.itemSize * traceLen;
    }

    private static int toIntCount(long pointCount, String columnName) {
        if (pointCount < 0L || pointCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("point_count out of int range for column " + columnName + ": " + pointCount);
        }
        return (int) pointCount;
    }
}
