package fs.io;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.LongBuffer;

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
}
