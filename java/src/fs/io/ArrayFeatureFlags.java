package fs.io;

public final class ArrayFeatureFlags {
    public static final byte PRESENT = 0x01;
    public static final byte EMPTY = 0x02;
    public static final byte HAS_NONFINITE_TIME = 0x04;
    public static final byte HAS_NONFINITE_VALUE = 0x08;

    private ArrayFeatureFlags() {
    }

    public static byte compute(double[] time, double[] value) {
        if (time == null || value == null) {
            throw new IllegalArgumentException("time/value must not be null");
        }
        if (time.length != value.length) {
            throw new IllegalArgumentException("time/value length mismatch: " + time.length + " != " + value.length);
        }
        byte flags = PRESENT;
        if (time.length == 0) {
            return (byte) (flags | EMPTY);
        }
        boolean hasNonfiniteTime = false;
        boolean hasNonfiniteValue = false;
        for (int i = 0; i < time.length; i++) {
            double t = time[i];
            double v = value[i];
            if (Double.isNaN(t) || Double.isInfinite(t)) {
                hasNonfiniteTime = true;
            }
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                hasNonfiniteValue = true;
            }
        }
        if (hasNonfiniteTime) {
            flags |= HAS_NONFINITE_TIME;
        }
        if (hasNonfiniteValue) {
            flags |= HAS_NONFINITE_VALUE;
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
