package fs.model.common;

/**
 * Point column이 의미적으로 어떤 타입인지 표현하는 enum이다.
 */
public enum LogicalType {
    CONTINUOUS("continuous"),
    INTEGER("integer"),
    CATEGORICAL("categorical"),
    TIMESTAMP_NS("timestamp_ns"),
    TIMEDELTA_NS("timedelta_ns");

    public final String value;

    LogicalType(String value) {
        this.value = value;
    }

    public static LogicalType fromValue(String value) {
        if (value == null) {
            throw new IllegalArgumentException("logical_type must not be null");
        }
        String normalized = value.trim().toLowerCase();
        for (LogicalType item : values()) {
            if (item.value.equals(normalized)) {
                return item;
            }
        }
        throw new IllegalArgumentException("unsupported point logical type: " + value);
    }
}
