package fs.model.common;

/**
 * Point column이 바이너리로 저장될 때의 물리 dtype을 표현하는 enum이다.
 */
public enum StorageType {
    FLOAT64("float64", 8),
    INT32("int32", 4),
    INT64("int64", 8),
    STRING("string", 16),
    UINT8("uint8", 1),
    UINT16("uint16", 2),
    UINT32("uint32", 4),
    UINT64("uint64", 8);

    public final String value;
    public final int itemSize;

    StorageType(String value, int itemSize) {
        this.value = value;
        this.itemSize = itemSize;
    }

    public static StorageType fromValue(String value) {
        if (value == null) {
            throw new IllegalArgumentException("storage_type must not be null");
        }
        String normalized = value.trim().toLowerCase();
        for (StorageType item : values()) {
            if (item.value.equals(normalized)) {
                return item;
            }
        }
        throw new IllegalArgumentException("unsupported point storage type: " + value);
    }
}
