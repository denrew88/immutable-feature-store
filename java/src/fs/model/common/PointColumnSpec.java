package fs.model.common;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Array point column 하나의 이름, storage type, logical type, dictionary 경로를 정의한다.
 */
public class PointColumnSpec {
    public final String name;
    public final StorageType storageType;
    public final LogicalType logicalType;
    public final String dictionaryPath;

    public PointColumnSpec(String name, StorageType storageType, LogicalType logicalType) {
        this(name, storageType, logicalType, "");
    }

    public PointColumnSpec(String name, StorageType storageType, LogicalType logicalType, String dictionaryPath) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("point column name must not be empty");
        }
        if (storageType == null) {
            throw new IllegalArgumentException("storage_type must not be null");
        }
        if (logicalType == null) {
            throw new IllegalArgumentException("logical_type must not be null");
        }
        validatePair(storageType, logicalType);
        this.name = name;
        this.storageType = storageType;
        this.logicalType = logicalType;
        this.dictionaryPath = (dictionaryPath == null) ? "" : dictionaryPath;
    }

    public PointColumnSpec(String name, String storageType, String logicalType) {
        this(name, StorageType.fromValue(storageType), LogicalType.fromValue(logicalType), "");
    }

    public PointColumnSpec(String name, String storageType, String logicalType, String dictionaryPath) {
        this(name, StorageType.fromValue(storageType), LogicalType.fromValue(logicalType), dictionaryPath);
    }

    public PointColumnSpec withDictionaryPath(String dictionaryPath) {
        return new PointColumnSpec(name, storageType, logicalType, dictionaryPath);
    }

    public static List<PointColumnSpec> normalizeList(List<PointColumnSpec> pointSchema) {
        if (pointSchema == null || pointSchema.isEmpty()) {
            throw new IllegalArgumentException("point_schema must be provided explicitly");
        }
        List<PointColumnSpec> source = pointSchema;
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(source.size());
        HashSet<String> names = new HashSet<String>();
        for (PointColumnSpec spec : source) {
            if (spec == null) {
                throw new IllegalArgumentException("point_schema must not contain null items");
            }
            if (!names.add(spec.name)) {
                throw new IllegalArgumentException("point_schema column names must be unique: " + spec.name);
            }
            out.add(spec);
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("point_schema must not be empty");
        }
        return out;
    }

    private static void validatePair(StorageType storageType, LogicalType logicalType) {
        switch (logicalType) {
            case CONTINUOUS:
                if (storageType != StorageType.FLOAT64) {
                    throw invalidPair(storageType, logicalType, "float64");
                }
                return;
            case INTEGER:
                if (storageType == StorageType.INT32
                        || storageType == StorageType.INT64
                        || storageType == StorageType.UINT32
                        || storageType == StorageType.UINT64) {
                    return;
                }
                throw invalidPair(storageType, logicalType, "int32, int64, uint32, uint64");
            case CATEGORICAL:
                if (storageType != StorageType.UINT32) {
                    throw invalidPair(storageType, logicalType, "uint32");
                }
                return;
            case TIMESTAMP_NS:
            case TIMEDELTA_NS:
                if (storageType != StorageType.INT64) {
                    throw invalidPair(storageType, logicalType, "int64");
                }
                return;
            default:
                throw new IllegalArgumentException("unsupported logical_type: " + logicalType.value);
        }
    }

    private static IllegalArgumentException invalidPair(StorageType storageType, LogicalType logicalType, String allowed) {
        return new IllegalArgumentException(
                "logical_type=" + logicalType.value + " requires storage_type in {" + allowed + "}; got " + storageType.value);
    }
}
