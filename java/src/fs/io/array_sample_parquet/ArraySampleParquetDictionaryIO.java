package fs.io.array_sample_parquet;

import fs.io.common.JsonUtils;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * sample-major Parquet 포맷의 categorical dictionary sidecar를 읽고 쓴다.
 */
public final class ArraySampleParquetDictionaryIO {
    private ArraySampleParquetDictionaryIO() {
    }

    public static void write(String path, String columnName, List<String> labels) throws IOException {
        JsonUtils.writeCategoricalDictionary(path, columnName, labels);
    }

    public static Map<Long, String> read(String path) throws IOException {
        return JsonUtils.readCategoricalDictionary(path);
    }
}
