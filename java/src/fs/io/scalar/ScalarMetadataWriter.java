package fs.io.scalar;

import fs.io.common.ArrayMetadataWriter;

import java.util.List;
import java.util.Map;

/**
 * scalar metadata 작성 facade다.
 *
 * <p>현재 scalar metadata 포맷은 array와 동일한 dense metadata 규칙을 쓰므로
 * 실제 구현은 {@link ArrayMetadataWriter}를 재사용한다.
 */
public final class ScalarMetadataWriter {
    private ScalarMetadataWriter() {
    }

    /**
     * scalar sample metadata parquet를 작성한다.
     */
    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeSampleMeta(records, path);
    }

    /**
     * scalar feature metadata parquet를 작성한다.
     */
    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeFeatureMeta(records, path);
    }
}
