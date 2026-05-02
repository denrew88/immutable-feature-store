package fs.io;

import java.util.List;
import java.util.Map;

public final class ScalarMetadataWriter {
    private ScalarMetadataWriter() {
    }

    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeSampleMeta(records, path);
    }

    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeFeatureMeta(records, path);
    }
}
