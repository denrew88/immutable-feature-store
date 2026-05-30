package fs.io.array_sample_parquet;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.array_sample_parquet.ArraySampleParquetManifest;
import fs.model.array_sample_parquet.ArraySampleParquetPart;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * array_sample_parquet v1 manifest JSON을 읽고 쓰는 helper다.
 */
public final class ArraySampleParquetManifestIO {
    public static final String FORMAT = "array-sample-parquet";
    public static final int VERSION = 1;

    private ArraySampleParquetManifestIO() {
    }

    public static void write(ArraySampleParquetManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", FORMAT);
        root.put("version", VERSION);
        root.put("id_scheme", "dense_row_ids");
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));
        root.put("n_samples", manifest.nSamples);
        root.put("n_features", manifest.nFeatures);
        root.put("sample_parts_path", relativeTo(path, manifest.samplePartsPath));
        root.put("trace_index_parts_path", relativeTo(path, manifest.traceIndexPartsPath));
        root.put("sample_key_col", manifest.sampleKeyCol);
        root.put("feature_key_col", manifest.featureKeyCol);

        ArrayNode schema = root.putArray("point_schema");
        for (PointColumnSpec spec : manifest.pointSchema) {
            ObjectNode item = schema.addObject();
            item.put("name", spec.name);
            item.put("storage_type", spec.storageType.value);
            item.put("logical_type", spec.logicalType.value);
        }

        ArrayNode parts = root.putArray("parts");
        for (ArraySampleParquetPart part : manifest.parts) {
            ObjectNode item = parts.addObject();
            item.put("part_id", part.partId);
            item.put("path", relativeTo(path, part.path));
            item.put("trace_index_path", relativeTo(path, part.traceIndexPath));
            item.put("first_sample_id", part.firstSampleId);
            item.put("last_sample_id", part.lastSampleId);
            item.put("sample_count", part.sampleCount);
            item.put("trace_count", part.traceCount);
            item.put("row_count", part.rowCount);
            item.put("byte_size", part.byteSize);
            item.put("trace_index_byte_size", part.traceIndexByteSize);
        }
        JsonUtils.writeJson(path, root);
    }

    public static ArraySampleParquetManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        if (!FORMAT.equals(textOrEmpty(root, "format"))) {
            throw new IOException("unsupported array sample parquet manifest format: " + textOrEmpty(root, "format"));
        }
        int version = intOrZero(root, "version");
        if (version != VERSION) {
            throw new IOException("unsupported array sample parquet manifest version: " + version);
        }
        return new ArraySampleParquetManifest(
                version,
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                intOrZero(root, "n_samples"),
                intOrZero(root, "n_features"),
                resolveAgainst(path, textOrEmpty(root, "sample_parts_path")),
                resolveAgainst(path, textOrEmpty(root, "trace_index_parts_path")),
                defaultText(root, "sample_key_col", "sample_key"),
                defaultText(root, "feature_key_col", "feature_key"),
                parsePointSchema(path, root.get("point_schema")),
                parseParts(path, root.get("parts")));
    }

    private static List<PointColumnSpec> parsePointSchema(String manifestPath, JsonNode raw) {
        if (raw == null || !raw.isArray() || raw.size() == 0) {
            throw new IllegalArgumentException("array sample parquet manifest must include point_schema");
        }
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(raw.size());
        for (JsonNode item : raw) {
            out.add(new PointColumnSpec(
                    textOrEmpty(item, "name"),
                    StorageType.fromValue(textOrEmpty(item, "storage_type")),
                    LogicalType.fromValue(defaultText(item, "logical_type", LogicalType.CONTINUOUS.value))));
        }
        return PointColumnSpec.normalizeList(out);
    }

    private static List<ArraySampleParquetPart> parseParts(String manifestPath, JsonNode raw) {
        ArrayList<ArraySampleParquetPart> out = new ArrayList<ArraySampleParquetPart>();
        if (raw == null || !raw.isArray()) {
            return out;
        }
        for (JsonNode item : raw) {
            out.add(new ArraySampleParquetPart(
                    intOrZero(item, "part_id"),
                    resolveAgainst(manifestPath, textOrEmpty(item, "path")),
                    resolveAgainst(manifestPath, textOrEmpty(item, "trace_index_path")),
                    longOrZero(item, "first_sample_id"),
                    longOrZero(item, "last_sample_id"),
                    intOrZero(item, "sample_count"),
                    intOrZero(item, "trace_count"),
                    intOrZero(item, "row_count"),
                    longOrZero(item, "byte_size"),
                    longOrZero(item, "trace_index_byte_size")));
        }
        return out;
    }

    static String relativeTo(String manifestPath, String targetPath) {
        if (targetPath == null || targetPath.isEmpty()) {
            return "";
        }
        File target = new File(targetPath);
        if (!target.isAbsolute()) {
            return targetPath.replace("\\", "/");
        }
        File manifestDir = new File(manifestPath).getAbsoluteFile().getParentFile();
        return manifestDir.toPath().relativize(target.getAbsoluteFile().toPath()).toString().replace("\\", "/");
    }

    static String resolveAgainst(String manifestPath, String storedPath) {
        if (storedPath == null || storedPath.isEmpty()) {
            return "";
        }
        File path = new File(storedPath);
        if (path.isAbsolute()) {
            return path.getAbsolutePath();
        }
        File manifestDir = new File(manifestPath).getAbsoluteFile().getParentFile();
        return new File(manifestDir, storedPath).getAbsolutePath();
    }

    static int intOrZero(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? 0 : child.asInt();
    }

    static long longOrZero(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? 0L : child.asLong();
    }

    static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    static String defaultText(JsonNode node, String fieldName, String defaultValue) {
        String value = textOrEmpty(node, fieldName);
        return value.isEmpty() ? defaultValue : value;
    }
}
