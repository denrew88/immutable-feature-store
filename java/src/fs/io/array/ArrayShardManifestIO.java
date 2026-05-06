package fs.io.array;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.array.ArrayBinaryShardInfo;
import fs.model.array.ArrayShardManifest;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Array binary shard manifest JSON을 읽고 쓰는 helper다.
 */
public class ArrayShardManifestIO {
    public static void write(ArrayShardManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", "array-binary-shard");
        root.put("version", manifest.version);
        root.put("endianness", manifest.endianness);
        root.put("id_scheme", "dense_row_ids");
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));
        root.put("n_samples", manifest.nSamples);
        root.put("n_features", manifest.nFeatures);
        root.put("shard_path", relativeTo(path, manifest.shardPath));
        root.put("n_shards", manifest.nShards);
        root.put("samples_per_block", manifest.samplesPerBlock);
        root.put("blocks_per_feature", manifest.blocksPerFeature);
        root.put("feature_id_dtype", manifest.featureIdType);
        root.put("flags_dtype", manifest.flagsType);
        root.put("offset_dtype", manifest.offsetType);
        root.put("default_codec", manifest.defaultCodec);
        root.put("sample_key_col", manifest.sampleKeyCol);
        root.put("feature_key_col", manifest.featureKeyCol);

        ArrayNode schema = root.putArray("point_schema");
        for (PointColumnSpec spec : manifest.pointSchema) {
            ObjectNode item = schema.addObject();
            item.put("name", spec.name);
            item.put("storage_type", spec.storageType.value);
            item.put("logical_type", spec.logicalType.value);
            if (spec.dictionaryPath != null && !spec.dictionaryPath.isEmpty()) {
                item.put("dictionary_path", relativeTo(path, spec.dictionaryPath));
            }
        }

        ArrayNode shards = root.putArray("shards");
        for (ArrayBinaryShardInfo shard : manifest.shards) {
            ObjectNode item = shards.addObject();
            item.put("shard_id", shard.shardId);
            item.put("feature_id_start", shard.featureIdStart);
            item.put("feature_id_end", shard.featureIdEnd);
            item.put("feature_count", shard.featureCount);
            item.put("block_count", shard.blockCount);
            item.put("blocks_index_name", shard.blocksIndexName);
            item.put("blocks_data_name", shard.blocksDataName);
        }

        JsonUtils.writeJson(path, root);
    }

    public static ArrayShardManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        int version = intOrZero(root, "version");
        if (version != ArrayBinaryFormat.FILE_VERSION) {
            throw new IOException("unsupported binary shard manifest version: " + version);
        }
        JsonNode shardList = root.get("shards");
        ArrayBinaryShardInfo[] shardInfos = parseShardInfos(shardList);
        return new ArrayShardManifest(
                version,
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                intOrZero(root, "n_samples"),
                intOrZero(root, "n_features"),
                resolveAgainst(path, textOrEmpty(root, "shard_path")),
                intOrZero(root, "n_shards"),
                intOrZero(root, "samples_per_block"),
                intOrZero(root, "blocks_per_feature"),
                textOrEmpty(root, "feature_id_dtype"),
                textOrEmpty(root, "flags_dtype"),
                textOrEmpty(root, "offset_dtype"),
                textOrEmpty(root, "default_codec"),
                textOrEmpty(root, "endianness"),
                defaultText(root, "sample_key_col", ArrayBinaryFormat.DEFAULT_SAMPLE_KEY_COL),
                defaultText(root, "feature_key_col", ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL),
                shardInfos,
                parsePointSchema(path, root.get("point_schema")));
    }

    private static ArrayBinaryShardInfo[] parseShardInfos(JsonNode raw) {
        if (raw == null || !raw.isArray()) {
            return new ArrayBinaryShardInfo[0];
        }
        ArrayBinaryShardInfo[] out = new ArrayBinaryShardInfo[raw.size()];
        for (int i = 0; i < raw.size(); i++) {
            JsonNode shard = raw.get(i);
            out[i] = new ArrayBinaryShardInfo(
                    intOrZero(shard, "shard_id"),
                    intOrZero(shard, "feature_id_start"),
                    intOrZero(shard, "feature_id_end"),
                    intOrZero(shard, "feature_count"),
                    intOrZero(shard, "block_count"),
                    textOrEmpty(shard, "blocks_index_name"),
                    textOrEmpty(shard, "blocks_data_name"));
        }
        return out;
    }

    private static List<PointColumnSpec> parsePointSchema(String manifestPath, JsonNode raw) {
        if (raw == null || !raw.isArray() || raw.size() == 0) {
            throw new IllegalArgumentException("array binary shard manifest must include point_schema");
        }
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(raw.size());
        for (JsonNode item : raw) {
            out.add(new PointColumnSpec(
                    textOrEmpty(item, "name"),
                    StorageType.fromValue(textOrEmpty(item, "storage_type")),
                    LogicalType.fromValue(defaultText(item, "logical_type", LogicalType.CONTINUOUS.value)),
                    resolveAgainst(manifestPath, textOrEmpty(item, "dictionary_path"))));
        }
        return PointColumnSpec.normalizeList(out);
    }

    private static String relativeTo(String manifestPath, String targetPath) {
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

    private static String resolveAgainst(String manifestPath, String storedPath) {
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

    private static int intOrZero(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? 0 : child.asInt();
    }

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static String defaultText(JsonNode node, String fieldName, String defaultValue) {
        String value = textOrEmpty(node, fieldName);
        return value.isEmpty() ? defaultValue : value;
    }
}
