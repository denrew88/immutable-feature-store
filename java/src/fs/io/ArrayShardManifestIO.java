package fs.io;

import fs.model.ArrayBinaryShardInfo;
import fs.model.ArrayShardManifest;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class ArrayShardManifestIO {
    public static void write(ArrayShardManifest manifest, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"format\": \"array-binary-shard\",\n");
        sb.append("  \"version\": 2,\n");
        sb.append("  \"endianness\": \"").append(escapeJson(manifest.endianness)).append("\",\n");
        sb.append("  \"id_scheme\": \"dense_row_ids\",\n");
        sb.append("  \"sample_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.sampleMetaPath))).append("\",\n");
        sb.append("  \"feature_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.featureMetaPath))).append("\",\n");
        sb.append("  \"n_samples\": ").append(manifest.nSamples).append(",\n");
        sb.append("  \"n_features\": ").append(manifest.nFeatures).append(",\n");
        sb.append("  \"shard_path\": \"").append(escapeJson(relativeTo(path, manifest.shardPath))).append("\",\n");
        sb.append("  \"n_shards\": ").append(manifest.nShards).append(",\n");
        sb.append("  \"samples_per_block\": ").append(manifest.samplesPerBlock).append(",\n");
        sb.append("  \"blocks_per_feature\": ").append(manifest.blocksPerFeature).append(",\n");
        sb.append("  \"feature_id_dtype\": \"").append(escapeJson(manifest.featureIdType)).append("\",\n");
        sb.append("  \"flags_dtype\": \"").append(escapeJson(manifest.flagsType)).append("\",\n");
        sb.append("  \"offset_dtype\": \"").append(escapeJson(manifest.offsetType)).append("\",\n");
        sb.append("  \"time_dtype\": \"").append(escapeJson(manifest.timeType)).append("\",\n");
        sb.append("  \"value_dtype\": \"").append(escapeJson(manifest.valueType)).append("\",\n");
        sb.append("  \"default_codec\": \"").append(escapeJson(manifest.defaultCodec)).append("\",\n");
        sb.append("  \"sample_key_col\": \"").append(escapeJson(manifest.sampleKeyCol)).append("\",\n");
        sb.append("  \"feature_key_col\": \"").append(escapeJson(manifest.featureKeyCol)).append("\",\n");
        sb.append("  \"shards\": [\n");
        for (int i = 0; i < manifest.shards.length; i++) {
            ArrayBinaryShardInfo shard = manifest.shards[i];
            sb.append("    {\n");
            sb.append("      \"shard_id\": ").append(shard.shardId).append(",\n");
            sb.append("      \"feature_id_start\": ").append(shard.featureIdStart).append(",\n");
            sb.append("      \"feature_id_end\": ").append(shard.featureIdEnd).append(",\n");
            sb.append("      \"feature_count\": ").append(shard.featureCount).append(",\n");
            sb.append("      \"block_count\": ").append(shard.blockCount).append(",\n");
            sb.append("      \"blocks_index_name\": \"").append(escapeJson(shard.blocksIndexName)).append("\",\n");
            sb.append("      \"blocks_data_name\": \"").append(escapeJson(shard.blocksDataName)).append("\"\n");
            sb.append("    }");
            if (i + 1 < manifest.shards.length) {
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(sb.toString());
        }
    }

    @SuppressWarnings("unchecked")
    public static ArrayShardManifest read(String path) throws IOException {
        String json = readAll(path);
        try {
            ScriptEngine engine = new ScriptEngineManager().getEngineByName("javascript");
            if (engine == null) {
                throw new IOException("javascript engine is unavailable");
            }
            Object parsed = engine.eval("Java.asJSONCompatible(" + json + ")");
            Map<String, Object> root = (Map<String, Object>) parsed;
            List<Object> shardList = (List<Object>) root.get("shards");
            ArrayBinaryShardInfo[] shardInfos = new ArrayBinaryShardInfo[(shardList == null) ? 0 : shardList.size()];
            if (shardList != null) {
                for (int i = 0; i < shardList.size(); i++) {
                    Map<String, Object> shard = (Map<String, Object>) shardList.get(i);
                    shardInfos[i] = new ArrayBinaryShardInfo(
                            intValue(shard.get("shard_id")),
                            intValue(shard.get("feature_id_start")),
                            intValue(shard.get("feature_id_end")),
                            intValue(shard.get("feature_count")),
                            intValue(shard.get("block_count")),
                            stringValue(shard.get("blocks_index_name")),
                            stringValue(shard.get("blocks_data_name")));
                }
            }
            return new ArrayShardManifest(
                    resolveAgainst(path, stringValue(root.get("sample_meta_path"))),
                    resolveAgainst(path, stringValue(root.get("feature_meta_path"))),
                    intValue(root.get("n_samples")),
                    intValue(root.get("n_features")),
                    resolveAgainst(path, stringValue(root.get("shard_path"))),
                    intValue(root.get("n_shards")),
                    intValue(root.get("samples_per_block")),
                    intValue(root.get("blocks_per_feature")),
                    stringValue(root.get("feature_id_dtype")),
                    stringValue(root.get("flags_dtype")),
                    stringValue(root.get("offset_dtype")),
                    stringValue(root.get("time_dtype")),
                    stringValue(root.get("value_dtype")),
                    stringValue(root.get("default_codec")),
                    stringValue(root.get("endianness")),
                    defaultString(root.get("sample_key_col"), ArrayBinaryFormat.DEFAULT_SAMPLE_KEY_COL),
                    defaultString(root.get("feature_key_col"), ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL),
                    shardInfos);
        } catch (Exception e) {
            throw new IOException("failed to parse manifest: " + path, e);
        }
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

    private static int intValue(Object value) {
        return ((Number) value).intValue();
    }

    private static String stringValue(Object value) {
        return (value == null) ? "" : value.toString();
    }

    private static String defaultString(Object value, String defaultValue) {
        String s = stringValue(value);
        return s.isEmpty() ? defaultValue : s;
    }

    private static String readAll(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    private static String escapeJson(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
