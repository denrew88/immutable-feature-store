import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.common.ArrayMetadataWriter;
import fs.model.scalar.ScalarBuildSessionStatus;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Builds scalar dense-long shards by calling the Python value API while depending
 * only on the packaged scalar-feature-shard Java jar and runtime jars.
 *
 * <p>The example assumes sample_meta and feature_meta already exist. It asks the
 * Python API for one sample and a chunk of feature ids at a time, writes each
 * completed sample into the resumable sample-major stage, and materializes the
 * final dense-long shard manifest at the end.</p>
 */
public class BuildScalarDenseLongFromValueApiWithJarExample {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        String baseUrl = getArg(args, "--base-url", "http://127.0.0.1:8010");
        String sampleMetaPath = requiredArg(args, "--sample-meta");
        String featureMetaPath = requiredArg(args, "--feature-meta");
        String outDir = requiredArg(args, "--out-dir");
        int featureChunkSize = Integer.parseInt(getArg(args, "--feature-chunk-size", "2048"));
        long seed = Long.parseLong(getArg(args, "--seed", "0"));

        int nFeatures = ArrayMetadataWriter.readRows(featureMetaPath).size();
        List<Integer> allFeatureIds = allFeatureIds(nFeatures);

        BuildShardConfig config = new BuildShardConfig();
        config.featureMetaPath = featureMetaPath;
        config.denseLongRowGroupFeatures = Integer.parseInt(getArg(args, "--row-group-features", "128"));
        config.yCol = getArg(args, "--y-col", config.yCol);
        String targetShardMb = getArg(args, "--target-shard-mb", null);
        if (targetShardMb != null) {
            config.targetShardBytes = Long.parseLong(targetShardMb) * 1024L * 1024L;
        }

        String manifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir,
                sampleMetaPath,
                featureMetaPath,
                null,
                config)) {
            ScalarBuildSessionStatus status = builder.status();
            for (Long sampleId : status.pendingSampleIds) {
                LinkedHashMap<Integer, Double> values = new LinkedHashMap<Integer, Double>();
                for (List<Integer> chunk : chunks(allFeatureIds, featureChunkSize)) {
                    JsonNode response = postJson(url(baseUrl, "/scalar/values"),
                            scalarRequest(sampleMetaPath, featureMetaPath, sampleId.longValue(), chunk, seed));
                    for (JsonNode item : response.path("values")) {
                        if (item.path("present").asBoolean(false) && item.hasNonNull("value")) {
                            values.put(
                                    Integer.valueOf(item.path("feature_id").asInt()),
                                    Double.valueOf(item.path("value").asDouble()));
                        }
                    }
                }
                builder.writeSample(sampleId.longValue(), values, true);
                System.out.println("scalar sample committed: " + sampleId + " values=" + values.size());
            }
            manifestPath = builder.buildDenseLongShards(true, null, true);
        }

        System.out.println("manifest=" + manifestPath);
    }

    private static ObjectNode scalarRequest(
            String sampleMetaPath,
            String featureMetaPath,
            long sampleId,
            List<Integer> featureIds,
            long seed) {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("sample_meta_path", sampleMetaPath);
        root.put("feature_meta_path", featureMetaPath);
        root.put("sample_id", sampleId);
        root.put("seed", seed);
        ArrayNode arr = root.putArray("feature_ids");
        for (Integer featureId : featureIds) {
            arr.add(featureId.intValue());
        }
        return root;
    }

    private static JsonNode postJson(String url, ObjectNode request) throws Exception {
        byte[] payload = MAPPER.writeValueAsBytes(request);
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod("POST");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(300000);
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        try (OutputStream out = conn.getOutputStream()) {
            out.write(payload);
        }
        int status = conn.getResponseCode();
        InputStream stream = status >= 200 && status < 300 ? conn.getInputStream() : conn.getErrorStream();
        String body = readAll(stream);
        if (status < 200 || status >= 300) {
            throw new IllegalStateException("HTTP " + status + " from " + url + ": " + body);
        }
        return MAPPER.readTree(body);
    }

    private static String readAll(InputStream stream) throws Exception {
        if (stream == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        return sb.toString();
    }

    private static List<Integer> allFeatureIds(int nFeatures) {
        ArrayList<Integer> out = new ArrayList<Integer>(nFeatures);
        for (int featureId = 0; featureId < nFeatures; featureId++) {
            out.add(Integer.valueOf(featureId));
        }
        return out;
    }

    private static List<List<Integer>> chunks(List<Integer> values, int chunkSize) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("featureChunkSize must be positive");
        }
        ArrayList<List<Integer>> out = new ArrayList<List<Integer>>();
        for (int i = 0; i < values.size(); i += chunkSize) {
            out.add(values.subList(i, Math.min(values.size(), i + chunkSize)));
        }
        return out;
    }

    private static String url(String baseUrl, String path) {
        return baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) + path : baseUrl + path;
    }

    private static String requiredArg(String[] args, String key) {
        String value = getArg(args, key, null);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("missing required argument: " + key);
        }
        File file = key.endsWith("meta") ? new File(value) : null;
        if (file != null && !file.exists()) {
            throw new IllegalArgumentException(key + " not found: " + file.getAbsolutePath());
        }
        return value;
    }

    private static String getArg(String[] args, String key, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
