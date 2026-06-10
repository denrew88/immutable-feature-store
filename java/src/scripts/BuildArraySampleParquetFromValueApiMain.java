package scripts;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.ArraySampleParquets;
import fs.io.array_sample_parquet.ArraySampleParquetDatasetBuilder;
import fs.io.array_sample_parquet.ArraySampleParquetSampleContext;
import fs.io.common.ArrayMetadataWriter;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.array_sample_parquet.ArraySampleParquetBuildSessionStatus;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Python value API를 호출해서 array_sample_parquet part 전체를 만드는 최소 예제입니다.
 *
 * <p>sample_meta와 feature_meta는 이미 존재한다고 가정합니다. Python API는 sample 하나와 feature id
 * 묶음에 대한 trace들을 반환하고, Java는 그 trace들을 현재 array_sample_parquet raw session에 그대로
 * 기록합니다. 모든 sample이 끝나면 compact 단계에서 최종 sample_parts/trace_index_parts를 만듭니다.</p>
 */
public class BuildArraySampleParquetFromValueApiMain {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        String baseUrl = getArg(args, "--base-url", "http://127.0.0.1:8010");
        String sampleMetaPath = requiredArg(args, "--sample-meta");
        String featureMetaPath = requiredArg(args, "--feature-meta");
        String outDir = requiredArg(args, "--out-dir");
        int featureChunkSize = Integer.parseInt(getArg(args, "--feature-chunk-size", "512"));
        long seed = Long.parseLong(getArg(args, "--seed", "0"));

        int nFeatures = ArrayMetadataWriter.readRows(featureMetaPath).size();
        List<Integer> allFeatureIds = allFeatureIds(nFeatures);
        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL)
        );

        ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
        options.targetPartBytes = Long.parseLong(getArg(args, "--target-part-mb", "128")) * 1024L * 1024L;
        options.maxPartRows = Integer.parseInt(getArg(args, "--max-part-rows", "10000000"));
        options.maxPartSamples = Integer.parseInt(getArg(args, "--max-part-samples", "0"));
        options.compression = getArg(args, "--compression", "zstd");

        String manifestPath;
        try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
                outDir,
                sampleMetaPath,
                pointSchema,
                featureMetaPath,
                options)) {
            ArraySampleParquetBuildSessionStatus status = builder.status();
            for (Long sampleId : status.pendingSampleIds) {
                int traceCount = 0;
                try (ArraySampleParquetSampleContext sample = builder.sample(sampleId.longValue(), true)) {
                    if (!sample.skipped) {
                        for (List<Integer> chunk : chunks(allFeatureIds, featureChunkSize)) {
                            JsonNode response = postJson(url(baseUrl, "/array/traces"),
                                    arrayRequest(sampleMetaPath, featureMetaPath, sampleId.longValue(), chunk, seed));
                            for (JsonNode trace : response.path("traces")) {
                                if (!trace.path("present").asBoolean(false)) {
                                    continue;
                                }
                                int featureId = trace.path("feature_id").asInt();
                                sample.addTrace(Integer.valueOf(featureId), null, columns(trace.path("columns")));
                                traceCount++;
                            }
                        }
                    }
                }
                System.out.println("array sample committed: " + sampleId + " traces=" + traceCount);
            }
            manifestPath = builder.compact();
        }

        System.out.println("manifest=" + manifestPath);
    }

    private static ObjectNode arrayRequest(
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
        root.put("include_missing", false);
        ArrayNode arr = root.putArray("feature_ids");
        for (Integer featureId : featureIds) {
            arr.add(featureId.intValue());
        }
        return root;
    }

    private static LinkedHashMap<String, Object> columns(JsonNode columns) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        out.put("time", doubleArray(columns.path("time")));
        out.put("value", doubleArray(columns.path("value")));
        out.put("ch_step", stringArray(columns.path("ch_step")));
        return out;
    }

    private static double[] doubleArray(JsonNode node) {
        double[] out = new double[node.size()];
        for (int i = 0; i < node.size(); i++) {
            out[i] = node.get(i).asDouble();
        }
        return out;
    }

    private static String[] stringArray(JsonNode node) {
        String[] out = new String[node.size()];
        for (int i = 0; i < node.size(); i++) {
            out[i] = node.get(i).asText();
        }
        return out;
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
