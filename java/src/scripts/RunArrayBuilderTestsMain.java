package scripts;

import fs.config.ArrayBinaryBuildOptions;
import fs.io.ArrayBinaryShardReader;
import fs.io.ArrayBinaryShards;
import fs.io.ArrayDatasetBuilder;
import fs.io.array.ArrayFeatureIdIndex;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.io.array.ArraySampleIdIndex;
import fs.io.common.ArrayMetadataWriter;
import fs.model.array.ArrayTrace;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * check_array.ipynb의 Builder Test와 같은 흐름을 자바 public API로 검증하는 스크립트다.
 *
 * <p>이 스크립트는
 * 1) sample/feature metadata 생성
 * 2) builder로 synthetic trace 기록
 * 3) 최종 array binary shard build
 * 4) key 기반 reader 조회
 * 5) 원본 trace와 복원 결과 전수검증
 * 순서로 동작한다.
 */
public final class RunArrayBuilderTestsMain {
    private static final int N_SAMPLES = 100;
    private static final int N_FEATURES = 16;
    private static final int FEATURES_PER_SENSOR = 4;
    private static final int N_CH_STEPS = 10;
    private static final String STEP_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    private RunArrayBuilderTestsMain() {
    }

    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_array_builder_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String outDir = new File(root, "array_shards").getAbsolutePath();
        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();

        List<Map<String, Object>> sampleRows = sampleRows();
        List<Map<String, Object>> featureRows = featureRows();
        ArrayBinaryShards.writeSampleMeta(sampleRows, sampleMetaPath);
        ArrayBinaryShards.writeFeatureMeta(featureRows, featureMetaPath);
        List<LinkedHashMap<String, Object>> sampleRowsWithIds = ArrayMetadataWriter.readRows(sampleMetaPath);
        List<LinkedHashMap<String, Object>> featureRowsWithIds = ArrayMetadataWriter.readRows(featureMetaPath);

        ArrayBinaryBuildOptions buildOptions = new ArrayBinaryBuildOptions();
        buildOptions.samplesPerBlock = 16;
        buildOptions.targetShardMb = 32;
        buildOptions.codec = "none";

        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("ch_step", StorageType.UINT32, LogicalType.CATEGORICAL)
        );

        LinkedHashMap<String, ExpectedTrace> answers = new LinkedHashMap<String, ExpectedTrace>();
        Random rng = new Random(10L);
        String manifestPath;

        try (ArrayDatasetBuilder builder = ArrayBinaryShards.newBuilder(
                outDir,
                sampleMetaPath,
                pointSchema,
                featureMetaPath,
                buildOptions)) {
            for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
                for (int featureBase = 0; featureBase < N_FEATURES; featureBase += FEATURES_PER_SENSOR) {
                    int signalLength = 300 + rng.nextInt(200);
                    double amplitude = -1.0 + rng.nextDouble() * 2.0;
                    double frequency = 0.1 + rng.nextDouble() * 9.9;

                    double[] time = buildTime(signalLength);
                    double[] baseSignal = buildBaseSignal(signalLength, amplitude, frequency);
                    String[] chSteps = buildChSteps(signalLength, rng);

                    for (int featureTail = 0; featureTail < FEATURES_PER_SENSOR; featureTail++) {
                        int featureId = featureBase + featureTail;
                        double[] values = buildSignalVariant(baseSignal, time, amplitude, frequency, featureTail);

                        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
                        columns.put("time", time.clone());
                        columns.put("value", values.clone());
                        columns.put("ch_step", chSteps.clone());
                        builder.addTrace((long) sampleId, Integer.valueOf(featureId), null, columns);

                        answers.put(key(sampleId, featureId), new ExpectedTrace(time, values, chSteps));
                    }
                }
            }
            manifestPath = builder.buildShards(false);
        }

        ArrayFeatureLocatorIndex locator = ArrayBinaryShards.loadLocator(ArrayBinaryShards.loadManifest(manifestPath));
        ArrayFeatureIdIndex featureIds = ArrayBinaryShards.loadFeatureIds(ArrayBinaryShards.loadManifest(manifestPath));
        ArraySampleIdIndex sampleIds = ArrayBinaryShards.loadSampleIds(ArrayBinaryShards.loadManifest(manifestPath));

        try (ArrayBinaryShardReader reader = ArrayBinaryShards.open(manifestPath)) {
            for (Map<String, Object> sampleRow : sampleRowsWithIds) {
                for (Map<String, Object> featureRow : featureRowsWithIds) {
                    long sampleId = ((Number) sampleRow.get("sample_id")).longValue();
                    int featureId = ((Number) featureRow.get("feature_id")).intValue();
                    String sampleKey = String.valueOf(sampleRow.get("sample_key"));
                    String featureKey = String.valueOf(featureRow.get("feature_key"));

                    Map<String, ArrayTrace> byKey = reader.loadFeatureSamplesByKeys(
                            featureKey,
                            new String[]{sampleKey},
                            locator,
                            featureIds,
                            sampleIds,
                            true);
                    ArrayTrace trace = byKey.get(sampleKey);
                    require(trace != null, "missing trace for " + sampleKey + " / " + featureKey);
                    require(trace.sampleId == sampleId, "sample id mismatch for " + sampleKey + " / " + featureKey);

                    ExpectedTrace expected = answers.get(key((int) sampleId, featureId));
                    require(expected != null, "missing expected trace for sample=" + sampleId + " feature=" + featureId);
                    assertDoubleArray((double[]) trace.columns.get("time"), expected.time, "time", sampleKey, featureKey);
                    assertDoubleArray((double[]) trace.columns.get("value"), expected.value, "value", sampleKey, featureKey);
                    assertStringArray((String[]) trace.columns.get("ch_step"), expected.chStep, "ch_step", sampleKey, featureKey);
                }
            }
        }

        System.out.println("java array builder tests passed");
    }

    private static List<Map<String, Object>> sampleRows() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        for (int i = 0; i < N_SAMPLES; i++) {
            int lotIdx = i / 24;
            int waferIdx = i % 24;

            String lotId = String.format("LOT%04d", lotIdx + 1);
            String waferId = String.format("WF%02d", waferIdx + 1);
            String eqpId = String.format("EQP%d", (lotIdx % 5) + 1);
            String chamberId = String.format("CHAMBER%d", (waferIdx % 4) + 1);

            out.add(row(
                    "sample_key", lotId + "_" + waferId,
                    "lot_id", lotId,
                    "wafer_id", waferId,
                    "eqp_id", eqpId,
                    "chamber_id", chamberId
            ));
        }
        return out;
    }

    private static List<Map<String, Object>> featureRows() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        for (int i = 0; i < N_FEATURES; i++) {
            int sensorIdx = i / FEATURES_PER_SENSOR;
            int subIdx = i % FEATURES_PER_SENSOR;

            String sensorName = String.format("Sensor%04d", sensorIdx + 1);
            String subName = String.format("Sub%02d", subIdx + 1);
            out.add(row(
                    "feature_key", sensorName + "|" + subName,
                    "sensor_name", sensorName,
                    "sub_name", subName,
                    "sub_idx", subIdx
            ));
        }
        return out;
    }

    private static double[] buildTime(int length) {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = (i + 1) * 0.1;
        }
        return out;
    }

    private static double[] buildBaseSignal(int length, double amplitude, double frequency) {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            double x = i * Math.PI / 50.0;
            out[i] = amplitude * Math.sin(frequency * x);
        }
        return out;
    }

    private static double[] buildSignalVariant(double[] baseSignal, double[] time, double amplitude, double frequency, int order) {
        double[] out = new double[baseSignal.length];
        if (order == 0) {
            System.arraycopy(baseSignal, 0, out, 0, baseSignal.length);
            return out;
        }

        double dx = Math.PI / 50.0;
        for (int i = 0; i < baseSignal.length; i++) {
            double phase = frequency * i * dx;
            switch (order) {
                case 1:
                    out[i] = amplitude * frequency * Math.cos(phase);
                    break;
                case 2:
                    out[i] = -amplitude * frequency * frequency * Math.sin(phase);
                    break;
                case 3:
                    out[i] = -amplitude * frequency * frequency * frequency * Math.cos(phase);
                    break;
                default:
                    throw new IllegalArgumentException("unsupported derivative order: " + order);
            }
        }
        return out;
    }

    private static String[] buildChSteps(int signalLength, Random rng) {
        int[] cuts = new int[N_CH_STEPS + 1];
        cuts[0] = 0;
        cuts[N_CH_STEPS] = signalLength;

        ArrayList<Integer> cutCandidates = new ArrayList<Integer>();
        for (int i = 1; i < signalLength; i++) {
            cutCandidates.add(Integer.valueOf(i));
        }
        java.util.Collections.shuffle(cutCandidates, rng);
        int[] chosen = new int[N_CH_STEPS - 1];
        for (int i = 0; i < chosen.length; i++) {
            chosen[i] = cutCandidates.get(i).intValue();
        }
        Arrays.sort(chosen);
        System.arraycopy(chosen, 0, cuts, 1, chosen.length);

        String[] out = new String[signalLength];
        for (int i = 0; i < N_CH_STEPS; i++) {
            String label = String.valueOf(STEP_LABELS.charAt(i));
            for (int pos = cuts[i]; pos < cuts[i + 1]; pos++) {
                out[pos] = label;
            }
        }
        return out;
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(String.valueOf(kvs[i]), kvs[i + 1]);
        }
        return out;
    }

    private static String key(int sampleId, int featureId) {
        return sampleId + ":" + featureId;
    }

    private static void assertDoubleArray(double[] actual, double[] expected, String column, String sampleKey, String featureKey) {
        require(actual != null, "missing column " + column + " for " + sampleKey + " / " + featureKey);
        require(actual.length == expected.length, "length mismatch for " + column + " at " + sampleKey + " / " + featureKey);
        for (int i = 0; i < expected.length; i++) {
            long actualBits = Double.doubleToLongBits(actual[i]);
            long expectedBits = Double.doubleToLongBits(expected[i]);
            require(actualBits == expectedBits, "value mismatch for " + column + " at " + sampleKey + " / " + featureKey + " index=" + i);
        }
    }

    private static void assertStringArray(String[] actual, String[] expected, String column, String sampleKey, String featureKey) {
        require(actual != null, "missing column " + column + " for " + sampleKey + " / " + featureKey);
        require(actual.length == expected.length, "length mismatch for " + column + " at " + sampleKey + " / " + featureKey);
        for (int i = 0; i < expected.length; i++) {
            require(expected[i].equals(actual[i]), "value mismatch for " + column + " at " + sampleKey + " / " + featureKey + " index=" + i);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            throw new IllegalStateException("failed to delete: " + file.getAbsolutePath());
        }
    }

    private static final class ExpectedTrace {
        final double[] time;
        final double[] value;
        final String[] chStep;

        ExpectedTrace(double[] time, double[] value, String[] chStep) {
            this.time = time.clone();
            this.value = value.clone();
            this.chStep = chStep.clone();
        }
    }
}
