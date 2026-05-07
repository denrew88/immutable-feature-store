package scripts;

import fs.config.ArrayBinaryBuildOptions;
import fs.io.ArrayBinaryShards;
import fs.io.ArrayDatasetBuilder;
import fs.io.common.ArrayMetadataWriter;
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
 * `check_array.ipynb`의 Builder Test와 비슷한 흐름을 자바 public API로 실행하는
 * synthetic 데이터 생성 스크립트다.
 *
 * <p>이 스크립트는 다음 순서로 동작한다.
 * 1) sample/feature metadata 생성
 * 2) builder로 synthetic trace 기록
 * 3) 최종 array binary shard build
 */
public final class GenerateArrayShardTestsMain {
    private static final int N_SAMPLES = 100;
    private static final int N_FEATURES = 16;
    private static final int FEATURES_PER_SENSOR = 4;
    private static final int N_CH_STEPS = 10;
    private static final String STEP_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_array_builder_test_large");

        String outDir = new File(root, "array_shards").getAbsolutePath();
        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();

        List<Map<String, Object>> sampleRows = sampleRows();
        List<Map<String, Object>> featureRows = featureRows();
        ArrayBinaryShards.writeSampleMeta(sampleRows, sampleMetaPath);
        ArrayBinaryShards.writeFeatureMeta(featureRows, featureMetaPath);
        ArrayMetadataWriter.readRows(sampleMetaPath);
        ArrayMetadataWriter.readRows(featureMetaPath);

        ArrayBinaryBuildOptions buildOptions = new ArrayBinaryBuildOptions();
        buildOptions.samplesPerBlock = 16;
        buildOptions.targetShardMb = 32;
        buildOptions.codec = "none";

        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("ch_step", StorageType.UINT32, LogicalType.CATEGORICAL)
        );

        Random rng = new Random(10L);

        try (ArrayDatasetBuilder builder = ArrayBinaryShards.openSession(
                outDir,
                sampleMetaPath,
                pointSchema,
                featureMetaPath,
                buildOptions)) {
            for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
                try (ArrayDatasetBuilder.ArraySampleContext sample = builder.sample((long) sampleId)) {
                    for (int featureBase = 0; featureBase < N_FEATURES; featureBase += FEATURES_PER_SENSOR) {
                        int signalLength = 300 + rng.nextInt(200);
                        double amplitude = -1.0 + rng.nextDouble() * 2.0;
                        double frequency = 0.1 + rng.nextDouble() * 9.9;

                        double[] time = buildTime(signalLength);
                        double[] baseSignal = buildBaseSignal(signalLength, amplitude, frequency);
                        String[] chSteps = buildChSteps(signalLength, rng);

                        for (int featureTail = 0; featureTail < FEATURES_PER_SENSOR; featureTail++) {
                            int featureId = featureBase + featureTail;
                            double[] values = buildSignalVariant(baseSignal, amplitude, frequency, featureTail);

                            LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
                            columns.put("time", time.clone());
                            columns.put("value", values.clone());
                            columns.put("ch_step", chSteps.clone());
                            sample.addTrace(Integer.valueOf(featureId), null, columns);
                        }
                    }
                }
            }
            String manifestPath = builder.buildShards(false);
            System.out.println(manifestPath);
        }
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

    private static double[] buildSignalVariant(double[] baseSignal, double amplitude, double frequency, int order) {
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
}
