package scripts;

import fs.io.ArrayBinaryShardReader;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.io.array.ArrayShardManifestIO;
import fs.model.array.ArrayShardManifest;

import java.util.Arrays;

/**
 * Array reader 모드별 조회 시간을 비교하는 벤치마크 엔트리포인트다.
 */
public class BenchmarkArrayReaderModesMain {
    public static void main(String[] args) throws Exception {
        String manifestPath = getArg(args, "--manifest", "data/array_synth_py_binary/array_binary_shard_manifest.json");
        int warmup = Integer.parseInt(getArg(args, "--warmup", "3"));
        int iters = Integer.parseInt(getArg(args, "--iters", "7"));
        int manyFeaturesCount = Integer.parseInt(getArg(args, "--many-features-count", "20"));
        int featureBase = Integer.parseInt(getArg(args, "--feature-base", "0"));
        int singleSampleId = Integer.parseInt(getArg(args, "--single-sample-id", "0"));
        int singleFeatureId = Integer.parseInt(getArg(args, "--single-feature-id", "0"));
        int contiguousCount = Integer.parseInt(getArg(args, "--contiguous-count", "256"));
        int spreadCount = Integer.parseInt(getArg(args, "--spread-count", "256"));
        int spreadStep = Integer.parseInt(getArg(args, "--spread-step", "19"));

        ArrayShardManifest manifest = ArrayShardManifestIO.read(manifestPath);
        ArrayFeatureLocatorIndex locatorIndex = ArrayFeatureLocatorIndex.load(manifest);
        long[] singleSampleIds = new long[]{singleSampleId};
        long[] contiguousSampleIds = buildContiguousSampleIds(contiguousCount);
        long[] spreadSampleIds = buildSpreadSampleIds(spreadCount, spreadStep, manifest.nSamples);

        try (ArrayBinaryShardReader reader = new ArrayBinaryShardReader(manifest)) {
            benchmarkCase("many_features_one_sample", warmup, iters, new RunnerAdapter(reader, featureBase, manyFeaturesCount, singleSampleIds, locatorIndex));
            benchmarkCase("single_feature_contiguous", warmup, iters, new SingleFeatureAdapter(reader, singleFeatureId, contiguousSampleIds, locatorIndex));
            benchmarkCase("single_feature_spread", warmup, iters, new SingleFeatureAdapter(reader, singleFeatureId, spreadSampleIds, locatorIndex));
        }
    }

    private static void benchmarkCase(String label, int warmup, int iters, RunnableCase runnable) throws Exception {
        for (int i = 0; i < warmup; i++) {
            runnable.run();
        }
        long[] times = measure(iters, runnable);
        System.out.println(label
                + "\tbinary_median_ms=" + formatMillis(median(times))
                + "\tbinary_all_ms=" + formatArrayMillis(times));
    }

    private static long[] measure(int iters, RunnableCase runnable) throws Exception {
        long[] times = new long[iters];
        for (int i = 0; i < iters; i++) {
            long start = System.nanoTime();
            runnable.run();
            times[i] = System.nanoTime() - start;
        }
        return times;
    }

    private static long median(long[] values) {
        long[] copy = Arrays.copyOf(values, values.length);
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    private static String formatMillis(long nanos) {
        return String.format("%.3f", nanos / 1_000_000.0);
    }

    private static String formatArrayMillis(long[] values) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(formatMillis(values[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private static long[] buildContiguousSampleIds(int count) {
        long[] out = new long[count];
        for (int i = 0; i < count; i++) {
            out[i] = i;
        }
        return out;
    }

    private static long[] buildSpreadSampleIds(int count, int step, int nSamples) {
        long[] out = new long[count];
        for (int i = 0; i < count; i++) {
            out[i] = ((long) i * (long) step) % (long) nSamples;
        }
        return out;
    }

    private static String getArg(String[] args, String key, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }

    private interface RunnableCase {
        void run() throws Exception;
    }

    private static final class RunnerAdapter implements RunnableCase {
        private final ArrayBinaryShardReader reader;
        private final int featureBase;
        private final int manyFeaturesCount;
        private final long[] sampleIds;
        private final ArrayFeatureLocatorIndex locatorIndex;

        RunnerAdapter(ArrayBinaryShardReader reader, int featureBase, int manyFeaturesCount, long[] sampleIds, ArrayFeatureLocatorIndex locatorIndex) {
            this.reader = reader;
            this.featureBase = featureBase;
            this.manyFeaturesCount = manyFeaturesCount;
            this.sampleIds = sampleIds;
            this.locatorIndex = locatorIndex;
        }

        @Override
        public void run() throws Exception {
            for (int featureId = featureBase; featureId < featureBase + manyFeaturesCount; featureId++) {
                reader.loadFeatureSamples(featureId, sampleIds, locatorIndex);
            }
        }
    }

    private static final class SingleFeatureAdapter implements RunnableCase {
        private final ArrayBinaryShardReader reader;
        private final int featureId;
        private final long[] sampleIds;
        private final ArrayFeatureLocatorIndex locatorIndex;

        SingleFeatureAdapter(ArrayBinaryShardReader reader, int featureId, long[] sampleIds, ArrayFeatureLocatorIndex locatorIndex) {
            this.reader = reader;
            this.featureId = featureId;
            this.sampleIds = sampleIds;
            this.locatorIndex = locatorIndex;
        }

        @Override
        public void run() throws Exception {
            reader.loadFeatureSamples(featureId, sampleIds, locatorIndex);
        }
    }
}
