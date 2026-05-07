package fs.synth;

import fs.config.SyntheticConfig;
import fs.model.synthetic.SyntheticData;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Scalar synthetic X/Y 데이터와 기본 메타를 생성하는 helper다.
 */
public class SyntheticGenerator {
    private static final long LATENT_SALT = 0x11a1b2c3d4e5f607L;
    private static final long FEATURE_NOISE_SALT = 0x21a1b2c3d4e5f607L;
    private static final long NOISE_ONLY_FEATURE_SALT = 0x31a1b2c3d4e5f607L;
    private static final long FEATURE_KEEP_SALT = 0x41a1b2c3d4e5f607L;
    private static final long Y_NOISE_SALT = 0x51a1b2c3d4e5f607L;
    private static final long Y_MISSING_SALT = 0x61a1b2c3d4e5f607L;
    private static final long GROUP_MASK_SALT = 0x71a1b2c3d4e5f607L;

    public static SyntheticData generate(SyntheticConfig cfg) {
        Random rng = new Random(cfg.seed);
        int nNoise = (int) Math.round(cfg.nFeatures * cfg.noiseFeatureRatio);
        int nGroupFeatures = cfg.nFeatures - nNoise;

        int[] groupSizes = sampleGroupSizes(rng, cfg.nLatentGroups, nGroupFeatures, cfg.groupSizeMean);
        double[][] latent = new double[cfg.nLatentGroups][cfg.nSamples];
        for (int g = 0; g < cfg.nLatentGroups; g++) {
            for (int i = 0; i < cfg.nSamples; i++) {
                latent[g][i] = rng.nextGaussian();
            }
        }

        int nInformativeGroups = Math.max(1, (int) Math.round(cfg.nLatentGroups * cfg.informativeGroupRatio));
        int[] informativeGroups = sampleDistinct(rng, cfg.nLatentGroups, nInformativeGroups);
        int nLatentForY = Math.min(cfg.nLatentForY, informativeGroups.length);
        int[] yGroups = sampleDistinctFrom(rng, informativeGroups, nLatentForY);
        double[] weights = new double[nLatentForY];
        for (int i = 0; i < nLatentForY; i++) {
            weights[i] = 1.0 + 0.2 * rng.nextGaussian();
        }

        double[] y = new double[cfg.nSamples];
        for (int i = 0; i < nLatentForY; i++) {
            int g = yGroups[i];
            double w = weights[i];
            for (int s = 0; s < cfg.nSamples; s++) {
                y[s] += w * latent[g][s];
            }
        }
        for (int s = 0; s < cfg.nSamples; s++) {
            y[s] += cfg.noiseScale * rng.nextGaussian();
        }

        if (cfg.yMissingRate > 0.0) {
            for (int s = 0; s < cfg.nSamples; s++) {
                if (rng.nextDouble() < cfg.yMissingRate) {
                    y[s] = Double.NaN;
                }
            }
        }

        double[][] X = new double[cfg.nFeatures][cfg.nSamples];
        byte[][] M = new byte[cfg.nFeatures][cfg.nSamples];
        for (int f = 0; f < cfg.nFeatures; f++) {
            for (int s = 0; s < cfg.nSamples; s++) {
                M[f][s] = 1;
            }
        }

        int sparseTargetValid = (cfg.sparseTargetValid != null) ? cfg.sparseTargetValid : Math.max(2, (int) (0.05 * cfg.nSamples));
        int sparseCount = (int) Math.round(cfg.nFeatures * cfg.sparseFeatureRatio);
        int[] sparseFeatures = sampleDistinct(rng, cfg.nFeatures, sparseCount);
        boolean[] isSparse = new boolean[cfg.nFeatures];
        for (int f : sparseFeatures) {
            isSparse[f] = true;
        }

        int featIdx = 0;
        for (int g = 0; g < groupSizes.length; g++) {
            boolean[] groupMask = null;
            if (rng.nextDouble() < cfg.groupMissingShareProb) {
                groupMask = new boolean[cfg.nSamples];
                double baseMr = clamp(cfg.missingRate, 0.0, 0.95);
                for (int s = 0; s < cfg.nSamples; s++) {
                    groupMask[s] = rng.nextDouble() > baseMr;
                }
            }
            for (int k = 0; k < groupSizes[g]; k++) {
                double sign = (rng.nextDouble() < cfg.negCorrRatio) ? -1.0 : 1.0;
                double strength = cfg.redundantStrength * (0.8 + 0.4 * rng.nextDouble());
                double[] x = new double[cfg.nSamples];
                for (int s = 0; s < cfg.nSamples; s++) {
                    double v = sign * strength * latent[g][s] + cfg.noiseScale * rng.nextGaussian();
                    x[s] = v;
                }
                double mr = clamp(cfg.missingRate + rng.nextGaussian() * cfg.missingRateVariability, 0.0, 0.95);
                if (isSparse[featIdx]) {
                    mr = 1.0 - ((double) sparseTargetValid / cfg.nSamples);
                    mr = clamp(mr, 0.0, 0.98);
                }
                for (int s = 0; s < cfg.nSamples; s++) {
                    boolean keep = rng.nextDouble() > mr;
                    if (groupMask != null) {
                        keep = keep && groupMask[s];
                    }
                    if (!keep) {
                        M[featIdx][s] = 0;
                        x[s] = Double.NaN;
                    }
                }
                X[featIdx] = x;
                featIdx++;
            }
        }

        for (int i = 0; i < nNoise; i++) {
            double[] x = new double[cfg.nSamples];
            for (int s = 0; s < cfg.nSamples; s++) {
                x[s] = cfg.noiseScale * rng.nextGaussian();
            }
            double mr = clamp(cfg.missingRate + rng.nextGaussian() * cfg.missingRateVariability, 0.0, 0.95);
            if (isSparse[featIdx]) {
                mr = 1.0 - ((double) sparseTargetValid / cfg.nSamples);
                mr = clamp(mr, 0.0, 0.98);
            }
            for (int s = 0; s < cfg.nSamples; s++) {
                boolean keep = rng.nextDouble() > mr;
                if (!keep) {
                    M[featIdx][s] = 0;
                    x[s] = Double.NaN;
                }
            }
            X[featIdx] = x;
            featIdx++;
        }

        int[] featureIds = new int[cfg.nFeatures];
        for (int i = 0; i < cfg.nFeatures; i++) {
            featureIds[i] = i;
        }
        return new SyntheticData(X, M, y, featureIds);
    }

    /**
     * 전체 X/M 행렬을 메모리에 만들지 않고 sample 하나씩 synthetic scalar 값을 생성하는
     * iterator-style source를 준비한다.
     *
     * <p>큰 feature 수에서 builder session에 바로 흘려 넣고 싶을 때 사용하는 경로다.
     * latent, y, feature별 생성 파라미터만 메모리에 유지하고, sample 값 맵은 요청 시점마다
     * 하나씩 계산해서 돌려준다.
     */
    public static StreamingSource stream(SyntheticConfig cfg) {
        return new StreamingSource(cfg);
    }

    /**
     * synthetic sample 하나를 나타내는 전송 객체다.
     *
     * <p>{@code values}는 해당 sample에서 present한 feature만 담는다. key는 dense
     * {@code feature_id}이고 value는 scalar 값이다.
     */
    public static final class SyntheticSample {
        public final int sampleId;
        public final double y;
        public final Map<Integer, Double> values;

        SyntheticSample(int sampleId, double y, Map<Integer, Double> values) {
            this.sampleId = sampleId;
            this.y = y;
            this.values = values;
        }
    }

    /**
     * sample 단위 synthetic 값을 순차적으로 생성하는 source다.
     *
     * <p>resume-safe builder 예제처럼 큰 데이터셋을 한 sample씩 생성해서 바로 ingest하고,
     * X/M 전체 행렬은 만들지 않으려는 용도에 맞춰져 있다.
     */
    public static final class StreamingSource {
        private final SyntheticConfig cfg;
        private final double[][] latent;
        private final double[] y;
        private final int[] featureGroupIds;
        private final double[] featureSigns;
        private final double[] featureStrengths;
        private final double[] featureMissingRates;
        private final boolean[] groupHasSharedMask;
        private final boolean[][] groupKeepMask;

        StreamingSource(SyntheticConfig cfg) {
            this.cfg = copyConfig(cfg);
            Random rng = new Random(cfg.seed);
            int nNoise = (int) Math.round(cfg.nFeatures * cfg.noiseFeatureRatio);
            int nGroupFeatures = cfg.nFeatures - nNoise;

            int[] groupSizes = sampleGroupSizes(rng, cfg.nLatentGroups, nGroupFeatures, cfg.groupSizeMean);
            this.latent = new double[cfg.nLatentGroups][cfg.nSamples];
            for (int g = 0; g < cfg.nLatentGroups; g++) {
                for (int s = 0; s < cfg.nSamples; s++) {
                    latent[g][s] = gaussianFromSeed(cfg.seed, LATENT_SALT, g, s);
                }
            }

            int nInformativeGroups = Math.max(1, (int) Math.round(cfg.nLatentGroups * cfg.informativeGroupRatio));
            int[] informativeGroups = sampleDistinct(rng, cfg.nLatentGroups, nInformativeGroups);
            int nLatentForY = Math.min(cfg.nLatentForY, informativeGroups.length);
            int[] yGroups = sampleDistinctFrom(rng, informativeGroups, nLatentForY);
            double[] weights = new double[nLatentForY];
            for (int i = 0; i < nLatentForY; i++) {
                weights[i] = 1.0 + 0.2 * rng.nextGaussian();
            }

            this.y = new double[cfg.nSamples];
            for (int s = 0; s < cfg.nSamples; s++) {
                double value = 0.0;
                for (int i = 0; i < nLatentForY; i++) {
                    value += weights[i] * latent[yGroups[i]][s];
                }
                value += cfg.noiseScale * gaussianFromSeed(cfg.seed, Y_NOISE_SALT, s, 0);
                if (cfg.yMissingRate > 0.0 && uniformFromSeed(cfg.seed, Y_MISSING_SALT, s, 0) < cfg.yMissingRate) {
                    value = Double.NaN;
                }
                y[s] = value;
            }

            int sparseTargetValid = (cfg.sparseTargetValid != null)
                    ? cfg.sparseTargetValid.intValue()
                    : Math.max(2, (int) (0.05 * cfg.nSamples));
            int sparseCount = (int) Math.round(cfg.nFeatures * cfg.sparseFeatureRatio);
            int[] sparseFeatures = sampleDistinct(rng, cfg.nFeatures, sparseCount);
            boolean[] isSparse = new boolean[cfg.nFeatures];
            for (int f : sparseFeatures) {
                isSparse[f] = true;
            }

            this.groupHasSharedMask = new boolean[cfg.nLatentGroups];
            this.groupKeepMask = new boolean[cfg.nLatentGroups][cfg.nSamples];
            for (int g = 0; g < cfg.nLatentGroups; g++) {
                groupHasSharedMask[g] = rng.nextDouble() < cfg.groupMissingShareProb;
                if (groupHasSharedMask[g]) {
                    double baseMr = clamp(cfg.missingRate, 0.0, 0.95);
                    for (int s = 0; s < cfg.nSamples; s++) {
                        groupKeepMask[g][s] = uniformFromSeed(cfg.seed, GROUP_MASK_SALT, g, s) > baseMr;
                    }
                }
            }

            this.featureGroupIds = new int[cfg.nFeatures];
            this.featureSigns = new double[cfg.nFeatures];
            this.featureStrengths = new double[cfg.nFeatures];
            this.featureMissingRates = new double[cfg.nFeatures];

            int featureId = 0;
            for (int g = 0; g < groupSizes.length; g++) {
                for (int k = 0; k < groupSizes[g]; k++) {
                    featureGroupIds[featureId] = g;
                    featureSigns[featureId] = (rng.nextDouble() < cfg.negCorrRatio) ? -1.0 : 1.0;
                    featureStrengths[featureId] = cfg.redundantStrength * (0.8 + 0.4 * rng.nextDouble());
                    featureMissingRates[featureId] = featureMissingRate(cfg, isSparse[featureId], sparseTargetValid, rng);
                    featureId++;
                }
            }
            while (featureId < cfg.nFeatures) {
                featureGroupIds[featureId] = -1;
                featureSigns[featureId] = 1.0;
                featureStrengths[featureId] = 0.0;
                featureMissingRates[featureId] = featureMissingRate(cfg, isSparse[featureId], sparseTargetValid, rng);
                featureId++;
            }
        }

        /**
         * sample metadata를 만들 때 사용할 y 값을 돌려준다.
         */
        public double yValue(int sampleId) {
            validateSampleId(sampleId);
            return y[sampleId];
        }

        /**
         * 지정한 구간의 sample을 순차적으로 생성하는 iterable을 돌려준다.
         *
         * <p>resume 시에는 {@code startSampleId}를 builder status의
         * {@code nextExpectedSampleId}에 맞춰 넣으면 된다.
         */
        public Iterable<SyntheticSample> samples(final int startSampleId, final int endExclusiveSampleId) {
            if (startSampleId < 0 || startSampleId > cfg.nSamples) {
                throw new IllegalArgumentException("startSampleId out of range: " + startSampleId);
            }
            if (endExclusiveSampleId < startSampleId || endExclusiveSampleId > cfg.nSamples) {
                throw new IllegalArgumentException("endExclusiveSampleId out of range: " + endExclusiveSampleId);
            }
            return new Iterable<SyntheticSample>() {
                @Override
                public Iterator<SyntheticSample> iterator() {
                    return new Iterator<SyntheticSample>() {
                        private int nextSampleId = startSampleId;

                        @Override
                        public boolean hasNext() {
                            return nextSampleId < endExclusiveSampleId;
                        }

                        @Override
                        public SyntheticSample next() {
                            SyntheticSample sample = sample(nextSampleId);
                            nextSampleId += 1;
                            return sample;
                        }

                        @Override
                        public void remove() {
                            throw new UnsupportedOperationException("remove");
                        }
                    };
                }
            };
        }

        /**
         * sample 하나를 즉시 계산해서 돌려준다.
         */
        public SyntheticSample sample(int sampleId) {
            validateSampleId(sampleId);
            LinkedHashMap<Integer, Double> values = new LinkedHashMap<Integer, Double>();
            for (int featureId = 0; featureId < cfg.nFeatures; featureId++) {
                if (!isPresent(featureId, sampleId)) {
                    continue;
                }
                values.put(Integer.valueOf(featureId), Double.valueOf(featureValue(featureId, sampleId)));
            }
            return new SyntheticSample(sampleId, y[sampleId], values);
        }

        private boolean isPresent(int featureId, int sampleId) {
            boolean keep = uniformFromSeed(cfg.seed, FEATURE_KEEP_SALT, featureId, sampleId) > featureMissingRates[featureId];
            int groupId = featureGroupIds[featureId];
            if (groupId >= 0 && groupHasSharedMask[groupId]) {
                keep = keep && groupKeepMask[groupId][sampleId];
            }
            return keep;
        }

        private double featureValue(int featureId, int sampleId) {
            int groupId = featureGroupIds[featureId];
            if (groupId >= 0) {
                return featureSigns[featureId] * featureStrengths[featureId] * latent[groupId][sampleId]
                        + cfg.noiseScale * gaussianFromSeed(cfg.seed, FEATURE_NOISE_SALT, featureId, sampleId);
            }
            return cfg.noiseScale * gaussianFromSeed(cfg.seed, NOISE_ONLY_FEATURE_SALT, featureId, sampleId);
        }

        private void validateSampleId(int sampleId) {
            if (sampleId < 0 || sampleId >= cfg.nSamples) {
                throw new IllegalArgumentException("sampleId out of range: " + sampleId);
            }
        }
    }

    private static int[] sampleGroupSizes(Random rng, int nGroups, int total, double mean) {
        int[] sizes = new int[nGroups];
        int sum = 0;
        for (int i = 0; i < nGroups; i++) {
            int v = Math.max(1, samplePoisson(rng, mean));
            sizes[i] = v;
            sum += v;
        }
        int diff = total - sum;
        while (diff != 0) {
            int idx = rng.nextInt(nGroups);
            if (diff > 0) {
                sizes[idx]++;
                diff--;
            } else {
                if (sizes[idx] > 1) {
                    sizes[idx]--;
                    diff++;
                }
            }
        }
        return sizes;
    }

    private static int samplePoisson(Random rng, double lambda) {
        double L = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > L);
        return k - 1;
    }

    private static int[] sampleDistinct(Random rng, int n, int k) {
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            pool.add(i);
        }
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            int idx = rng.nextInt(pool.size());
            out[i] = pool.remove(idx);
        }
        return out;
    }

    private static int[] sampleDistinctFrom(Random rng, int[] pool, int k) {
        List<Integer> list = new ArrayList<>();
        for (int v : pool) {
            list.add(v);
        }
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            int idx = rng.nextInt(list.size());
            out[i] = list.remove(idx);
        }
        return out;
    }

    private static double clamp(double v, double lo, double hi) {
        if (v < lo) {
            return lo;
        }
        if (v > hi) {
            return hi;
        }
        return v;
    }

    private static double featureMissingRate(SyntheticConfig cfg, boolean sparse, int sparseTargetValid, Random rng) {
        double mr = clamp(cfg.missingRate + rng.nextGaussian() * cfg.missingRateVariability, 0.0, 0.95);
        if (sparse) {
            mr = 1.0 - ((double) sparseTargetValid / cfg.nSamples);
            mr = clamp(mr, 0.0, 0.98);
        }
        return mr;
    }

    private static SyntheticConfig copyConfig(SyntheticConfig src) {
        SyntheticConfig out = new SyntheticConfig();
        out.nSamples = src.nSamples;
        out.nFeatures = src.nFeatures;
        out.nLatentGroups = src.nLatentGroups;
        out.groupSizeMean = src.groupSizeMean;
        out.informativeGroupRatio = src.informativeGroupRatio;
        out.nLatentForY = src.nLatentForY;
        out.noiseScale = src.noiseScale;
        out.redundantStrength = src.redundantStrength;
        out.noiseFeatureRatio = src.noiseFeatureRatio;
        out.negCorrRatio = src.negCorrRatio;
        out.missingRate = src.missingRate;
        out.missingRateVariability = src.missingRateVariability;
        out.sparseFeatureRatio = src.sparseFeatureRatio;
        out.sparseTargetValid = src.sparseTargetValid;
        out.groupMissingShareProb = src.groupMissingShareProb;
        out.yMissingRate = src.yMissingRate;
        out.seed = src.seed;
        return out;
    }

    private static double uniformFromSeed(long seed, long salt, int a, int b) {
        long mixed = mix64(seed ^ salt ^ (((long) a) << 32) ^ (b & 0xffffffffL));
        long mantissa = (mixed >>> 11) & ((1L << 53) - 1L);
        return mantissa / (double) (1L << 53);
    }

    private static double gaussianFromSeed(long seed, long salt, int a, int b) {
        double u1 = uniformFromSeed(seed, salt, a, b);
        double u2 = uniformFromSeed(seed, salt ^ 0x9e3779b97f4a7c15L, a, b);
        u1 = Math.max(u1, 1e-12);
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }
}
