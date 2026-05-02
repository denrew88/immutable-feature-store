package fs.synth;

import fs.config.SyntheticConfig;
import fs.model.SyntheticData;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SyntheticGenerator {
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
}
