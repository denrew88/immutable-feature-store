package fs.synth;

import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.config.ArraySyntheticConfig;
import fs.io.ArraySampleBundleWriter;
import fs.io.ArrayShardBuilder;
import fs.io.DuckDBUtils;
import fs.model.ArraySyntheticArtifacts;

import java.io.File;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ArraySyntheticGenerator {
    public static ArraySyntheticArtifacts generate(
            String bundleOutDir,
            String sampleMetaPath,
            ArraySyntheticConfig cfg,
            ArrayBundleConfig bundleCfg,
            String shardOutDir,
            ArrayShardConfig shardCfg) throws Exception {
        ArraySyntheticConfig synthCfg = (cfg == null) ? new ArraySyntheticConfig() : cfg;
        ArrayBundleConfig outBundleCfg = (bundleCfg == null) ? new ArrayBundleConfig() : bundleCfg;
        ArrayShardConfig outShardCfg = (shardCfg == null) ? new ArrayShardConfig() : shardCfg;

        Random rng = new Random(synthCfg.seed);
        int nNoise = (int) Math.round(synthCfg.nFeatures * synthCfg.noiseFeatureRatio);
        int nGroupFeatures = Math.max(0, synthCfg.nFeatures - nNoise);
        int nGroups = Math.max(1, Math.min(synthCfg.nLatentGroups, Math.max(synthCfg.nFeatures, 1)));
        int nGroupFeatureGroups = (nGroupFeatures > 0) ? Math.min(nGroups, nGroupFeatures) : 0;
        int[] groupSizes = (nGroupFeatureGroups > 0)
                ? sampleGroupSizes(rng, nGroupFeatureGroups, nGroupFeatures, synthCfg.groupSizeMean)
                : new int[0];

        double[] groupFreqs = new double[nGroups];
        double[] groupPhaseShift = new double[nGroups];
        double[] groupTimeScale = new double[nGroups];
        for (int g = 0; g < nGroups; g++) {
            groupFreqs[g] = uniform(rng, 0.6, 2.4);
            groupPhaseShift[g] = uniform(rng, -Math.PI, Math.PI);
            groupTimeScale[g] = uniform(rng, 0.85, 1.2);
        }

        double[][] sampleAmp = new double[nGroups][synthCfg.nSamples];
        double[][] samplePhase = new double[nGroups][synthCfg.nSamples];
        double[][] sampleBaseline = new double[nGroups][synthCfg.nSamples];
        for (int g = 0; g < nGroups; g++) {
            for (int s = 0; s < synthCfg.nSamples; s++) {
                sampleAmp[g][s] = 1.0 + 0.35 * rng.nextGaussian();
                samplePhase[g][s] = 0.45 * rng.nextGaussian();
                sampleBaseline[g][s] = 0.2 * rng.nextGaussian();
            }
        }

        int nInformativeGroups = Math.max(1, (int) Math.round(nGroups * synthCfg.informativeGroupRatio));
        int[] informativeGroups = sampleDistinct(rng, nGroups, nInformativeGroups);
        int nLatentForY = Math.min(synthCfg.nLatentForY, informativeGroups.length);
        int[] yGroups = sampleDistinctFrom(rng, informativeGroups, nLatentForY);
        double[] yWeights = new double[nLatentForY];
        for (int i = 0; i < nLatentForY; i++) {
            yWeights[i] = 1.0 + 0.25 * rng.nextGaussian();
        }
        double[] y = new double[synthCfg.nSamples];
        for (int i = 0; i < nLatentForY; i++) {
            int groupId = yGroups[i];
            double weight = yWeights[i];
            for (int s = 0; s < synthCfg.nSamples; s++) {
                y[s] += weight * sampleAmp[groupId][s];
                y[s] += 0.15 * weight * sampleBaseline[groupId][s];
            }
        }
        for (int s = 0; s < synthCfg.nSamples; s++) {
            y[s] += 0.3 * rng.nextGaussian();
        }

        int[] featureGroup = new int[synthCfg.nFeatures];
        String[] featureKind = new String[synthCfg.nFeatures];
        double[] featureSign = new double[synthCfg.nFeatures];
        double[] featureScale = new double[synthCfg.nFeatures];
        double[] featureLocalFreq = new double[synthCfg.nFeatures];
        double[] featurePhaseOffset = new double[synthCfg.nFeatures];
        for (int f = 0; f < synthCfg.nFeatures; f++) {
            featureGroup[f] = -1;
            featureKind[f] = "noise";
            featureSign[f] = (rng.nextBoolean() ? 1.0 : -1.0);
            featureScale[f] = uniform(rng, 0.9, synthCfg.redundantStrength);
            featureLocalFreq[f] = uniform(rng, 0.75, 1.35);
            featurePhaseOffset[f] = uniform(rng, -0.8, 0.8);
        }
        int featIdx = 0;
        for (int g = 0; g < groupSizes.length; g++) {
            for (int i = 0; i < groupSizes[g] && featIdx < synthCfg.nFeatures; i++) {
                featureGroup[featIdx] = g;
                featureKind[featIdx] = "group";
                featIdx++;
            }
        }

        long[] sampleIds = new long[synthCfg.nSamples];
        for (int s = 0; s < synthCfg.nSamples; s++) {
            sampleIds[s] = s;
        }

        ensureParentDir(sampleMetaPath);
        String featureMetaPath = new File(bundleOutDir, "array_feature_meta.parquet").getAbsolutePath();
        try (ArraySampleBundleWriter writer = new ArraySampleBundleWriter(bundleOutDir, sampleMetaPath, featureMetaPath, synthCfg.nSamples, outBundleCfg)) {
            for (int sampleId = 0; sampleId < synthCfg.nSamples; sampleId++) {
                for (int featureId = 0; featureId < synthCfg.nFeatures; featureId++) {
                    if (rng.nextDouble() < synthCfg.missingFeatureRate) {
                        continue;
                    }
                    if (rng.nextDouble() < synthCfg.emptyTraceRate) {
                        writer.appendTrace(sampleId, featureId, new double[0], new double[0]);
                        continue;
                    }

                    int traceLen = synthCfg.minTraceLen + rng.nextInt(synthCfg.maxTraceLen - synthCfg.minTraceLen + 1);
                    double duration = synthCfg.timeDuration * uniform(rng, 0.8, 1.2);
                    double[] time = buildTimeAxis(rng, traceLen, duration, synthCfg.timeJitter);
                    double[] value = new double[traceLen];

                    int groupId = featureGroup[featureId];
                    if (groupId >= 0) {
                        double freq = groupFreqs[groupId] * featureLocalFreq[featureId];
                        double phase = groupPhaseShift[groupId] + samplePhase[groupId][sampleId] + featurePhaseOffset[featureId];
                        for (int i = 0; i < traceLen; i++) {
                            double base = Math.sin(freq * time[i] + phase);
                            double harmonic = 0.35 * Math.cos(0.5 * freq * time[i] - phase);
                            double envelope = 1.0 + 0.2 * Math.sin((time[i] / Math.max(duration, 1e-9)) * Math.PI);
                            value[i] = featureSign[featureId]
                                    * featureScale[featureId]
                                    * sampleAmp[groupId][sampleId]
                                    * (base + harmonic)
                                    * envelope;
                            value[i] += sampleBaseline[groupId][sampleId];
                            value[i] += synthCfg.noiseScale * rng.nextGaussian();
                            time[i] *= groupTimeScale[groupId];
                        }
                    } else {
                        double driftCoef = 0.15 * rng.nextGaussian();
                        for (int i = 0; i < traceLen; i++) {
                            value[i] = 0.8 * rng.nextGaussian();
                            value[i] += driftCoef * time[i];
                            value[i] += synthCfg.noiseScale * rng.nextGaussian();
                        }
                    }

                    if (rng.nextDouble() < synthCfg.nonfiniteTraceRate && traceLen > 0) {
                        int nBad = Math.max(1, (int) Math.round(traceLen * synthCfg.nonfinitePointRate));
                        int[] badIdx = sampleDistinct(rng, traceLen, Math.min(traceLen, nBad));
                        int split = badIdx.length / 2;
                        for (int i = 0; i < split; i++) {
                            time[badIdx[i]] = Double.NaN;
                        }
                        for (int i = split; i < badIdx.length; i++) {
                            value[badIdx[i]] = ((i - split) % 2 == 0) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
                        }
                    }

                    writer.appendTrace(sampleId, featureId, time, value);
                }
            }
            writer.finish();
        }

        writeSampleMetaParquet(sampleMetaPath, sampleIds, y);
        writeFeatureMetaParquet(featureMetaPath, featureGroup, featureKind, featureScale);
        String bundleManifestPath = new File(bundleOutDir, "array_bundle_manifest.json").getAbsolutePath();
        String shardManifestPath = null;
        if (shardOutDir != null && !shardOutDir.isEmpty()) {
            shardManifestPath = ArrayShardBuilder.buildFromBundles(bundleManifestPath, shardOutDir, outShardCfg);
        }
        return new ArraySyntheticArtifacts(bundleManifestPath, sampleMetaPath, featureMetaPath, shardManifestPath);
    }

    private static double[] buildTimeAxis(Random rng, int traceLen, double duration, double timeJitter) {
        double[] time = new double[traceLen];
        double sigma = Math.max(timeJitter, 1e-6);
        double sum = 0.0;
        for (int i = 0; i < traceLen; i++) {
            sum += Math.exp(sigma * rng.nextGaussian());
            time[i] = sum;
        }
        for (int i = 0; i < traceLen; i++) {
            time[i] = (time[i] / sum) * duration;
        }
        return time;
    }

    private static void writeSampleMetaParquet(String sampleMetaPath, long[] sampleIds, double[] y) throws SQLException {
        try (Connection conn = DuckDBUtils.connect(null)) {
            try (Statement st = conn.createStatement()) {
                st.execute("CREATE TEMP TABLE tmp_array_sample_meta (sample_id BIGINT, sample_key VARCHAR, y DOUBLE)");
            }
            try (PreparedStatement ps = conn.prepareStatement("INSERT INTO tmp_array_sample_meta VALUES (?, ?, ?)")) {
                int batch = 0;
                for (int i = 0; i < sampleIds.length; i++) {
                    ps.setLong(1, i);
                    ps.setString(2, String.format("sample_%06d", i));
                    ps.setDouble(3, y[i]);
                    ps.addBatch();
                    batch++;
                    if (batch >= 1024) {
                        ps.executeBatch();
                        batch = 0;
                    }
                }
                if (batch > 0) {
                    ps.executeBatch();
                }
            }
            try (Statement st = conn.createStatement()) {
                st.execute("COPY tmp_array_sample_meta TO " + DuckDBUtils.quotePath(sampleMetaPath) + " (FORMAT PARQUET)");
                st.execute("DROP TABLE tmp_array_sample_meta");
            }
        }
    }

    private static void writeFeatureMetaParquet(String featureMetaPath, int[] featureGroup, String[] featureKind, double[] featureScale) throws SQLException {
        try (Connection conn = DuckDBUtils.connect(null)) {
            try (Statement st = conn.createStatement()) {
                st.execute("CREATE TEMP TABLE tmp_array_feature_meta (feature_id INTEGER, feature_key VARCHAR, feature_kind VARCHAR, group_id INTEGER, scale DOUBLE)");
            }
            try (PreparedStatement ps = conn.prepareStatement("INSERT INTO tmp_array_feature_meta VALUES (?, ?, ?, ?, ?)")) {
                int batch = 0;
                for (int i = 0; i < featureGroup.length; i++) {
                    ps.setInt(1, i);
                    ps.setString(2, String.format("feature_%06d", i));
                    ps.setString(3, featureKind[i]);
                    ps.setInt(4, featureGroup[i]);
                    ps.setDouble(5, featureScale[i]);
                    ps.addBatch();
                    batch++;
                    if (batch >= 1024) {
                        ps.executeBatch();
                        batch = 0;
                    }
                }
                if (batch > 0) {
                    ps.executeBatch();
                }
            }
            try (Statement st = conn.createStatement()) {
                st.execute("COPY tmp_array_feature_meta TO " + DuckDBUtils.quotePath(featureMetaPath) + " (FORMAT PARQUET)");
                st.execute("DROP TABLE tmp_array_feature_meta");
            }
        }
    }

    private static void ensureParentDir(String path) {
        File parent = new File(path).getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("failed to create parent dir: " + parent.getAbsolutePath());
        }
    }

    private static int[] sampleGroupSizes(Random rng, int nGroups, int total, double mean) {
        int[] sizes = new int[nGroups];
        int sum = 0;
        for (int i = 0; i < nGroups; i++) {
            int v = Math.max(1, samplePoisson(rng, Math.max(mean, 1.0)));
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
        List<Integer> pool = new ArrayList<Integer>();
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
        List<Integer> list = new ArrayList<Integer>();
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

    private static double uniform(Random rng, double lo, double hi) {
        return lo + (hi - lo) * rng.nextDouble();
    }
}
