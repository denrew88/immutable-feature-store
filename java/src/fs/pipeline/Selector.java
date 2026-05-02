package fs.pipeline;

import fs.config.SelectionConfig;
import fs.io.ShardReader;
import fs.math.Pearson;
import fs.model.Candidate;
import fs.model.Feature;
import fs.model.RowBatch;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Selector {
    public static List<Candidate> selectFeaturesIncremental(List<Candidate> candidates, ShardReader reader, SelectionConfig config) throws SQLException {
        if (config.maxCandidates > 0 && candidates.size() > config.maxCandidates) {
            candidates = new ArrayList<Candidate>(candidates.subList(0, config.maxCandidates));
        }
        int n = candidates.size();
        Candidate[] cands = candidates.toArray(new Candidate[0]);
        boolean[] alive = new boolean[n];
        boolean[] selected = new boolean[n];
        for (int i = 0; i < n; i++) {
            alive[i] = true;
        }
        List<Integer> selectedIdx = new ArrayList<>();
        Map<Integer, CachedFeature> cache = new HashMap<>();

        int capEnd = Math.min(config.initialCap, n);
        int processed = 0;

        while (selectedIdx.size() < config.topM && processed < n) {
            StageBuckets stage = buildStageBuckets(cands, reader, processed, capEnd);

            // prune new range by already selected
            for (int sIdx : selectedIdx) {
                CachedFeature f = cache.get(sIdx);
                if (f == null) {
                    Candidate c = cands[sIdx];
                    f = toCachedFeature(reader.loadFeatureByOffset(c.shardId, c.offsetInShard));
                    cache.put(sIdx, f);
                }
                for (Map.Entry<Integer, BucketData> entry : stage.buckets.entrySet()) {
                    BucketData bucket = entry.getValue();
                    for (int bStart = 0; bStart < bucket.idxs.length; bStart += config.batchSize) {
                        int batchEnd = Math.min(bStart + config.batchSize, bucket.idxs.length);
                        int keepCount = 0;
                        for (int pos = bStart; pos < batchEnd; pos++) {
                            int candIdx = bucket.idxs[pos];
                            if (!alive[candIdx]) {
                                continue;
                            }
                            if (hasMinOverlap(f.validBits, bucket.validBits[pos], config.minNonNullPair)) {
                                keepCount++;
                            }
                        }
                        if (keepCount == 0) {
                            continue;
                        }
                        int[] batchIdxs = new int[keepCount];
                        double[][] batchValues = new double[keepCount][];
                        byte[][] batchValid = new byte[keepCount][];
                        int outPos = 0;
                        for (int pos = bStart; pos < batchEnd; pos++) {
                            int candIdx = bucket.idxs[pos];
                            if (!alive[candIdx]) {
                                continue;
                            }
                            if (!hasMinOverlap(f.validBits, bucket.validBits[pos], config.minNonNullPair)) {
                                continue;
                            }
                            batchIdxs[outPos] = candIdx;
                            batchValues[outPos] = bucket.rows.values[pos];
                            batchValid[outPos] = bucket.rows.valid[pos];
                            outPos++;
                        }
                        Pearson.BatchResult r = Pearson.batchR2OneVsMany(
                                f.values,
                                f.valid,
                                batchValues,
                                batchValid,
                                config.minNonNullPair
                        );
                        for (int i = 0; i < keepCount; i++) {
                            int candIdx = batchIdxs[i];
                            if (alive[candIdx] && r.n[i] >= config.minNonNullPair && r.r2[i] >= config.ffR2Threshold) {
                                alive[candIdx] = false;
                            }
                        }
                    }
                }
            }

            // greedy inside new range
            for (int idx = processed; idx < capEnd; idx++) {
                if (!alive[idx] || selected[idx]) {
                    continue;
                }
                selected[idx] = true;
                selectedIdx.add(idx);
                if (selectedIdx.size() >= config.topM) {
                    break;
                }

                CachedFeature f = cache.get(idx);
                if (f == null) {
                    f = stage.featureByCandidate.get(idx);
                    if (f == null) {
                        Candidate c = cands[idx];
                        f = toCachedFeature(reader.loadFeatureByOffset(c.shardId, c.offsetInShard));
                    }
                    cache.put(idx, f);
                }

                for (Map.Entry<Integer, BucketData> entry : stage.buckets.entrySet()) {
                    BucketData bucket = entry.getValue();
                    for (int bStart = 0; bStart < bucket.idxs.length; bStart += config.batchSize) {
                        int batchEnd = Math.min(bStart + config.batchSize, bucket.idxs.length);
                        int keepCount = 0;
                        for (int pos = bStart; pos < batchEnd; pos++) {
                            int candIdx = bucket.idxs[pos];
                            if (candIdx > idx && alive[candIdx] && !selected[candIdx]
                                    && hasMinOverlap(f.validBits, bucket.validBits[pos], config.minNonNullPair)) {
                                keepCount++;
                            }
                        }
                        if (keepCount == 0) {
                            continue;
                        }
                        int[] batchIdxs = new int[keepCount];
                        double[][] batchValues = new double[keepCount][];
                        byte[][] batchValid = new byte[keepCount][];
                        int outPos = 0;
                        for (int pos = bStart; pos < batchEnd; pos++) {
                            int candIdx = bucket.idxs[pos];
                            if (candIdx <= idx || !alive[candIdx] || selected[candIdx]) {
                                continue;
                            }
                            if (!hasMinOverlap(f.validBits, bucket.validBits[pos], config.minNonNullPair)) {
                                continue;
                            }
                            batchIdxs[outPos] = candIdx;
                            batchValues[outPos] = bucket.rows.values[pos];
                            batchValid[outPos] = bucket.rows.valid[pos];
                            outPos++;
                        }
                        Pearson.BatchResult r = Pearson.batchR2OneVsMany(
                                f.values,
                                f.valid,
                                batchValues,
                                batchValid,
                                config.minNonNullPair
                        );
                        for (int i = 0; i < keepCount; i++) {
                            int candIdx = batchIdxs[i];
                            if (alive[candIdx] && r.n[i] >= config.minNonNullPair && r.r2[i] >= config.ffR2Threshold) {
                                alive[candIdx] = false;
                            }
                        }
                    }
                }
            }

            if (capEnd == n) {
                break;
            }
            processed = capEnd;
            capEnd = Math.min(growCap(capEnd, config.maxStep), n);
        }

        List<Candidate> out = new ArrayList<>();
        for (int idx : selectedIdx) {
            out.add(cands[idx]);
        }
        return out;
    }

    private static int growCap(int current, int maxStep) {
        return current + Math.min(current, maxStep);
    }

    private static StageBuckets buildStageBuckets(Candidate[] cands, ShardReader reader, int start, int end) throws SQLException {
        Map<Integer, ArrayList<Integer>> buckets = new HashMap<>();
        for (int i = start; i < end; i++) {
            Candidate c = cands[i];
            ArrayList<Integer> list = buckets.get(c.shardId);
            if (list == null) {
                list = new ArrayList<Integer>();
                buckets.put(c.shardId, list);
            }
            list.add(i);
        }
        Map<Integer, BucketData> out = new HashMap<>();
        Map<Integer, CachedFeature> featureByCandidate = new HashMap<>();
        for (Map.Entry<Integer, ArrayList<Integer>> entry : buckets.entrySet()) {
            ArrayList<Integer> list = entry.getValue();
            int[] idxs = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                idxs[i] = list.get(i);
            }
            sortByOffset(idxs, cands);
            int[] offsets = new int[idxs.length];
            for (int i = 0; i < idxs.length; i++) {
                offsets[i] = cands[idxs[i]].offsetInShard;
            }
            RowBatch rows = reader.loadRows(entry.getKey(), offsets);
            for (int i = 0; i < idxs.length; i++) {
                featureByCandidate.put(idxs[i], new CachedFeature(rows.values[i], rows.valid[i], packValidBits(rows.valid[i])));
            }
            out.put(entry.getKey(), new BucketData(idxs, rows));
        }
        return new StageBuckets(out, featureByCandidate);
    }

    private static CachedFeature toCachedFeature(Feature feature) {
        return new CachedFeature(feature.values, feature.valid, packValidBits(feature.valid));
    }

    private static long[] packValidBits(byte[] valid) {
        int words = (valid.length + 63) >>> 6;
        long[] bits = new long[words];
        for (int i = 0; i < valid.length; i++) {
            if (valid[i] != 0) {
                bits[i >>> 6] |= (1L << (i & 63));
            }
        }
        return bits;
    }

    private static boolean hasMinOverlap(long[] left, long[] right, int minRequired) {
        if (minRequired <= 0) {
            return true;
        }
        int count = 0;
        for (int i = 0; i < left.length; i++) {
            count += Long.bitCount(left[i] & right[i]);
            if (count >= minRequired) {
                return true;
            }
        }
        return false;
    }

    private static void sortByOffset(int[] idxs, Candidate[] cands) {
        Integer[] idxObj = new Integer[idxs.length];
        for (int i = 0; i < idxs.length; i++) {
            idxObj[i] = idxs[i];
        }
        Arrays.sort(idxObj, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return Integer.compare(cands[a].offsetInShard, cands[b].offsetInShard);
            }
        });
        for (int i = 0; i < idxs.length; i++) {
            idxs[i] = idxObj[i];
        }
    }

    private static class StageBuckets {
        final Map<Integer, BucketData> buckets;
        final Map<Integer, CachedFeature> featureByCandidate;

        StageBuckets(Map<Integer, BucketData> buckets, Map<Integer, CachedFeature> featureByCandidate) {
            this.buckets = buckets;
            this.featureByCandidate = featureByCandidate;
        }
    }

    private static class BucketData {
        final int[] idxs;
        final RowBatch rows;
        final long[][] validBits;

        BucketData(int[] idxs, RowBatch rows) {
            this.idxs = idxs;
            this.rows = rows;
            this.validBits = new long[rows.valid.length][];
            for (int i = 0; i < rows.valid.length; i++) {
                this.validBits[i] = packValidBits(rows.valid[i]);
            }
        }
    }

    private static class CachedFeature {
        final double[] values;
        final byte[] valid;
        final long[] validBits;

        CachedFeature(double[] values, byte[] valid, long[] validBits) {
            this.values = values;
            this.valid = valid;
            this.validBits = validBits;
        }
    }
}
