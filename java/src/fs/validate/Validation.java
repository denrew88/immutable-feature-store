package fs.validate;

import fs.config.SelectionConfig;
import fs.io.scalar.InMemoryShardReader;
import fs.math.Pearson;
import fs.model.selection.Candidate;
import fs.model.common.SampleMeta;
import fs.pipeline.CandidateBuilder;
import fs.pipeline.Selector;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

/**
 * Selection 결과나 synthetic 데이터의 기본 정합성을 확인하는 검증 helper 모음이다.
 */
public class Validation {
    public static void validateBatchKernel(double[][] X, byte[][] M, double[] y, byte[] yMask, int minNonNull, double tol) {
        Pearson.BatchResult batch = Pearson.batchR2OneVsMany(y, yMask, X, M, minNonNull);
        for (int i = 0; i < X.length; i++) {
            Pearson.PairwiseResult pr = Pearson.pairwiseR2(y, yMask, X[i], M[i], minNonNull);
            if (pr.n != batch.n[i]) {
                throw new IllegalStateException("n_valid mismatch at " + i);
            }
            if (Math.abs(pr.r2 - batch.r2[i]) > tol) {
                throw new IllegalStateException("r2 mismatch at " + i);
            }
        }
    }

    public static void validateIncrementalVsNaive(double[][] X, byte[][] M, double[] y, byte[] yMask, SelectionConfig config, int shardSize) throws SQLException {
        SampleMeta meta = new SampleMeta(makeSampleIds(y.length), y, yMask, new String[0]);
        int[] featureIds = new int[X.length];
        for (int i = 0; i < featureIds.length; i++) {
            featureIds[i] = i;
        }

        List<Candidate> candidates = CandidateBuilder.buildCandidatesFromInMemory(X, M, featureIds, shardSize, meta, config);
        InMemoryShardReader reader = new InMemoryShardReader(X, M, shardSize);
        List<Candidate> selected = Selector.selectFeaturesIncremental(candidates, reader, config);

        List<Integer> naive = naiveGreedyFull(X, M, y, yMask, config);
        if (selected.size() != naive.size()) {
            throw new IllegalStateException("selection size mismatch");
        }
        for (int i = 0; i < naive.size(); i++) {
            if (selected.get(i).featureId != naive.get(i)) {
                throw new IllegalStateException("selection mismatch at " + i);
            }
        }
    }

    private static List<Integer> naiveGreedyFull(double[][] X, byte[][] M, double[] y, byte[] yMask, SelectionConfig config) {
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            Pearson.PairwiseResult r = Pearson.pairwiseR2(y, yMask, X[i], M[i], config.minNonNullY);
            if (r.n >= config.minNonNullY && r.r2 >= config.yR2Threshold) {
                candidates.add(i);
            }
        }
        candidates.sort((a, b) -> {
            Pearson.PairwiseResult ra = Pearson.pairwiseR2(y, yMask, X[a], M[a], config.minNonNullY);
            Pearson.PairwiseResult rb = Pearson.pairwiseR2(y, yMask, X[b], M[b], config.minNonNullY);
            int c = Double.compare(rb.r2, ra.r2);
            if (c != 0) {
                return c;
            }
            return Integer.compare(a, b);
        });

        boolean[] alive = new boolean[X.length];
        for (int idx : candidates) {
            alive[idx] = true;
        }
        List<Integer> selected = new ArrayList<>();
        for (int idx : candidates) {
            if (!alive[idx]) {
                continue;
            }
            selected.add(idx);
            if (selected.size() >= config.topM) {
                break;
            }
            for (int j : candidates) {
                if (j <= idx || !alive[j]) {
                    continue;
                }
                Pearson.PairwiseResult r = Pearson.pairwiseR2(X[idx], M[idx], X[j], M[j], config.minNonNullPair);
                if (r.n >= config.minNonNullPair && r.r2 >= config.ffR2Threshold) {
                    alive[j] = false;
                }
            }
        }
        return selected;
    }

    private static long[] makeSampleIds(int n) {
        long[] ids = new long[n];
        for (int i = 0; i < n; i++) {
            ids[i] = i;
        }
        return ids;
    }
}
