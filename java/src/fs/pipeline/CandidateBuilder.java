package fs.pipeline;

import fs.config.SelectionConfig;
import fs.math.Pearson;
import fs.model.common.SampleMeta;
import fs.model.selection.Candidate;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Selection 검증용 feature 후보 생성 helper다.
 */
public class CandidateBuilder {
    public static List<Candidate> buildCandidatesFromInMemory(double[][] X, byte[][] M, int[] featureIds, int shardSize, SampleMeta meta, SelectionConfig config) {
        List<Candidate> candidates = new ArrayList<Candidate>();
        int shardId = 0;
        int offset = 0;
        for (int i = 0; i < X.length; i++) {
            Pearson.PairwiseResult r = Pearson.pairwiseR2(meta.y, meta.yMask, X[i], M[i], config.minNonNullY);
            if (r.n >= config.minNonNullY && r.r2 >= config.yR2Threshold) {
                candidates.add(new Candidate(featureIds[i], shardId, offset, r.r2, r.n));
            }
            offset++;
            if (offset >= shardSize) {
                shardId++;
                offset = 0;
            }
        }
        Collections.sort(candidates, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate a, Candidate b) {
                int c = Double.compare(b.r2y, a.r2y);
                if (c != 0) {
                    return c;
                }
                return Integer.compare(a.featureId, b.featureId);
            }
        });
        if (config.maxCandidates > 0 && candidates.size() > config.maxCandidates) {
            return new ArrayList<Candidate>(candidates.subList(0, config.maxCandidates));
        }
        return candidates;
    }
}
