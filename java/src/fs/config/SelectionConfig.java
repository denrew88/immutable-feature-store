package fs.config;

/**
 * Scalar feature selection 설정입니다.
 *
 * <p>y와의 상관 후보를 만들고, 이미 선택한 feature와 너무 비슷한 후보를 제거하는
 * incremental selection 기준값입니다. 처음에는 {@link #topM}만 원하는 개수로 바꾸고
 * 나머지는 기본값을 쓰는 것을 권장합니다.</p>
 */
public class SelectionConfig {
    /** y와의 최소 R^2입니다. 낮을수록 후보가 많이 남습니다. */
    public double yR2Threshold = 0.01;

    /** feature와 y가 동시에 present인 sample 최소 개수입니다. */
    public int minNonNullY = 200;

    /** 후보 feature와 이미 선택한 feature 사이의 최대 허용 R^2입니다. */
    public double ffR2Threshold = 0.9;

    /** feature-feature R^2 계산에 필요한 공통 present sample 수입니다. */
    public int minNonNullPair = 200;

    /** 최종 선택할 feature 수입니다. */
    public int topM = 100;

    /** selection 시작 때 가져올 후보 수입니다. */
    public int initialCap = 2048;

    /** 후보가 부족할 때 한 번에 늘리는 최대 후보 수입니다. */
    public int maxStep = 4096;

    /** 후보 feature를 reader에서 읽어오는 batch 크기입니다. */
    public int batchSize = 1024;

    /** 추가 후보 탐색 중 개선 없이 허용할 gap입니다. 0이면 제한하지 않습니다. */
    public int maxGap = 0;

    /** 후보 총 상한입니다. 0이면 제한하지 않습니다. */
    public int maxCandidates = 0;

    /** 기본값을 사용하는 빈 설정을 만듭니다. */
    public SelectionConfig() {
    }

    /**
     * 자주 바꾸는 threshold와 최종 선택 개수만 지정하는 설정을 만듭니다.
     *
     * <p>stage 확장 관련 값과 maxGap, maxCandidates는 기본값을 유지합니다.</p>
     */
    public SelectionConfig(
            double yR2Threshold,
            int minNonNullY,
            double ffR2Threshold,
            int minNonNullPair,
            int topM) {
        this.yR2Threshold = yR2Threshold;
        this.minNonNullY = minNonNullY;
        this.ffR2Threshold = ffR2Threshold;
        this.minNonNullPair = minNonNullPair;
        this.topM = topM;
    }

    /** 모든 selection parameter를 한 번에 지정하는 설정을 만듭니다. */
    public SelectionConfig(
            double yR2Threshold,
            int minNonNullY,
            double ffR2Threshold,
            int minNonNullPair,
            int topM,
            int initialCap,
            int maxStep,
            int batchSize,
            int maxGap,
            int maxCandidates) {
        this.yR2Threshold = yR2Threshold;
        this.minNonNullY = minNonNullY;
        this.ffR2Threshold = ffR2Threshold;
        this.minNonNullPair = minNonNullPair;
        this.topM = topM;
        this.initialCap = initialCap;
        this.maxStep = maxStep;
        this.batchSize = batchSize;
        this.maxGap = maxGap;
        this.maxCandidates = maxCandidates;
    }
}
