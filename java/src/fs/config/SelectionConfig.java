package fs.config;

/**
 * Selection 단계에서 후보 수와 상관계수 임계값을 제어하는 설정이다.
 */
public class SelectionConfig {
    public double yR2Threshold = 0.01;
    public int minNonNullY = 200;
    public double ffR2Threshold = 0.9;
    public int minNonNullPair = 200;
    public int topM = 100;
    public int initialCap = 2048;
    public int maxStep = 4096;
    public int batchSize = 1024;
    public int maxGap = 0;
    public int maxCandidates = 0;

    /**
     * 기본값을 그대로 쓰는 빈 설정을 만든다.
     */
    public SelectionConfig() {
    }

    /**
     * 자주 바꾸는 핵심 임계값만 한 번에 지정하는 설정을 만든다.
     *
     * <p>stage 성장 관련 값과 maxGap, maxCandidates는 기본값을 유지한다.
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

    /**
     * 모든 selection 파라미터를 한 번에 지정하는 설정을 만든다.
     */
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
