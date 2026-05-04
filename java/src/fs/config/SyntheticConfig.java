package fs.config;

/**
 * Scalar synthetic 데이터 생성기에서 사용하는 데이터 크기와 타깃 분포를 정의하는 설정이다.
 */
public class SyntheticConfig {
    public int nSamples = 1000;
    public int nFeatures = 2000;
    public int nLatentGroups = 20;
    public double groupSizeMean = 50.0;
    public double informativeGroupRatio = 0.3;
    public int nLatentForY = 5;
    public double noiseScale = 1.0;
    public double redundantStrength = 1.5;
    public double noiseFeatureRatio = 0.2;
    public double negCorrRatio = 0.2;
    public double missingRate = 0.1;
    public double missingRateVariability = 0.05;
    public double sparseFeatureRatio = 0.05;
    public Integer sparseTargetValid = null;
    public double groupMissingShareProb = 0.2;
    public double yMissingRate = 0.0;
    public long seed = 0L;
}
