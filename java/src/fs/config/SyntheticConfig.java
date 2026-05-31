package fs.config;

/**
 * Scalar synthetic data 생성 설정입니다.
 *
 * <p>저장 포맷 설정이 아니라 테스트용 scalar matrix의 크기, 상관 구조, missing 분포,
 * y column 생성 방식을 제어합니다.</p>
 */
public class SyntheticConfig {
    /** 생성할 sample 수입니다. */
    public int nSamples = 1000;

    /** 생성할 feature 수입니다. */
    public int nFeatures = 2000;

    /** latent group 수입니다. feature 간 상관 구조를 만듭니다. */
    public int nLatentGroups = 20;

    /** feature group의 평균 크기입니다. */
    public double groupSizeMean = 50.0;

    /** y에 정보를 주는 latent group 비율입니다. */
    public double informativeGroupRatio = 0.3;

    /** y 생성에 사용할 latent factor 수입니다. */
    public int nLatentForY = 5;

    /** scalar value noise scale입니다. */
    public double noiseScale = 1.0;

    /** 같은 latent group 안에서 feature redundancy를 키우는 강도입니다. */
    public double redundantStrength = 1.5;

    /** y와 무관한 noise feature 비율입니다. */
    public double noiseFeatureRatio = 0.2;

    /** y와 음의 상관을 갖는 feature 비율입니다. */
    public double negCorrRatio = 0.2;

    /** 기본 missing value 비율입니다. */
    public double missingRate = 0.1;

    /** feature별 missing rate 변동폭입니다. */
    public double missingRateVariability = 0.05;

    /** sparse feature 비율입니다. */
    public double sparseFeatureRatio = 0.05;

    /** sparse feature가 가질 목표 valid sample 수입니다. null이면 자동 계산합니다. */
    public Integer sparseTargetValid = null;

    /** 같은 group 안에서 missing pattern을 공유할 확률입니다. */
    public double groupMissingShareProb = 0.2;

    /** y column missing 비율입니다. */
    public double yMissingRate = 0.0;

    /** random seed입니다. */
    public long seed = 0L;
}
