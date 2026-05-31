package fs.config;

/**
 * Array synthetic data 생성 설정입니다.
 *
 * <p>저장 포맷 설정이 아니라 테스트 데이터의 크기, trace 길이, missing/nonfinite 비율,
 * random seed를 제어합니다.</p>
 */
public class ArraySyntheticConfig {
    /** 생성할 sample 수입니다. */
    public int nSamples = 1000;

    /** 생성할 feature 수입니다. */
    public int nFeatures = 256;

    /** synthetic latent group 수입니다. feature 간 상관 구조를 만듭니다. */
    public int nLatentGroups = 12;

    /** y에 정보를 주는 latent group 비율입니다. */
    public double informativeGroupRatio = 0.3;

    /** y 생성에 사용할 latent factor 수입니다. */
    public int nLatentForY = 4;

    /** feature group의 평균 크기입니다. */
    public double groupSizeMean = 24.0;

    /** y와 무관한 noise feature 비율입니다. */
    public double noiseFeatureRatio = 0.2;

    /** 같은 latent group 안에서 feature redundancy를 키우는 강도입니다. */
    public double redundantStrength = 1.25;

    /** trace value noise scale입니다. */
    public double noiseScale = 0.15;

    /** trace 최소 point 수입니다. */
    public int minTraceLen = 96;

    /** trace 최대 point 수입니다. */
    public int maxTraceLen = 384;

    /** present지만 길이가 0인 empty trace 비율입니다. */
    public double emptyTraceRate = 0.02;

    /** feature trace가 sample에서 missing일 비율입니다. */
    public double missingFeatureRate = 0.1;

    /** nonfinite 값을 포함하는 trace 비율입니다. */
    public double nonfiniteTraceRate = 0.03;

    /** nonfinite trace 안에서 nonfinite point가 들어갈 비율입니다. */
    public double nonfinitePointRate = 0.02;

    /** time column의 전체 duration입니다. */
    public double timeDuration = 10.0;

    /** time point jitter scale입니다. */
    public double timeJitter = 0.25;

    /** 생성 sample id에 더할 offset입니다. */
    public long sampleIdOffset = 0L;

    /** random seed입니다. */
    public long seed = 0L;
}
