package fs.config;

/**
 * Array bundle 생성 단계에서 사용하는 출력 크기와 flush 기준을 묶은 설정이다.
 */
public class ArrayBundleConfig {
    public int maxBundleRows = 10000;
    public long maxBundleBytes = 128L * 1024L * 1024L;
}
