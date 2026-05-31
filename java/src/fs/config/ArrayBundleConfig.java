package fs.config;

/**
 * Array custom binary의 중간 bundle flush 설정입니다.
 *
 * <p>bundle은 custom binary shard를 만들기 전의 중간 parquet 묶음입니다. bundle 파일이
 * 너무 많이 생기면 값을 올리고, bundle 생성 중 메모리 피크가 크면 값을 줄입니다.</p>
 */
public class ArrayBundleConfig {
    /** bundle 하나에 넣을 최대 trace row 수입니다. */
    public int maxBundleRows = 10000;

    /** bundle 하나의 목표 최대 byte 수입니다. */
    public long maxBundleBytes = 128L * 1024L * 1024L;
}
