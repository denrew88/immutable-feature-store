package fs.model.scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * dense-long scalar parquet shard 전체를 설명하는 manifest 모델이다.
 *
 * <p>dense-long은 모든 {@code (feature_id, sample_id)} 조합을 parquet row로 저장한다.
 * missing은 row 생략이 아니라 {@code mask=0}으로 표현하므로, sample 기준 조회에서
 * dense matrix를 다시 채우는 scatter 비용이 작다.
 */
public final class ScalarDenseLongManifest {
    public final String manifestPath;
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final int nSamples;
    public final int nFeatures;
    public final String partsPath;
    public final List<ScalarDenseLongPart> parts;
    public final String featureLocatorPath;
    public final String sampleKeyCol;
    public final String featureKeyCol;
    public final String sampleIdCol;
    public final String featureIdCol;
    public final String valueCol;
    public final String maskCol;
    public final String compression;
    public final int rowGroupFeatures;
    public final Long targetPartBytes;
    public final Map<String, String> selectionStats;

    public ScalarDenseLongManifest(
            String manifestPath,
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            int nFeatures,
            String partsPath,
            List<ScalarDenseLongPart> parts,
            String featureLocatorPath,
            String sampleKeyCol,
            String featureKeyCol,
            String sampleIdCol,
            String featureIdCol,
            String valueCol,
            String maskCol,
            String compression,
            int rowGroupFeatures,
            Long targetPartBytes,
            Map<String, String> selectionStats) {
        this.manifestPath = manifestPath == null ? "" : manifestPath;
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.nSamples = nSamples;
        this.nFeatures = nFeatures;
        this.partsPath = partsPath;
        this.parts = Collections.unmodifiableList(new ArrayList<ScalarDenseLongPart>(parts));
        this.featureLocatorPath = featureLocatorPath;
        this.sampleKeyCol = sampleKeyCol == null ? "sample_key" : sampleKeyCol;
        this.featureKeyCol = featureKeyCol == null ? "feature_key" : featureKeyCol;
        this.sampleIdCol = sampleIdCol == null ? "sample_id" : sampleIdCol;
        this.featureIdCol = featureIdCol == null ? "feature_id" : featureIdCol;
        this.valueCol = valueCol == null ? "value" : valueCol;
        this.maskCol = maskCol == null ? "mask" : maskCol;
        this.compression = compression == null ? "zstd" : compression;
        this.rowGroupFeatures = rowGroupFeatures <= 0 ? 128 : rowGroupFeatures;
        this.targetPartBytes = targetPartBytes;
        this.selectionStats = normalize(selectionStats);
    }

    public String selectionStatsPath(String yCol) {
        String path = selectionStats.get(yCol);
        return path == null ? "" : path;
    }

    private static Map<String, String> normalize(Map<String, String> raw) {
        if (raw == null || raw.isEmpty()) {
            return Collections.emptyMap();
        }
        LinkedHashMap<String, String> out = new LinkedHashMap<String, String>();
        for (Map.Entry<String, String> entry : raw.entrySet()) {
            out.put(entry.getKey(), entry.getValue());
        }
        return Collections.unmodifiableMap(out);
    }
}
