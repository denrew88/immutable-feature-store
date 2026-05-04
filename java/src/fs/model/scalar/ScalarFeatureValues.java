package fs.model.scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Feature 하나에 대해 여러 sample의 scalar 값을 public 형태로 노출하는 모델이다.
 */
public class ScalarFeatureValues {
    public final int featureId;
    public final String featureKey;
    public final List<ScalarValue> values;

    public ScalarFeatureValues(int featureId, String featureKey, List<ScalarValue> values) {
        this.featureId = featureId;
        this.featureKey = featureKey;
        this.values = Collections.unmodifiableList(new ArrayList<ScalarValue>(values));
    }
}
