package fs.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
