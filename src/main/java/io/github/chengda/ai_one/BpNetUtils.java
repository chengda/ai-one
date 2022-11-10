package io.github.chengda.ai_one;

import java.util.Arrays;

public class BpNetUtils {
    public static double[] normalize(double[] data, double normalizationFactor) {
        data = Arrays.copyOf(data, data.length);
        for (int i = 0, n = data.length; i < n; i++) {
            data[i] = data[i] / normalizationFactor;
        }
        return data;
    }

    public static double[] denormalize(double[] data, double normalizationFactor) {
        data = Arrays.copyOf(data, data.length);
        for (int i = 0, n = data.length; i < n; i++) {
            data[i] = data[i] * normalizationFactor;
        }
        return data;
    }
}
