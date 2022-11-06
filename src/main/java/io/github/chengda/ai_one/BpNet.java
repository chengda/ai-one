package io.github.chengda.ai_one;

import java.util.List;

public class BpNet {
    private BpNetModel model;

    private BpNet(BpNetModel model) {
        this.model = model;
    }

    public static BpNet build(BpNetModel model) {
        return new BpNet(model);
    }

    public double[] execute(double[] inputs) {
        List<double[][]> weights = model.getWeights();
        List<double[]> biases = model.getBiases();
        double[] outputs = inputs;
        for (int k = 0, p = weights.size(); k < p; k++) {
            double[][] layerWeights = weights.get(k);
            double[] layerBiases = biases.get(k);
            inputs = outputs;
            outputs = new double[layerWeights[0].length];
            for (int i = 0, m = outputs.length; i < m; i++) {
                for (int j = 0, n = inputs.length; j < n; j++) {
                    if (j == 0) {
                        outputs[i] = 0.0d;
                    }
                    outputs[i] += inputs[j] * layerWeights[j][i];
                }
                //执行激活函数
                outputs[i] = active(outputs[i]) + layerBiases[i];
            }
        }
        return outputs;
    }

    private double active(double output) {
        return 1d / (1d + Math.pow(Math.E, -output));
    }
}
