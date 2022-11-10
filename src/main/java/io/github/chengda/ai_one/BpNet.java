package io.github.chengda.ai_one;

import java.util.ArrayList;
import java.util.List;

public class BpNet {
    private BpNetModel model;
    private List<double[]> outputs;

    private BpNet(BpNetModel model) {
        this.model = model;
    }

    public static BpNet build(BpNetModel model) {
        return new BpNet(model);
    }

    public double[] execute(double[] inputs) {
        inputs = BpNetUtils.normalize(inputs, model.getInputNormalizationFactor());
        List<double[][]> weights = model.getWeights();
        double bias = model.getBias();
        List<double[]> biasWeights = model.getBiasWeights();
        List<double[]> outputs = new ArrayList<>();
        double[] layerOutputs = inputs;
        outputs.add(layerOutputs);
        for (int k = 0, p = weights.size(); k < p; k++) {
            double[][] layerWeights = weights.get(k);
            double[] layerBiasWeights = biasWeights.get(k);
            inputs = layerOutputs;
            layerOutputs = new double[layerWeights[0].length];
            for (int i = 0, m = layerOutputs.length; i < m; i++) {
                for (int j = 0, n = inputs.length; j < n; j++) {
                    if (j == 0) {
                        layerOutputs[i] = 0.0d;
                    }
                    layerOutputs[i] += inputs[j] * layerWeights[j][i];

                }
                //加偏置
                layerOutputs[i] = layerOutputs[i] + bias * layerBiasWeights[i];
                //执行激活函数
                layerOutputs[i] = active(layerOutputs[i]);
            }
            outputs.add(layerOutputs);
        }
        this.outputs = outputs;
        double[] finalOutputs = BpNetUtils.denormalize(outputs.get(outputs.size() - 1), model.getOutputNormalizationFactor());
        return finalOutputs;
    }

    public List<double[]> getOutputs() {
        return outputs;
    }

    private double active(double output) {
        return 1d / (1d + Math.exp(-output));
    }
}
