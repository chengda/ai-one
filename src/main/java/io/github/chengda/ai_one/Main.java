package io.github.chengda.ai_one;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        BpNetTrainer bpNetTrainer = BpNetTrainer.builder()
                .initLayers(2, 2, 1)
                .initBias(2)
                .initLearningRate(0.6)
                .build();
        String result = bpNetTrainer
                .addExample(1, 1, 0)
                .addExample(1, 0, 1)
                .addExample(0, 0, 0)
                .addExample(0, 1, 1)
                .train(100000, 0.005);

        BpNetModel model = bpNetTrainer.getModel();
        List<double[][]> weights = model.getWeights();
        List<double[]> biasWeights = model.getBiasWeights();
        for (int i = 0; i < weights.size(); i++) {
            System.out.println("第" + (i + 1) + "~" + (i + 2) + "层：---------------");
            double[][] layerWeights = weights.get(i);
            for (int j = 0; j < layerWeights.length; j++) {
                for (int k = 0; k < layerWeights[j].length; k++) {
                    System.out.println(j + "\t" + k + "\t" + layerWeights[j][k]);
                }
            }
            double[] layerBiasWeights = biasWeights.get(i);
            for (int j = 0; j < layerBiasWeights.length; j++) {
                System.out.println("B\t" + j + "\t" + layerBiasWeights[j]);
            }
            System.out.println();
        }

        if (BpNetTrainer.SUCCEEDED.equals(result)) {
            BpNet bpNet = BpNet.build(model);
            System.out.println("SUCCEEDED");
            List<double[]> outputs = bpNet.execute(new double[]{1, 1});
            System.out.println(outputs.get(outputs.size() - 1)[0]);
            outputs = bpNet.execute(new double[]{0, 1});
            System.out.println(outputs.get(outputs.size() - 1)[0]);
            outputs = bpNet.execute(new double[]{0, 0});
            System.out.println(outputs.get(outputs.size() - 1)[0]);
            outputs = bpNet.execute(new double[]{1, 0});
            System.out.println(outputs.get(outputs.size() - 1)[0]);
        } else {
            System.out.println("FAILED");
        }
    }
}
