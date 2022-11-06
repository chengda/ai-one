package io.github.chengda.ai_one;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BpNetTrainer {
    public static String SUCCEEDED = "SUCCEEDED";
    public static String FAILED = "FAILED";
    private BpNetModel model;
    private Builder builder;
    private List<double[]> examples;

    private BpNetTrainer(Builder builder) {
        this.builder = builder;
    }

    public static class Builder {
        private int[] neuronNums;
        private double weight;
        private double bias;
        private double learningRate;

        public int[] getNeuronNums() {
            return neuronNums;
        }

        public double getWeight() {
            return weight;
        }

        public double getBias() {
            return bias;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public Builder initLayers(int... neuronNums) {
            this.neuronNums = neuronNums;
            return this;
        }

        public Builder initWeight(double weight) {
            this.weight = weight;
            return this;
        }

        public Builder initBias(double bias) {
            this.bias = bias;
            return this;
        }

        public Builder initLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public BpNetTrainer build() {
            return new BpNetTrainer(Builder.this);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public BpNetTrainer loadExamples(double[][] examples) {
        if (examples == null || examples.length == 0) {
            return this;
        }
        for (double[] example : examples) {
            this.addExample(example);
        }
        return this;
    }

    public BpNetTrainer addExample(double... example) {
        if (this.examples == null) {
            this.examples = new ArrayList<>();
        }
        this.examples.add(example);
        return this;
    }

    private Builder getBuilder() {
        return builder;
    }

    public BpNetModel getModel() {
        return model;
    }

    public void setModel(BpNetModel model) {
        this.model = model;
    }

    public String train(int times) {
        initModel();
        BpNet bpNet = BpNet.build(getModel());
        for (int i = 0; i < times; i++) {
            getExamples().forEach(example -> {
                int[] neuronNums = builder.getNeuronNums();
                double[] inputs = Arrays.copyOfRange(example, 0, neuronNums[0]);
                double[] outputs = bpNet.execute(inputs);
                double error = calculateError(outputs, Arrays.copyOfRange(example, neuronNums[0], example.length));
                feedback(error);
            });
        }
        return SUCCEEDED;
    }

    private void feedback(double error) {
    }

    private double calculateError(double[] outputs, double[] expectedOutputs) {
        double error = 0.0d;
        for (int i = 0, n = outputs.length; i < n; i++) {
            error += Math.exp(outputs[i] - expectedOutputs[i]);
        }
        return error / 2.0d;
    }

    private List<double[]> getExamples() {
        return this.examples;
    }

    private void initModel() {
        setModel(new BpNetModel());
        int[] neuronNums = builder.getNeuronNums();
        double weight = builder.getWeight();
        double bias = builder.getBias();
        //初始化权重
        List<double[][]> weights = new ArrayList<>();
        for (int i = 0, n = neuronNums.length - 1; i < n; i++) {
            double[][] layerWeights = new double[neuronNums[i]][neuronNums[i + 1]];
            for (int j = 0, jn = layerWeights.length; j < jn; j++) {
                for (int k = 0, kn = layerWeights[j].length; k < kn; k++) {
                    layerWeights[j][k] = weight;
                }
            }
            weights.add(layerWeights);
        }
        getModel().setWeights(weights);
        //初始化偏置
        List<double[]> biases = new ArrayList<>();
        for (int i = 0, n = neuronNums.length - 1; i < n; i++) {
            double[] layerBiases = new double[neuronNums[i + 1]];
            for (int j = 0, jn = layerBiases.length; j < jn; j++) {
                layerBiases[j] = bias;
            }
            biases.add(layerBiases);
        }
        getModel().setBiases(biases);
    }
}
