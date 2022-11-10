package io.github.chengda.ai_one;

import java.lang.reflect.Array;
import java.util.*;

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
        private double bias;
        private double learningRate;
        private double inputNormalizationFactor;
        private double outputNormalizationFactor;

        public int[] getNeuronNums() {
            return neuronNums;
        }

        public double getBias() {
            return bias;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public double getInputNormalizationFactor() {
            return inputNormalizationFactor;
        }

        public double getOutputNormalizationFactor() {
            return outputNormalizationFactor;
        }

        public Builder initLayers(int... neuronNums) {
            this.neuronNums = neuronNums;
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

        public Builder initNormalizationFactor(double inputNormalizationFactor, double outputNormalizationFactor) {
            this.inputNormalizationFactor = inputNormalizationFactor;
            this.outputNormalizationFactor = outputNormalizationFactor;
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

    public String train(int maxTimes, double acceptableError) {
        initModel();
        BpNet bpNet = BpNet.build(getModel());
        String result = SUCCEEDED;
        for (int i = 0; i < maxTimes; i++) {
            result = SUCCEEDED;
            for (int j = 0, n = getExamples().size(); j < n; j++) {
                double[] example = getExamples().get(j);
                int[] neuronNums = builder.getNeuronNums();
                double[] inputs = Arrays.copyOfRange(example, 0, neuronNums[0]);
                double[] finalOutputs = bpNet.execute(inputs);
                List<double[]> outputs = bpNet.getOutputs();
                double[] expectedOutputs = Arrays.copyOfRange(example, neuronNums[0], example.length);
                double error = calculateError(finalOutputs, expectedOutputs);
                showError(error);
                if (error >= acceptableError) {
                    result = FAILED;
                    feedback(outputs, BpNetUtils.normalize(expectedOutputs, builder.getOutputNormalizationFactor()));
                }
            }
            if (result.equals(SUCCEEDED)) {
                System.out.println("训练次数：" + (i + 1));
                break;
            }
        }
        return result;
    }

    private List<Double> errorList = new LinkedList<>();

    private void showError(double error) {
        if (errorList.size() < 10) {
            errorList.add(error);
        } else {
            errorList.remove(0);
            errorList.add(error);
        }
        double avgError = 0;
        for (double v : errorList) {
            avgError += v;
        }
        avgError = avgError / errorList.size();
        System.out.print("\r");
        int rate = (int) avgError * 10;
        for (int i = 0; i < 100; i++) {
            if (i <= rate) {
                System.out.print("#");
            } else {
                System.out.print(" ");
            }
        }
        System.out.println(" |" + avgError);
    }

    private void feedback(List<double[]> outputs, double[] expectedOutputs) {
        List<double[][]> weights = getModel().getWeights();
        double bias = getModel().getBias();
        List<double[]> biasWeights = getModel().getBiasWeights();
        double[] outputLayerOutputs = outputs.get(outputs.size() - 1);
        double[] errors = new double[outputLayerOutputs.length];
        //计算输出层误差
        for (int i = 0; i < errors.length; i++) {
            errors[i] = expectedOutputs[i] - outputLayerOutputs[i];
        }
        for (int i = 0, n = outputs.size(); i < n - 1; i++) {
            int layer = n - i - 1;
            double[] layerOutputs = outputs.get(layer);
            double[] preLayerOutputs = outputs.get(layer - 1);
            //计算本层残差
            double[] losses = new double[layerOutputs.length];
            for (int j = 0; j < losses.length; j++) {
                losses[j] = errors[j] * layerOutputs[j] * (1 - layerOutputs[j]);
            }
            double[][] layerWeights = weights.get(layer - 1);
            //调整前一层到本层的权重
            for (int j = 0; j < layerWeights.length; j++) {
                for (int k = 0; k < losses.length; k++) {
                    layerWeights[j][k] += preLayerOutputs[j] * losses[k] * builder.getLearningRate();
                }
            }
            weights.set(layer - 1, layerWeights);
            //计算前一层误差
            errors = new double[layerWeights.length];
            for (int j = 0; j < layerWeights.length; j++) {
                errors[j] = 0d;
                for (int k = 0; k < losses.length; k++) {
                    errors[j] += layerWeights[j][k] * losses[k];
                }
            }
            //调整前一层到本层的偏置权重
            double[] layerBiasWeights = biasWeights.get(layer - 1);
            for (int j = 0; j < layerBiasWeights.length; j++) {
                layerBiasWeights[j] += bias * losses[j] * builder.getLearningRate();
            }
            biasWeights.set(layer - 1, layerBiasWeights);
        }
    }

    private double calculateError(double[] outputs, double[] expectedOutputs) {
        double error = 0.0d;
        for (int i = 0, n = outputs.length; i < n; i++) {
            error += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return error / 2.0d;
    }

    private List<double[]> getExamples() {
        return this.examples;
    }

    private void initModel() {
        setModel(new BpNetModel());
        int[] neuronNums = builder.getNeuronNums();
        double bias = builder.getBias();
        //初始化权重
        List<double[][]> weights = new ArrayList<>();
        for (int i = 0, n = neuronNums.length - 1; i < n; i++) {
            double[][] layerWeights = new double[neuronNums[i]][neuronNums[i + 1]];
            for (int j = 0, jn = layerWeights.length; j < jn; j++) {
                for (int k = 0, kn = layerWeights[j].length; k < kn; k++) {
                    layerWeights[j][k] = 0;//Math.random() / jn;
                }
            }
            weights.add(layerWeights);
        }
        getModel().setWeights(weights);
        //设置偏置
        getModel().setBias(bias);
        //初始化偏置权重
        List<double[]> biasWeights = new ArrayList<>();
        for (int i = 0, n = neuronNums.length - 1; i < n; i++) {
            double[] layerBiasWeights = new double[neuronNums[i + 1]];
            for (int j = 0, jn = layerBiasWeights.length; j < jn; j++) {
                layerBiasWeights[j] =0;// Math.random() / jn;
            }
            biasWeights.add(layerBiasWeights);
        }
        getModel().setBiasWeights(biasWeights);
        //初始化归一化因子
        getModel().setInputNormalizationFactor(builder.getInputNormalizationFactor());
        getModel().setOutputNormalizationFactor(builder.getOutputNormalizationFactor());
    }
}
