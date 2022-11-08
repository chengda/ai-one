package io.github.chengda.ai_one;

import java.util.List;

public class BpNetModel {
    private List<double[][]> weights;
    private double bias;
    private List<double[]> biasWeights;

    public List<double[][]> getWeights() {
        return weights;
    }

    public void setWeights(List<double[][]> weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public List<double[]> getBiasWeights() {
        return biasWeights;
    }

    public void setBiasWeights(List<double[]> biasWeights) {
        this.biasWeights = biasWeights;
    }
}
