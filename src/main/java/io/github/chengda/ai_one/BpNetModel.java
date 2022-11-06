package io.github.chengda.ai_one;

import java.util.List;

public class BpNetModel {
    private List<double[][]> weights;
    private List<double[]> biases;

    public List<double[][]> getWeights() {
        return weights;
    }

    public void setWeights(List<double[][]> weights) {
        this.weights = weights;
    }

    public List<double[]> getBiases() {
        return biases;
    }

    public void setBiases(List<double[]> biases) {
        this.biases = biases;
    }
}
