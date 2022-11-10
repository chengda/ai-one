package io.github.chengda.ai_one;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class BpNetModel {
    private List<double[][]> weights;
    private double bias;
    private List<double[]> biasWeights;
    private double inputNormalizationFactor;
    private double outputNormalizationFactor;

    public double getInputNormalizationFactor() {
        return inputNormalizationFactor;
    }

    public void setInputNormalizationFactor(double inputNormalizationFactor) {
        this.inputNormalizationFactor = inputNormalizationFactor;
    }

    public double getOutputNormalizationFactor() {
        return outputNormalizationFactor;
    }

    public void setOutputNormalizationFactor(double outputNormalizationFactor) {
        this.outputNormalizationFactor = outputNormalizationFactor;
    }

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

    public void exportTo(OutputStream outputStream) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new OutputStreamWriter(outputStream, StandardCharsets.UTF_8), this);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void importFrom(InputStream inputStream) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            BpNetModel model = objectMapper.readValue(new InputStreamReader(inputStream, StandardCharsets.UTF_8), getClass());
            this.setBiasWeights(model.getBiasWeights());
            this.setWeights(model.getWeights());
            this.setBias(model.getBias());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
