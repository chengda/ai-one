package io.github.chengda.ai_one;

public class Main {
    public static void main(String[] args) {
        BpNetTrainer bpNetTrainer = BpNetTrainer.builder()
                .initLayers(2, 3, 1)
                .initWeight(1)
                .initBias(1)
                .build();
        String result = bpNetTrainer.addExample(1, 1, 0)
                .addExample(1, 0, 1)
                .addExample(0, 1, 1)
                .addExample(0, 0, 0)
                .train(10000);

        if (BpNetTrainer.SUCCEEDED.equals(result)) {
            BpNet bpNet = BpNet.build(bpNetTrainer.getModel());
            double[] outputs = bpNet.execute(new double[]{1, 0});
            System.out.println(outputs[0]);
        }
    }
}
