package io.github.ai_one;

import io.github.chengda.ai_one.BpNet;
import io.github.chengda.ai_one.BpNetModel;
import io.github.chengda.ai_one.BpNetTrainer;
import org.junit.Test;

import java.util.List;

public class UnitTest1 {
    @Test
    public void test1() throws Exception {
        BpNetTrainer trainer = BpNetTrainer.builder()
                .initBias(-1)
                .initLearningRate(0.7)
                .initLayers(2, 3, 1)
                .initNormalizationFactor(4, 10)
                .build();
        String result = trainer
                .addExample(1, 1, 2)
                .addExample(1, 2, 3)
                .train(5000000, 0.00001);
        System.out.println(result);
        if (result == BpNetTrainer.SUCCEEDED) {
            BpNet bpNet = BpNet.build(trainer.getModel());
            double[] outputs = bpNet.execute(new double[]{1, 2});
            System.out.println(outputs[0]);
            outputs = bpNet.execute(new double[]{1, 5});
            System.out.println(outputs[0]);
        }
    }

    @Test
    public void test2() throws Exception {
        BpNetModel model = new BpNetModel();
        model.importFrom(getClass().getResourceAsStream("xor.json"));
        BpNet bpNet = BpNet.build(model);
        double[] outputs = bpNet.execute(new double[]{1, 1});
        System.out.println(outputs[0]);
        outputs = bpNet.execute(new double[]{0, 1});
        System.out.println(outputs[0]);
        outputs = bpNet.execute(new double[]{0, 0});
        System.out.println(outputs[0]);
        outputs = bpNet.execute(new double[]{1, 0});
        System.out.println(outputs[0]);
    }
}
