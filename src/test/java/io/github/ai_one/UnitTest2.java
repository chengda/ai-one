package io.github.ai_one;

import io.github.chengda.ai_one.BpNet;
import io.github.chengda.ai_one.BpNetTrainer;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;

public class UnitTest2 {
    @Test
    public void test1() throws Exception {
        BpNetTrainer trainer = BpNetTrainer.builder()
                .initBias(-1)
                .initLearningRate(0.6)
                .initLayers(900, 3, 1)
                .initNormalizationFactor(26777216d, 10d)
                .build();
        String result = trainer
                .addExample(createExample("num_train/1.png", 1))
                .addExample(createExample("num_train/2.png", 2))
                .addExample(createExample("num_train/3.png", 3))
                .addExample(createExample("num_train/3.1.png", 3))
                .addExample(createExample("num_train/4.png", 4))
                .addExample(createExample("num_train/5.png", 5))
                .train(1000000, 0.00001);
        System.out.println(result);
        if (result == BpNetTrainer.SUCCEEDED) {
            BpNet bpNet = BpNet.build(trainer.getModel());
            double[] outputs = bpNet.execute(loadPic("num_test/3.png"));
            System.out.println(outputs[0]);
            outputs = bpNet.execute(loadPic("num_test/2.png"));
            System.out.println(outputs[0]);
        }
    }

    private double[] loadPic(String picSrc) throws Exception {
        InputStream stream = getClass().getResourceAsStream(picSrc);
        BufferedImage img = ImageIO.read(stream);
        int w = img.getWidth();
        int h = img.getHeight();
        int[] inputs = new int[w * h];
        img.getRGB(0, 0, w, h, inputs, 0, w);
        double[] example = new double[900];
        int offsetVertical = (30 - h) / 2;
        int offsetHorizontal = (30 - w) / 2;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
                int ii = i - offsetVertical;
                int jj = j - offsetHorizontal;
                double value = ii < 0 || jj < 0 || ii > h - 1 || jj > w - 1 ? 0d : (inputs[ii * w + jj] & 0xffffff);
                example[i * 30 + j] = value;
            }
        }
        return example;
    }

    private double[] createExample(String picSrc, int num) throws Exception {
        InputStream stream = getClass().getResourceAsStream(picSrc);
        BufferedImage img = ImageIO.read(stream);
        int w = img.getWidth();
        int h = img.getHeight();
        int[] inputs = new int[w * h];
        img.getRGB(0, 0, w, h, inputs, 0, w);
        double[] example = new double[901];
        int offsetVertical = (30 - h) / 2;
        int offsetHorizontal = (30 - w) / 2;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
                int ii = i - offsetVertical;
                int jj = j - offsetHorizontal;
                double value = ii < 0 || jj < 0 || ii > h - 1 || jj > w - 1 ? 0d : (inputs[ii * w + jj] & 0xffffff);
                example[i * 30 + j] = value;
            }
        }
        example[900] = num;
        return example;
    }
}
