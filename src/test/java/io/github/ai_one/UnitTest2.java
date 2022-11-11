package io.github.ai_one;

import io.github.chengda.ai_one.BpNet;
import io.github.chengda.ai_one.BpNetTrainer;
import org.junit.Test;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

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
                .addExample(createExample("num_train/4.png", 4))
                .addExample(createExample("num_train/5.png", 5))
                .train(1000000, 0.00001);
        System.out.println(result);
        if (result == BpNetTrainer.SUCCEEDED) {
            BpNet bpNet = BpNet.build(trainer.getModel());
            double[] outputs = bpNet.execute(loadPic("num_test/3.png"));
            System.out.println(outputs[0]);
            outputs = bpNet.execute(loadPic("num_test/4.png"));
            System.out.println(outputs[0]);
        }
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Test
    public void test2() throws Exception {

        Mat src = Imgcodecs.imread("samples/nums/3.1.png");
        HighGui.imshow("nums-1", src);
        HighGui.resizeWindow("nums-1", 400, 300);
        HighGui.waitKey();
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY);
        HighGui.imshow("nums-2", gray);
        HighGui.resizeWindow("nums-2", 400, 300);
        HighGui.waitKey();
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 80, 100, 3);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hi = new Mat();
        Imgproc.findContours(edges, contours, hi, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println(contours.size());
        Rect rect = null;
        int index = -1;
        for (int i = 0; i < contours.size(); i++) {
            Rect tmp = Imgproc.boundingRect(contours.get(i));
            if (rect == null || tmp.area() > rect.area()) {
                rect = tmp;
                index = i;
            }
        }
        MatOfPoint2f mp2f = new MatOfPoint2f(contours.get(index).toArray());
        RotatedRect rrect = Imgproc.minAreaRect(mp2f);
        MatOfPoint points=new MatOfPoint();
        Imgproc.boxPoints(rrect,points);
        List<MatOfPoint> boxes=new ArrayList<>();
        boxes.add(points);
     //   Imgproc.polylines(src,boxes,true,new Scalar(0,255,0),1,Imgproc.LINE_AA);
        Imgproc.drawContours(src,contours,index,new Scalar(255,0,0),1,Imgproc.LINE_AA);
        HighGui.imshow("nums-3", src);
        HighGui.resizeWindow("nums-3", 400, 300);
        HighGui.waitKey();
        System.out.println("he");

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
