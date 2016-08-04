package com.zimmem;

import com.zimmem.math.ActivationFunction;
import com.zimmem.math.Functions;
import com.zimmem.math.Matrix;
import com.zimmem.mnist.Mnist;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.bp.TrainContext;
import com.zimmem.neural.network.cnn.*;
import org.omg.CORBA.Object;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class CnnRunner {

    private static Logger log = LoggerFactory.getLogger(CnnRunner.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        ConvolutionNeuralNetwork network = NetworkBuilder.cnn()
                .setInputConverter(i -> {
                    MnistImage image = (MnistImage) i;
                    Matrix m = new Matrix(28, 28);
                    for (int c = 0; c < 28; c++) {
                        for (int r = 0; r < 28; r++) {
                            m.setValue(r, c, (image.getValues()[28 * c + r] & 0xff) > 0 ? 1d : 0d);
                        }
                    }
                    return Arrays.asList(m);
                })
                .addLayer(new CnnInputLayer(28, 28))
                .addLayer(new CnnConvolutionLayer(5, 5, 6))
                .addLayer(new ActivationLayer(ActivationFunction.Sigmoid))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Means))
                .addLayer(new CnnConvolutionLayer(5, 5, 16))
                .addLayer(new ActivationLayer(ActivationFunction.Sigmoid))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Means))
                .addLayer(new CnnConvolutionLayer(4, 4, 10))
                .addLayer(new ActivationLayer(ActivationFunction.Sigmoid))
                .build();

        try {
            network.train(Mnist.loadImages("/mnist/train-images.idx3-ubyte").subList(0,10000), Mnist.loadLabels("/mnist/train-labels.idx1-ubyte"), 50, 1);
        }finally {
            network.shutdown();
        }

//        List<Matrix> result = network.forward(new CnnContext(), Mnist.loadImages("/mnist/train-images.idx3-ubyte").get(0));
//        System.out.println(result.size());
//        result.stream().forEach(System.out::println);

        List<MnistImage> testImages = Mnist.loadImages("/mnist/t10k-images.idx3-ubyte");
        List<MnistLabel> testLabels = Mnist.loadLabels("/mnist/t10k-labels.idx1-ubyte");

        AtomicInteger collect = new AtomicInteger(0);
        AtomicInteger tested = new AtomicInteger(0);
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        CountDownLatch latch = new CountDownLatch(testImages.size());
        File dir = new File("d:\\cnn-verify\\" + System.currentTimeMillis() + "\\");
        dir.mkdirs();
        log.info("created verify dir at {}", dir.getAbsoluteFile());
        IntStream.range(0, testImages.size()).forEach(i -> {
            MnistLabel label = testLabels.get(i);
            MnistImage image = testImages.get(i);
            executor.execute(() -> {
                List<Matrix> output = network.forward(new CnnContext(), image);
                double max = output.get(0).getValue(0, 0);
                int resultLabel = 0;
                for (int oi = 1; oi < output.size(); oi++) {
                    if (output.get(oi).getValue(0, 0) > max) {
                        max = output.get(oi).getValue(0, 0);
                        resultLabel = oi;
                    }
                }

                File imageFile = null;
                if (resultLabel == label.getValue()) {
                    collect.incrementAndGet();
                    imageFile = new File(dir.getAbsoluteFile() + "\\" + "r_" + i + "_" + resultLabel + ".jpg");
                    System.out.println(output);
                } else {
                    imageFile = new File(dir.getAbsoluteFile() + "\\" + "w_" + i + "_" + resultLabel + ".jpg");
                }
                try {
                    ImageIO.write(image.asImage(), "jpg", imageFile);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                tested.incrementAndGet();
                if (tested.get() % 100 == 0) {
                    log.info("testing mnist : {}/{} - {}", collect, tested, collect.doubleValue() / tested.doubleValue());
                }
                latch.countDown();
            });
        });
        latch.await();
        log.info("test finish {}/{} = {}", collect, testImages.size(), (double) collect.get() / testImages.size());
        executor.shutdown();

        network.shutdown();

    }

}

