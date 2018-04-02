package com.zimmem;

import com.zimmem.math.Functions;
import com.zimmem.mnist.Mnist;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.bp.BPNetwork;
import com.zimmem.neural.network.bp.Layer;
import com.zimmem.neural.network.bp.TrainContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BpNetworkRunner {

    private static Logger log = LoggerFactory.getLogger(BpNetworkRunner.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        BPNetwork network = NetworkBuilder.bp()
                .addLayer(new Layer(28 * 28, null))
                .addLayer(new Layer(200, Functions.Sigmoid))
                //.addLayer(new Layer(50, Functions.Sigmoid))
                .addLayer(new Layer(10, Functions.Sigmoid))
                .build();

        network.train(Mnist.loadImages("/mnist/train-images.idx3-ubyte"), Mnist.loadLabels("/mnist/train-labels.idx1-ubyte"), 50, 10);

//        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model." + System.currentTimeMillis()));
//        oos.writeObject(network);
//        oos.close();


        // test
        List<MnistImage> testImages = Mnist.loadImages("/mnist/t10k-images.idx3-ubyte");
        List<MnistLabel> testLabels = Mnist.loadLabels("/mnist/t10k-labels.idx1-ubyte");

        AtomicInteger collect = new AtomicInteger(0);
        AtomicInteger tested = new AtomicInteger(0);
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        CountDownLatch latch = new CountDownLatch(testImages.size());
        for (int i = 0; i < testImages.size(); i++) {
            MnistLabel label = testLabels.get(i);
            MnistImage image = testImages.get(i);
            executor.execute(() -> {
                double[] output = network.forward(new TrainContext(), image);
                double max = Arrays.stream(output).max().getAsDouble();
                if (!Double.isNaN(output[label.getValue()]) && Objects.equals(max, output[label.getValue()])) {
                    collect.incrementAndGet();
                }
                tested.incrementAndGet();
                if (tested.get() % 100 == 0) {
                    log.info("testing mnist : {}/{} - {}", collect, tested, (double) collect.doubleValue() / tested.doubleValue());
                }
                latch.countDown();
            });
        }
        latch.await();
        log.info("test finish {}/{} = {}", collect, testImages.size(), (double) collect.get() / testImages.size());
        executor.shutdown();

    }
}
