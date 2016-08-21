package com.zimmem;

import com.zimmem.cifar.Cifar;
import com.zimmem.cifar.CifarImage;
import com.zimmem.math.ActivationFunction;
import com.zimmem.math.Matrix;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.cnn.*;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/8.
 */
public class Cifar10CnnRunner {

    static ConvolutionNeuralNetwork network;

    public static void main(String[] args) throws InterruptedException, IOException {

        List<CifarImage> testImages = Cifar.loadTestImages();

        network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(32, 32, 3))
                .addLayer(new CnnConvolutionLayer(5, 5, 16)) // 28
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 14
                .addLayer(new CnnConvolutionLayer(5, 5, 30)) // 10
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 5
                .addLayer(new CnnConvolutionLayer(5, 5, 10))
                .addLayer(new CnnSoftmaxLayer())
                .addListener(new Stat2LogListener() {

                    @Override
                    public void onBatchFinish(List<CnnTrainContext> contexts) {

                        AtomicInteger success = new AtomicInteger(0);
                        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                        CountDownLatch latch = new CountDownLatch(1000);
                        if (totalTrained.intValue() % 1000 == 0) {
                            testImages.forEach(img -> {
                                executor.execute(() -> {
                                    CnnContext context = new CnnContext();
                                    context.setInputs(img.asMatrices());
                                    List<Matrix> output = network.forward(context);
                                    List<Matrix> expected = IntStream.range(0, 10).mapToObj(i -> Matrix.single(i == img.getLabel() ? 1 : 0)).collect(Collectors.toList());
                                    if (maxLabel(output) == maxLabel(expected)) {
                                        success.incrementAndGet();
                                    }
                                    latch.countDown();
                                });
                            });
                            try {
                                latch.await();
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            log.info("validate : {} / 1000 = {}", success.intValue(), success.doubleValue() / 1000);

                        }
                        super.onBatchFinish(contexts);
                    }
                })
                .build();


        try {
            List<CifarImage> cifarImages = Cifar.loadTrandImages();
            List<CnnTrainInput> inputs = IntStream.range(0, cifarImages.size()).mapToObj(i -> {
                CifarImage image = cifarImages.get(i);
                int label = image.getLabel();
                List<Matrix> expected = IntStream.range(0, 10).mapToObj(l -> Matrix.single(l == label ? 1d : 0d)).collect(Collectors.toList());
                return new CnnTrainInput(image.asMatrices(), expected);
            }).collect(Collectors.toList());

            network.train(inputs, 4, 50, 0.01);

        } finally {
            network.shutdown();
        }
    }
}