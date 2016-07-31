package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class ConvolutionNeuralNetwork implements Network {

    private static Logger log = LoggerFactory.getLogger(ConvolutionNeuralNetwork.class);

    private ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    CnnLayer inputLayer;

    CnnLayer outputLayer;

    Function inputConverter;

    @Override
    public void train(List<MnistImage> images, List<MnistLabel> labels, int batchSize, int repeat) {

        long start = System.currentTimeMillis();
        log.info("begin to train at {}", start);

        for (int i = 0; i < images.size(); i++) {
            images.get(i).setLabel(labels.get(i).getValue());
        }

        for (int epoch = 1; epoch <= repeat; epoch++) {
            Collections.shuffle(images);
            int correct = 0;
            AtomicInteger batchCorrect = new AtomicInteger(0);
            double verifyRate = 0d;
            for (int batch = 0; batch < images.size(); batch += batchSize) {
//                if (batch % 1000 == 0) {
//                    // 每训练1000个数据， 拿最后1000个数据做下验证
//                    verifyRate = verify(images.subList(images.size() - 1000, images.size()));
//                }

                batchCorrect.set(0);
                List<CnnContext> contexts = new ArrayList<>(batchSize);
                CountDownLatch latch = new CountDownLatch(batchSize);
                for (int index = batch; index < batch + batchSize && index < images.size(); index++) {
                    MnistImage image = images.get(index);
                    executor.execute(() -> {
                        CnnContext context = new CnnContext();
                        synchronized (contexts) {
                            contexts.add(context);
                        }
                        Matrix output = forward(context, image).get(0);

//                        if (!Double.isNaN(output[image.getLabel()]) && Objects.equals(Arrays.stream(output).max().getAsDouble(), output[image.getLabel()])) {
//                            //System.out.println(Arrays.toString(output));
//                            batchCorrect.getAndIncrement();
//                        }
//
//                        // 输出与期望的偏差
//                        double[] expect = new double[output.length];
//                        expect[image.getLabel()] = 1;
//                        double[] error = new double[output.length];
//                        for (int i = 0; i < output.length; i++) {
//                            error[i] = output[i] - expect[i];
//                        }

                        // 计算 output 层偏差
//                        double[] deltas = new double[outputLayer.size];
//                        IntStream.range(0, outputLayer.size).forEach(i -> deltas[i] = error[i] * Functions.SigmoidDerivative.apply(context.weightedInputs.get(outputLayer)[i]));
//                        outputLayer.backPropagationDelta(context, deltas);
                        latch.countDown();
                    });
                }
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                correct += batchCorrect.get();
                log.debug("batch {} : {}/{} - total {}/{} = {} ", batch / batchSize + 1, batchCorrect, batchSize, correct, batch + batchSize, (double) correct / (batch + batchSize));

                outputLayer.backPropagationUpdate(contexts, Math.pow(1 - verifyRate, 3));
                //resetTrainData();


            }
            //present(epoch);
            log.info("epoch {}:  {} / {} ", epoch, correct, images.size());
        }

    }

    public List<Matrix> forward(CnnContext context, MnistImage image) {

        context.features.put(inputLayer, (List<Matrix>) inputConverter.apply(image));
        return inputLayer.forward(context);

    }

    private double verify(List<MnistImage> mnistImages) {
        return 0d;
    }
}
