package com.zimmem.neural.network.bp;

import com.zimmem.ThreadContext;
import com.zimmem.math.Functions;
import com.zimmem.mnist.*;
import com.zimmem.neural.network.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BPNetwork implements Network {

    private ExecutorService executor = Executors.newFixedThreadPool(1);

    private Logger log = LoggerFactory.getLogger(BPNetwork.class);

    Layer inputLayer;

    Layer outputLayer;

    @Override
    public void train(List<MnistImage> images, List<MnistLabel> labels, int batchSize, int repeat) {
        long start = System.currentTimeMillis();
        log.info("begin to train at {}", start);

        while (repeat-- > 0) {
            //重复 repeat 次

            int correct = 0;
            AtomicInteger batchCorrect = new AtomicInteger(0);
            double verifyRate = 0d;
            for (int batchBegin = 0; batchBegin < images.size(); batchBegin += batchSize) {
                batchCorrect.set(0);
                AtomicInteger batchNum = new AtomicInteger();

                CountDownLatch countdown = new CountDownLatch(batchSize);
                List<Context> contexts = new ArrayList<>();
                IntStream.range(batchBegin, batchBegin + batchSize).forEach(index -> {
                    executor.execute(() -> {
                        MnistImage image = images.get(index);
                        MnistLabel label = labels.get(index);
                        Context context = new Context();
                        synchronized (contexts) {
                            contexts.add(context);
                        }
                        double[] output = forward(image, context);
                        double[] expect = new double[output.length];
                        expect[label.getValue()] = 1;

                        if (!Double.isNaN(output[label.getValue()]) && Objects.equals(Arrays.stream(output).max().getAsDouble(), output[label.getValue()])) {
                            batchCorrect.incrementAndGet();
                        }
                        batchNum.incrementAndGet();

                        // 输出与期望的偏差
                        double[] error = new double[output.length];
                        for (int i = 0; i < output.length; i++) {
                            error[i] = output[i] - expect[i];
                        }

                        // 计算 output 层偏差
                        double[] deltas = new double[outputLayer.size];
                        IntStream.range(0, outputLayer.size).forEach(i -> deltas[i] = error[i] * Functions.SigmoidDerivative.apply(output[i]));
                        outputLayer.backPropagationDelta(context, deltas);
                        ThreadContext.clear();
                        countdown.countDown();
                    });
                });
                try {
                    countdown.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                correct += batchCorrect.get();
                log.debug("batch {} : {}/{} - total {}/{} = {} ", batchBegin / batchSize + 1, batchCorrect, batchNum, correct, batchBegin + batchSize, (double) correct / (batchBegin + batchSize));
                //
                outputLayer.backPropagationUpdate(0.8, contexts);

                if (batchBegin % 1000 == 0) {
                    // 每训练1000个数据， 拿最后1000个数据做下验证
                    verifyRate = verify(images.subList(images.size() - 1000, images.size()), labels.subList(labels.size() - 1000, labels.size()));
                }

            }
            log.info("repeat {}:  {} / {} ", repeat, correct, images.size());

        }

        log.info("train finish, cast {} ms", System.currentTimeMillis() - start);
    }

    private double verify(List<MnistImage> mnistImages, List<MnistLabel> mnistLabels) {

        AtomicInteger collect = new AtomicInteger();

        CountDownLatch latch = new CountDownLatch(mnistImages.size());
        IntStream.range(0, mnistImages.size()).forEach(i -> executor.execute(() -> {
            double[] output = forward(mnistImages.get(i), new Context());
            double max = Arrays.stream(output).max().getAsDouble();
            if (Objects.equals(max, output[mnistLabels.get(i).getValue()]) && !Double.isNaN(max)) {
                collect.incrementAndGet();
            }
            latch.countDown();
        }));

        try {
            latch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        double rate = collect.doubleValue() / mnistImages.size();
        log.info("valified {}/{} = {}", collect, mnistImages.size(), rate);
        return rate;
    }


    /**
     * 分类
     *
     * @param image
     * @return
     */
    public double[] forward(MnistImage image, Context context) {

        double[] input = new double[image.getValues().length];
        for (int i = 0; i < image.getValues().length; i++) {
            input[i] = (double) image.getValues()[i];
        }
        context.activations.put(inputLayer, input);
        return inputLayer.forward(context);

    }

}
