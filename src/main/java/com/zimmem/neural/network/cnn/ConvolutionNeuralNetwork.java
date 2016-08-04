package com.zimmem.neural.network.cnn;

import com.zimmem.math.Functions;
import com.zimmem.math.Matrix;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.Network;
import com.zimmem.neural.network.bp.TrainContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.IntStream;

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
                if (batch % 1000 == 0) {
                    // 每训练1000个数据， 拿最后1000个数据做下验证
                    verifyRate = verify(images.subList(images.size() - 1000, images.size()));
                }

                batchCorrect.set(0);
                List<CnnContext> contexts = new ArrayList<>(batchSize);
                CountDownLatch latch = new CountDownLatch(batchSize);
                for (int index = batch; index < batch + batchSize && index < images.size(); index++) {
                    MnistImage image = images.get(index);
                    //executor.execute(() -> {
                    CnnContext context = new CnnContext();
                    synchronized (contexts) {
                        contexts.add(context);
                    }
                    List<Matrix> output = forward(context, image);
                    long count = output.stream().mapToDouble(m -> m.getValue(0, 0)).filter(i -> i == 1d).count();
                    if (count > 0) {
                        throw new RuntimeException("xx " + count);
                    }
                    double max = output.stream().mapToDouble(m -> m.getValue(0, 0)).max().getAsDouble();
                    long noMaxCount = output.stream().mapToDouble(m -> m.getValue(0, 0)).filter(d -> d == max).count();

                    if (!Double.isNaN(max) && Objects.equals(max, output.get(image.getLabel()).getValue(0, 0)) && noMaxCount < 2) {
                        //System.out.println(output);
                        batchCorrect.getAndIncrement();
                    }

                    List<Matrix> outputDeltas = new ArrayList<Matrix>(outputLayer.outputCount);
                    for (int i = 0; i < outputLayer.outputCount; i++) {
                        // 输出层残差， 目前只支持激活函数为Sigmoid的情况
                        double delta = ((i == image.getLabel() ? 1 : 0) - output.get(i).getValue(0, 0)) * Functions.SigmoidDerivative.apply(context.features.get(outputLayer.preLayer).get(i).getValue(0, 0));
                        outputDeltas.add(Matrix.zeros(1, 1).setValue(0, 0, delta));

                    }
                    context.deltas.put(outputLayer, outputDeltas);
                    outputLayer.backPropagationDelta(context);
                    //System.out.println(context.deltas.get(inputLayer.nextLayer).get(0).getColumn());
                    latch.countDown();
                    //});
                }
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                correct += batchCorrect.get();
                log.debug("batch {} : {}/{} - total {}/{} = {} ", batch / batchSize + 1, batchCorrect, batchSize, correct, batch + batchSize, (double) correct / (batch + batchSize));

                outputLayer.backPropagationUpdate(contexts, .5d);
                //outputLayer.backPropagationUpdate(contexts, Math.pow(1 - verifyRate , epoch));
                //outputLayer.backPropagationUpdate(contexts, Math.pow(1 - verifyRate, 0.85));


            }
            //present(epoch);
            log.info("epoch {}:  {} / {} ", epoch, correct, images.size());
        }

    }

    public List<Matrix> forward(CnnContext context, MnistImage image) {

        context.features.put(inputLayer, (List<Matrix>) inputConverter.apply(image));
        return inputLayer.forward(context);

    }

    public void shutdown() throws InterruptedException {
        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);
    }

    private double verify(List<MnistImage> mnistImages) {
        AtomicInteger collect = new AtomicInteger(0);


        CountDownLatch latch = new CountDownLatch(mnistImages.size());
        IntStream.range(0, mnistImages.size()).forEach(i -> {
            executor.execute(() -> {
                List<Matrix> output = forward(new CnnContext(), mnistImages.get(i));

                double max = output.stream().mapToDouble(o -> o.getValue(0, 0)).max().getAsDouble();
                if (Objects.equals(output.get(mnistImages.get(i).getLabel()).getValue(0, 0), max)) {
                    collect.incrementAndGet();
                }
                if(i == mnistImages.size() - 1){
                    System.out.println(output);
                }
                latch.countDown();
            });
        });
        try {
            latch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        double rate = collect.doubleValue() / mnistImages.size();
        log.info("verified {}/{} = {}", collect, mnistImages.size(), rate);
        return rate;
    }
}
