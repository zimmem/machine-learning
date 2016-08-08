package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class ConvolutionNeuralNetwork /*implements Network*/ {

    private static Logger log = LoggerFactory.getLogger(ConvolutionNeuralNetwork.class);

    private ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    CnnLayer inputLayer;

    CnnLayer outputLayer;

    public List<CnnTrainListener> listeners;

    //@Override
    public void train(List<CnnTrainInput> inputs, int batchSize, int repeat) {

        long start = System.currentTimeMillis();
        log.info("begin to train at {}", start);


        IntStream.range(1, repeat + 1).forEach(epoch -> {
            Collections.shuffle(inputs);
            for (int batch = 0; batch < inputs.size(); batch += batchSize) {
                List<CnnTrainContext> contexts = new ArrayList<>(batchSize);
                CountDownLatch latch = new CountDownLatch(batchSize);
                for (int index = batch; index < batch + batchSize && index < inputs.size(); index++) {
                    CnnTrainInput input = inputs.get(index);
                    CnnTrainContext context = new CnnTrainContext();
                    context.setExcepted(input.getExpected());
                    context.setInputs(input.getInputs());

                    executor.execute(() -> {
                        synchronized (contexts) {
                            contexts.add(context);
                        }
                        List<Matrix> output = forward(context);
                        listeners.stream().forEach(l -> l.onForwardFinish(context, output));

                        List<Matrix> outputDeltas = IntStream.range(0, outputLayer.outputCount).mapToObj(i ->
                                context.getExcepted().get(i).minus(output.get(i))
                        ).collect(Collectors.toList());

                        context.deltas.put(outputLayer, outputDeltas);
                        outputLayer.backPropagationDelta(context);
                        //System.out.println(context.deltas.get(inputLayer.nextLayer).get(0).getColumn());
                        latch.countDown();
                    });
                }
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                outputLayer.backPropagationUpdate(contexts, .5d);
                listeners.forEach(l -> l.onBatchFinish(contexts));

            }
            listeners.forEach(l -> l.onEpochFinish(epoch));
            //present(epoch);
            //log.info("epoch {}:  {} / {} ", epoch, correct, trainContexts.size());
        });

    }

    public List<Matrix> forward(CnnContext context) {

        context.features.put(inputLayer, context.getInputs());
        return inputLayer.forward(context);

    }

    public void shutdown() throws InterruptedException {
        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);
    }


}
