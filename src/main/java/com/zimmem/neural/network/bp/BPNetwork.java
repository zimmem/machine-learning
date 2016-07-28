package com.zimmem.neural.network.bp;

import com.zimmem.math.Functions;
import com.zimmem.mnist.*;
import com.zimmem.neural.network.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BPNetwork implements Network , Serializable{

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
            int batchCorrect;
            double verifyRate = 0d;
            for (int batch = 0; batch < images.size(); batch += batchSize) {

                if (batch % 1000 == 0) {
                    // 每训练1000个数据， 拿最后1000个数据做下验证
                    verifyRate = verify(images.subList(images.size() - 10000, images.size()), labels.subList(labels.size() - 10000, labels.size()));
                }

                batchCorrect = 0;
                int batchNum = 0;
                List<TrainContext> contexts = new ArrayList<>(batchSize);
                for (int index = batch; index < batch + batchSize; index++) {
                    MnistImage image = images.get(index);
                    MnistLabel label = labels.get(index);
                    TrainContext context = new TrainContext();
                    contexts.add(context);
                    double[] output = forward(context, image);
                    double[] expect = new double[output.length];
                    expect[label.getValue()] = 1;
                    if (!Double.isNaN(output[label.getValue()]) && Objects.equals(Arrays.stream(output).max().getAsDouble(), output[label.getValue()])) {
                        //System.out.println(Arrays.toString(output));
                        batchCorrect++;
                    }
                    batchNum++;

                    // 输出与期望的偏差
                    double[] error = new double[output.length];
                    for (int i = 0; i < output.length; i++) {
                        error[i] = output[i] - expect[i];
                    }

                    // 计算 output 层偏差
                    double[] deltas = new double[outputLayer.size];
                    IntStream.range(0, outputLayer.size).forEach(i -> deltas[i] = error[i] * Functions.SigmoidDerivative.apply(output[i]));
                    outputLayer.backPropagationDelta(context, deltas);

                }
                correct += batchCorrect;
                log.debug("batch {} : {}/{} - total {}/{} = {} ", batch / batchSize + 1, batchCorrect, batchNum, correct, batch + batchSize, (double) correct / (batch + batchSize));

                outputLayer.backPropagationUpdate(contexts, 0.8);
                //resetTrainData();


            }
            log.info("repeat {}:  {} / {} ", repeat, correct, images.size());

        }

        log.info("train finish, cast {} ms", System.currentTimeMillis() - start);
    }

    private double verify(List<MnistImage> mnistImages, List<MnistLabel> mnistLabels) {
        AtomicInteger collect = new AtomicInteger(0);

        IntStream.range(0, mnistImages.size()).forEach(i ->{
            double[] output = forward(new TrainContext(), mnistImages.get(i));
            double max = Arrays.stream(output).max().getAsDouble();
            if (Objects.equals(max, output[mnistLabels.get(i).getValue()]) && !Double.isNaN(max)) {
                collect.incrementAndGet();
            }
        });

        double rate = collect.doubleValue() / mnistImages.size();
        log.info("valified {}/{} = {}", collect, mnistImages.size(), rate);
        return rate;
    }

    private void resetTrainData() {
        Layer current = outputLayer;
        while (current != null) {
            current.cleanTrain();
            current = current.nextLayer;
        }
    }

    /**
     * 反向传播
     */
    private void backPropagation() {

        double p = 0.8; // 学习速率

//        for (int L = layers.size() - 1; L > 0; L--) {
//            // 第L层
//
//            Layer layer = layers.get(L);
//            IntStream.range(0, layer.size).forEach(j -> {
//                // 第j 个结点
//
//                //代价函数C
//                double Cj = layer.errors.stream().mapToDouble(error -> Math.pow(error[j], 2)).sum() * 0.5d / layer.errors.size();
//
//                // 计算错误量
//                double delta_bias = IntStream.range(0, layer.errors.size()).mapToDouble(i -> {
//                    //针对每个训练样本
//                    return layer.errors.get(i)[j] * Functions.SigmoidDerivative.apply(layer.weightedInputs.get(i)[j]);
//                }).sum() * p / layer.errors.size();
//                layer.biases[j] -= delta_bias;
//
//                IntStream.range(0, layer.preLayer.size).forEach(i -> {
//                    //针对上层每个输出的 weight
//                    double delta_weight = layer.preLayer.activations[i] * delta_bias;
//                    layer.weights[j][i] -= delta_weight;
//                });
//
//
//            });
//            //计算上层偏差
//            Layer preLayer = layers.get(L - 1);
//            double[] errors = new double[preLayer.size];
//            IntStream.range(0, preLayer.size).forEach(j -> {
//                errors[j] = 0 * Functions.SigmoidDerivative(preLayer.weightedInputs[j]);
//            });
//            preLayer.errors.add(errors);
//            layer.errors.clear();
//        }

    }


    /**
     * 分类
     *
     *
     * @param context
     * @param image
     * @return
     */
    public double[] forward(TrainContext context, MnistImage image) {

        double[] input = new double[image.getValues().length];
        for (int i = 0; i < image.getValues().length; i++) {
            input[i] = (double) image.getValues()[i];
        }
        context.activations.put(inputLayer, input);
        return inputLayer.forward(context);

    }

}
