package com.zimmem;

import com.zimmem.cifar.Cifar;
import com.zimmem.cifar.CifarImage;
import com.zimmem.math.ActivationFunction;
import com.zimmem.math.Matrix;
import com.zimmem.mnist.Mnist;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.cnn.*;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/8.
 */
public class Cifar10CnnRunner {

    public static void main(String[] args) throws InterruptedException, IOException {
        ConvolutionNeuralNetwork network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(28, 28, 1))
                .addLayer(new CnnConvolutionLayer(5, 5, 16)) // 28
                .addLayer(new CnnActivationLayer(ActivationFunction.Sigmoid))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 14
                .addLayer(new CnnConvolutionLayer(5, 5, 20)) // 10
                .addLayer(new CnnActivationLayer(ActivationFunction.Sigmoid))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 5
                .addLayer(new CnnConvolutionLayer(4, 4, 10))
                //.addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnSoftmaxLayer())
                .addListener(new Stat2LogListener())
                .build();

        List<CifarImage> cifarImages = Cifar.loadTrandImages();

        List<CnnTrainInput> inputs = cifarImages.stream().map(img -> {
            List<Matrix> expected = IntStream.range(0, 10).mapToObj(i -> new Matrix(new double[][]{new double[]{i == img.getLabel() ? 1 : 0}})).collect(Collectors.toList());
            return new CnnTrainInput(img.asMatrices().stream().map(m -> m.processUnits(d -> d/255)).collect(Collectors.toList()), expected);
        }).collect(Collectors.toList());

        network.train(inputs.subList(0,1000), 1, 100);

        network.shutdown();

        CnnContext context = new CnnContext();
        context.setInputs(inputs.get(0).getInputs().stream().map(m -> m.processUnits(d -> d/255)).collect(Collectors.toList()));
        List<Matrix> output = network.forward(context);
        System.out.println(output);


    }
}