package com.zimmem;

import com.zimmem.cifar.Cifar;
import com.zimmem.cifar.CifarImage;
import com.zimmem.math.ActivationFunction;
import com.zimmem.math.Matrix;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.cnn.*;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/8.
 */
public class Cifar10CnnRunner {

    public static void main(String[] args) throws InterruptedException {
        ConvolutionNeuralNetwork network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(32, 32, 3))
                .addLayer(new CnnConvolutionLayer(5, 5, 6)) // 28
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 14
                .addLayer(new CnnConvolutionLayer(5, 5, 16)) // 10
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 5
                .addLayer(new CnnConvolutionLayer(5, 5, 10))
                .addLayer(new CnnSoftmaxLayer())
                .addListener(new Stat2LogListener())
                .build();

        List<CifarImage> cifarImages = Cifar.loadTrandImages();

        List<CnnTrainInput> inputs = cifarImages.stream().map(img -> {
            List<Matrix> expected = IntStream.range(0, 10).mapToObj(i -> new Matrix(new double[][]{new double[]{i == img.getLabel() ? 1 : 0}})).collect(Collectors.toList());
            return new CnnTrainInput(img.asMatrices(), expected);
        }).collect(Collectors.toList());

        network.train(inputs, 20, 10);

        network.shutdown();
    }
}