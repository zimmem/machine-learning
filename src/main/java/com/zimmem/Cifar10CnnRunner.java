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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/8.
 */
public class Cifar10CnnRunner {

    public static void main(String[] args) throws InterruptedException, IOException {
        ConvolutionNeuralNetwork network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(32, 32,3))
                .addLayer(new CnnConvolutionLayer(5, 5, 6)) // 28
                .addLayer(new CnnActivationLayer(ActivationFunction.Sigmoid))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 14
                .addLayer(new CnnConvolutionLayer(5, 5, 20)) // 10
                .addLayer(new CnnActivationLayer(ActivationFunction.Sigmoid))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 5
                .addLayer(new CnnConvolutionLayer(5, 5, 10))
                .addLayer(new CnnSoftmaxLayer())
                .addListener(new Stat2LogListener())
                .build();

//        List<CifarImage> cifarImages = Cifar.loadTrandImages();
//
//        List<CnnTrainInput> inputs = cifarImages.stream().map(img -> {
//            List<Matrix> expected = IntStream.range(0, 10).mapToObj(i -> new Matrix(new double[][]{new double[]{i == img.getLabel() ? 1 : 0}})).collect(Collectors.toList());
//            return new CnnTrainInput(img.asMatrices(), expected);
//        }).collect(Collectors.toList());
//
//        network.train(inputs, 50, 10);
//
//        network.shutdown();

        try {
            List<CifarImage> cifarImages = Cifar.loadTrandImages();
            List<CnnTrainInput> inputs = IntStream.range(0, cifarImages.size()).mapToObj(i -> {
                CifarImage image = cifarImages.get(i);
                int label = image.getLabel();
                List<Matrix> expected = IntStream.range(0, 10).mapToObj(l -> Matrix.single(l == label ? 1d : 0d)).collect(Collectors.toList());
                return new CnnTrainInput(image.asMatrices(), expected);
            }).collect(Collectors.toList());

            network.train(inputs.subList(0,10000), 20, 10000);

        } finally {
            network.shutdown();
        }
    }
}