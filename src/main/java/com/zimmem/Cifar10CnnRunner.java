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
                .addLayer(new CnnInputLayer(28, 28, 1))
                .addLayer(new CnnConvolutionLayer(5, 5, 6)) // 28
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 14
                .addLayer(new CnnConvolutionLayer(5, 5, 16)) // 10
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 5
                .addLayer(new CnnConvolutionLayer(4, 4, 10))
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
            List<MnistImage> trainImages = Mnist.loadImages("/mnist/train-images.idx3-ubyte");
            List<MnistLabel> trainLabels = Mnist.loadLabels("/mnist/train-labels.idx1-ubyte");
            List<CnnTrainInput> inputs = IntStream.range(0, trainImages.size()).mapToObj(i -> {
                MnistImage image = trainImages.get(i);
                List<Matrix> input = Collections.singletonList(image.asMatrix());
                int label = trainLabels.get(i).getValue();
                List<Matrix> expected = IntStream.range(0, 10).mapToObj(l -> Matrix.single(l == label ? 1d : 0d)).collect(Collectors.toList());
                return new CnnTrainInput(input, expected);
            }).collect(Collectors.toList());

            network.train(inputs.subList(0,1), 1, 10000);

        } finally {
            network.shutdown();
        }
    }
}