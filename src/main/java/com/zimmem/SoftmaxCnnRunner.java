package com.zimmem;

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
public class SoftmaxCnnRunner {

    public static void main(String[] args) throws InterruptedException, IOException {
        ConvolutionNeuralNetwork network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(28, 28, 1))
                .addLayer(new CnnConvolutionLayer(5, 5, 4)) // 24
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 12
                .addLayer(new CnnConvolutionLayer(5, 5, 4)) // 8
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max)) // 4
                .addLayer(new CnnConvolutionLayer(4, 4, 10))
                .addLayer(new CnnSoftmaxLayer())
                .addListener(new Stat2LogListener())
                .build();


        try {
            List<MnistImage> trainImages = Mnist.loadImages("/mnist/train-images.idx3-ubyte");
            List<MnistLabel> trainLabels = Mnist.loadLabels("/mnist/train-labels.idx1-ubyte");
            List<CnnTrainInput> inputs = IntStream.range(0, trainImages.size()).mapToObj(i -> {
                MnistImage image = trainImages.get(i);
                List<Matrix> input = Collections.singletonList(image.asMatrix().processUnits(d -> d / 255));
                int label = trainLabels.get(i).getValue();
                List<Matrix> expected = IntStream.range(0, 10).mapToObj(l -> Matrix.single(l == label ? 1d : 0d)).collect(Collectors.toList());
                return new CnnTrainInput(input, expected);
            }).collect(Collectors.toList());

            network.train(inputs.subList(0, 1), 1, 100);

            CnnContext context = new CnnContext();
            context.setInputs(inputs.get(0).getInputs().stream().map(m -> m.processUnits(d -> d/255)).collect(Collectors.toList()));
            List<Matrix> output = network.forward(context);
            System.out.println(output);


        } finally {
            network.shutdown();
        }
    }
}