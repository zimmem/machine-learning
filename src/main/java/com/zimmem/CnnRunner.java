package com.zimmem;

import com.zimmem.math.Matrix;
import com.zimmem.mnist.Mnist;
import com.zimmem.mnist.MnistImage;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.cnn.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class CnnRunner {

    public static void main(String[] args) throws IOException {
        ConvolutionNeuralNetwork network = NetworkBuilder.cnn()
                .setInputConverter(i -> {
                    MnistImage image = (MnistImage) i;
                    Matrix m = new Matrix(28, 28);
                    for (int c = 0; c < 28; c++) {
                        for (int r = 0; r < 28; r++) {
                            m.setValue(r, c, (image.getValues()[28 * c + r] & 0xff) > 0 ? 1d : 0d);
                        }
                    }
                    return Arrays.asList(m);
                })
                .addLayer(new CnnInputLayer(28, 28))
                .addLayer(new CnnConvolutionLayer(5, 5, 6))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max))
                .addLayer(new CnnConvolutionLayer(5, 5, 16))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max))
                .addLayer(new CnnConvolutionLayer(4, 4, 120))
                .build();

        //network.train(Mnist.loadImages("/mnist/train-images.idx3-ubyte").subList(0, 1), Mnist.loadLabels("/mnist/train-labels.idx1-ubyte"), 20, 10);

        List<Matrix> result = network.forward(new CnnContext(), Mnist.loadImages("/mnist/train-images.idx3-ubyte").get(0));
        System.out.println(result.size());
        result.stream().forEach( m ->{
            System.out.println(m );
        });

    }

}

