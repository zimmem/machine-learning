package com.zimmem;

import com.zimmem.math.Functions;
import com.zimmem.mnist.Mnist;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.bp.BPNetwork;
import com.zimmem.neural.network.bp.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BpNetworkRunner {

    private static Logger log = LoggerFactory.getLogger(BpNetworkRunner.class);

    public static void main(String[] args) throws IOException {
        BPNetwork network = NetworkBuilder.bp()
                .addLayer(new Layer(28 * 28, null))
                .addLayer(new Layer(50, Functions.Sigmoid))
                //.addLayer(new Layer(30, Functions.Sigmoid))
                .addLayer(new Layer(10, Functions.Sigmoid))
                .build();

        network.train(Mnist.loadImages("/mnist/train-images.idx3-ubyte"), Mnist.loadLabels("/mnist/train-labels.idx1-ubyte"), 50, 1);

        // test
        List<MnistImage> testImages = Mnist.loadImages("/mnist/t10k-images.idx3-ubyte");
        List<MnistLabel> testLabels = Mnist.loadLabels("/mnist/t10k-labels.idx1-ubyte");
        int collect = 0;
        int current = 0;
        for (int i = 0; i < testImages.size(); i++) {
            current++;
            if (current % 100 == 0) {
                log.info("testing mnist : {}/{} - {}", collect, current, (double) collect / current);
                double[] output = network.forward(testImages.get(i));
                if (Objects.equals(Arrays.stream(output).max().getAsDouble(), output[testLabels.get(i).getValue()]) && !Double.isNaN(output[testLabels.get(i).getValue()])) {
                    collect++;
                }
            }
        }


    }
}
