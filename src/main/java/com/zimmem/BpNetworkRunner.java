package com.zimmem;

import com.zimmem.math.Functions;
import com.zimmem.mnist.Mnist;
import com.zimmem.neural.network.NetworkBuilder;
import com.zimmem.neural.network.bp.BPNetwork;
import com.zimmem.neural.network.bp.Layer;

import java.io.IOException;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BpNetworkRunner {

    public static void main(String[] args) throws IOException {
        BPNetwork network = NetworkBuilder.bp()
                .addLayer(new Layer(28 * 28, null))
                .addLayer(new Layer(16, Functions.sigmoid))
                .addLayer(new Layer(10,  Functions.sigmoid))
                .build();

        network.train(Mnist.loadImages("/mnist/t10k-images.idx3-ubyte"), 10, 1);

    }
}
