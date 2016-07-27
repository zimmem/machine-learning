package com.zimmem.neural.network.bp;

import com.zimmem.neural.network.NetworkBuilder;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BPNetWorkBuilder extends NetworkBuilder<BPNetwork> {

    Layer inputLayer;

    Layer outputLayer;

    public BPNetWorkBuilder addLayer(Layer layer) {
        if (inputLayer == null) {
            inputLayer = layer;
            outputLayer = layer;
        } else {
            outputLayer.nextLayer = layer;
            layer.preLayer = outputLayer;
            outputLayer = layer;
        }
        return this;
    }

    public BPNetwork build() {
        BPNetwork network = new BPNetwork();
        network.inputLayer = inputLayer;
        network.outputLayer = outputLayer;
        Layer current = inputLayer;
        while(current != null ){
            current.init();
            current = current.nextLayer;
        }
        return network;
    }
}
