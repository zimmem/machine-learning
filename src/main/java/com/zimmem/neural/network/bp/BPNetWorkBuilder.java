package com.zimmem.neural.network.bp;

import com.zimmem.neural.network.NetworkBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BPNetWorkBuilder  extends NetworkBuilder<BPNetwork>{

    private List<Layer> layers;

    public BPNetWorkBuilder addLayer(Layer layer){
        layers = Optional.ofNullable(layers).orElse(new ArrayList<>());
        layers.add(layer);
        return this;
    }

    public BPNetwork build() {
        BPNetwork network = new BPNetwork();
        network.layers = this.layers;
        return network;
    }
}
