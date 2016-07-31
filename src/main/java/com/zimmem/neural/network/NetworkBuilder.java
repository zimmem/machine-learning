package com.zimmem.neural.network;

import com.zimmem.neural.network.bp.BPNetWorkBuilder;
import com.zimmem.neural.network.cnn.ConvolutionNeuralNetworkBuilder;

/**
 * Created by zimmem on 2016/7/26.
 */
public abstract class NetworkBuilder<T extends  Network> {

    public static BPNetWorkBuilder bp(){
        return new BPNetWorkBuilder();
    }

    public static ConvolutionNeuralNetworkBuilder cnn() {return new ConvolutionNeuralNetworkBuilder();}

    public abstract  T build();
}
