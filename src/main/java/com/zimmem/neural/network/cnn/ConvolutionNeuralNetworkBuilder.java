package com.zimmem.neural.network.cnn;

import com.zimmem.neural.network.NetworkBuilder;

import java.util.function.Function;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class ConvolutionNeuralNetworkBuilder extends NetworkBuilder<ConvolutionNeuralNetwork> {

    private CnnLayer firstLayer;

    private CnnLayer latestLayer;



    Function inputConverter;


    public ConvolutionNeuralNetworkBuilder addLayer(CnnLayer layer) {

        if (firstLayer == null) {
            firstLayer = layer;
            latestLayer = layer;
        } else {
            layer.preLayer = latestLayer;
            latestLayer.nextLayer = layer;
            latestLayer = layer;
        }
        return this;
    }

    public ConvolutionNeuralNetworkBuilder setInputConverter(Function inputConverter) {
        this.inputConverter = inputConverter;
        return this;
    }


    @Override
    public ConvolutionNeuralNetwork build() {

        CnnLayer current = firstLayer;
        while(current != null ){
            current.init();
            current = current.nextLayer;
        }

        ConvolutionNeuralNetwork network = new ConvolutionNeuralNetwork();
        network.inputLayer = firstLayer;
        network.outputLayer = latestLayer;
        network.inputConverter = this.inputConverter;
        return network;
    }
}
