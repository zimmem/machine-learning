package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by zimmem on 2016/8/8.
 */
public class CnnSoftmaxLayer extends  CnnLayer {

    @Override
    void init() {
        this.outputCount = preLayer.outputCount;
        this.outputColumn = preLayer.outputColumn;
        this.outputRow = preLayer.outputRow;
    }

    @Override
    protected void propagation(CnnContext context) {
        List<Matrix> preFeatures = context.features.get(preLayer);
        List<Double> values = preFeatures.stream().map(d -> d.getValue(0, 0)).collect(Collectors.toList());
        List<Matrix> features = softmax(values).stream().map(Matrix::single).collect(Collectors.toList());
        context.features.put(this, features);
    }

    @Override
    protected List<Matrix> calculatePreDelta(CnnTrainContext context) {
        return null;
    }

    @Override
    protected void updateWeightsAndBias(List<CnnTrainContext> contexts, double eta) {

    }

    private List<Double> softmax(List<Double> input){
        List<Double> powers = input.stream().map(d -> Math.pow(Math.E, d)).collect(Collectors.toList());
        double sum = powers.stream().mapToDouble(d -> d).sum();
        return powers.stream().map(d -> d / sum).collect(Collectors.toList());
    }

    public static void main(String[] args){
        CnnSoftmaxLayer soft = new CnnSoftmaxLayer();
        List<Double> result = soft.softmax(Arrays.asList(1d, 2d, 3d, 10d));
        System.out.println(result);
    }
}
