package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/8.
 */
public class CnnSoftmaxLayer extends CnnLayer {

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

    /**
     * https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s4.html
     *
     * @param context
     * @return
     */
    @Override
    protected List<Matrix> calculatePreDelta(CnnTrainContext context) {

//        return context.deltas.get(this).stream().map(m -> m.processUnits(d -> -d)).collect(Collectors.toList());
        return context.deltas.get(this);
    }

    @Override
    protected void updateWeightsAndBias(List<CnnTrainContext> contexts, double eta) {

    }

    private List<Double> softmax(List<Double> input) {
        double max = input.stream().mapToDouble(d -> d).max().getAsDouble();

        List<Double> powers = input.stream().map(d -> Math.exp(max < 700 ? d : d - max + 700)).collect(Collectors.toList());
        double sum = powers.stream().mapToDouble(d -> d).sum();
        return powers.stream().map(d -> d / sum).collect(Collectors.toList());
    }

    public static void main(String[] args) {
        System.out.println(Math.exp(3000));
        CnnSoftmaxLayer soft = new CnnSoftmaxLayer();
        List<Double> result = soft.softmax(Arrays.asList(2500d, 2600d, 2700d, 2700d,2500d, 2600d, 2700d, 2701d,2500d, 2600d));
        System.out.println(result);


        System.out.println(Math.log(Double.MAX_VALUE));
        System.out.println(Math.exp(Double.MIN_VALUE));
    }
}
