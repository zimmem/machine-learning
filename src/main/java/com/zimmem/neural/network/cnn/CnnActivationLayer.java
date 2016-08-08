package com.zimmem.neural.network.cnn;

import com.zimmem.math.ActivationFunction;
import com.zimmem.math.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/4.
 */
public class CnnActivationLayer extends CnnLayer {

    private ActivationFunction activationFunction;

    public CnnActivationLayer(ActivationFunction function) {

        activationFunction = function;
    }

    @Override
    void init() {
        this.outputCount = preLayer.outputCount;
        this.outputColumn = preLayer.outputColumn;
        this.outputRow = preLayer.outputRow;
    }

    @Override
    protected void propagation(CnnContext context) {
        List<Matrix> preFeatures = context.features.get(preLayer);
        List<Matrix> features = preFeatures.stream().map(m -> m.processUnits(activationFunction::run)).collect(Collectors.toList());
        context.features.put(this, features);
    }

    @Override
    protected List<Matrix> calculatePreDelta(CnnTrainContext context) {
        return IntStream.range(0, outputCount).mapToObj(i -> {
            Matrix predelta = context.deltas.get(this).get(i);
            Matrix result = Matrix.zeros(predelta.getRow(), predelta.getColumn());
            for (int r = 0; r < predelta.getRow(); r++) {
                for (int c = 0; c < predelta.getColumn(); c++) {
                    result.setValue(r, c, predelta.getValue(r, c) * activationFunction.runDerivative(context.features.get(preLayer).get(i).getValue(r, c)));
                }
            }
            return result;
        }).collect(Collectors.toList());
    }

    @Override
    protected void updateWeightsAndBias(List<CnnTrainContext> contexts, double eta) {
        // do nothing
    }
}
