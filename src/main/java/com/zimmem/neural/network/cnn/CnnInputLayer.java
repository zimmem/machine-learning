package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.List;

/**
 * Created by Zimmem on 2016/7/31.
 */
public class CnnInputLayer extends  CnnLayer {

    public CnnInputLayer(int outputRow, int outputColumn, int outputCount){
        this.outputRow = outputRow;
        this.outputColumn = outputColumn;
        this.outputCount = outputCount;
    }


    @Override
    protected void propagation(CnnContext context) {
    }

    @Override
    protected List<Matrix> calculatePreDelta(CnnTrainContext context) {
        // donothing

        return null;
    }

    @Override
    protected void updateWeightsAndBias(List<CnnTrainContext> contexts, double eta) {

    }
}
