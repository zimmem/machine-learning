package com.zimmem.neural.network.cnn;

import java.util.List;

/**
 * Created by Zimmem on 2016/7/31.
 */
public class CnnInputLayer extends  CnnLayer {

    public CnnInputLayer(int outputRow, int outputColumn){
        this.outputRow = outputRow;
        this.outputColumn = outputColumn;
        this.outputCount = 1;
    }


    @Override
    protected void propagation(CnnContext context) {
    }

    @Override
    protected void updateWeightsAndBias(List<CnnContext> contexts, double eta) {

    }
}
