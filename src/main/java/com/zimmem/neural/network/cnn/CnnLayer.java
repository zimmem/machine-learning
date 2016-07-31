package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.List;

/**
 * Created by Zimmem on 2016/7/30.
 */
public abstract class CnnLayer {

    CnnLayer preLayer;

    CnnLayer nextLayer;

    int outputCount;

    int outputRow;

    int outputColumn;



    List<Matrix> forward(CnnContext context) {

        propagation(context);
        if (nextLayer != null) {
            return nextLayer.forward(context);
        } else {
            return context.features.get(this);
        }
    }

    void init() {
    }

    void backPropagationUpdate(List<CnnContext> contexts, double eta) {

        updateWeightsAndBias(contexts, eta);

        if (preLayer != null) {
            preLayer.backPropagationUpdate(contexts, eta);
        }

    }


    protected abstract void propagation(CnnContext context);

    protected  abstract void updateWeightsAndBias(List<CnnContext> contexts, double eta) ;
}
