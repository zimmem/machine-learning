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

    void backPropagationDelta(CnnTrainContext context){

        //第一层不用计算残差
        if(preLayer != null && preLayer.preLayer != null){
            context.deltas.put(preLayer,  calculatePreDelta(context));
            preLayer.backPropagationDelta(context);
        }
    }


    /**
     *
     * @param contexts
     * @param eta 学习速率
     */
    void backPropagationUpdate(List<CnnTrainContext> contexts, double eta) {

        updateWeightsAndBias(contexts, eta);

        // 第一层不用更新
        if (preLayer != null) {
            preLayer.backPropagationUpdate(contexts, eta);
        }

    }

    protected abstract void propagation(CnnContext context);

    /**
     * 计算上层残差
     * @param context
     */
    protected abstract List<Matrix> calculatePreDelta(CnnTrainContext context);

    protected abstract void updateWeightsAndBias(List<CnnTrainContext> contexts, double eta);
}
