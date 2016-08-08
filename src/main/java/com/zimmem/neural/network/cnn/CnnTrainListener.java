package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.List;

/**
 * Created by zimmem on 2016/8/8.
 */
public interface CnnTrainListener {

    void onForwardFinish(CnnTrainContext context,  List<Matrix> output);

    void onBatchFinish(List<CnnTrainContext> contexts);

    void onEpochFinish(int epoch);

}
