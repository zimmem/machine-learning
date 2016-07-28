package com.zimmem.neural.network.bp;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by zimmem on 2016/7/28.
 */
public class TrainContext {

    public TrainContext() {
        weightedInputs = new HashMap<>();
        deltas = new HashMap<>();
        activations = new HashMap<>();
    }

    Map<Layer, double[]> weightedInputs;

    Map<Layer, double[]> deltas;

    Map<Layer, double[]> activations;
}
