package com.zimmem.neural.network.bp;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Zimmem on 2016/7/28.
 */
public class Context {

    public Context() {
        activations = new HashMap<>();
        weightedInputs = new HashMap<>();
        deltas = new HashMap<>();
    }

    Map<Layer, double[]> activations;

    Map<Layer, double[]> weightedInputs;

    Map<Layer, double[]> deltas;


}
