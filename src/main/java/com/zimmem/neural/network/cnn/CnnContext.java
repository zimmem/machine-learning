package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class CnnContext {

    Map<CnnLayer, List<Matrix>> features = new HashMap<>();

    Map<CnnLayer, List<Matrix>> weightedInputs = new HashMap<>();

    /**
     * 每层每个输出 map 的 delta
     */
    Map<CnnLayer, List<Matrix>> deltas = new HashMap<>();

}
