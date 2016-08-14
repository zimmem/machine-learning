package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class CnnContext {

    public Map<CnnLayer, List<Matrix>> features = new HashMap<>();

    private List<Matrix> inputs ;

    public List<Matrix> getInputs() {
        return inputs;
    }

    public void setInputs(List<Matrix> inputs) {
        this.inputs = inputs;
    }
}
