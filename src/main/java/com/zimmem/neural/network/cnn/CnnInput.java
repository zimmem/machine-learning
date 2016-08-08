package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.List;

/**
 * Created by zimmem on 2016/8/8.
 */
public class CnnInput {

    public CnnInput(List<Matrix> inputs) {
        this.inputs = inputs;
    }

    private List<Matrix> inputs ;

    public List<Matrix> getInputs() {
        return inputs;
    }

    public void setInputs(List<Matrix> inputs) {
        this.inputs = inputs;
    }
}
