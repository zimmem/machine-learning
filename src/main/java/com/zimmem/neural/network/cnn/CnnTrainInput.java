package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.io.Serializable;
import java.util.List;

/**
 * Created by zimmem on 2016/8/8.
 */
public class CnnTrainInput extends  CnnInput  implements Serializable {


    public CnnTrainInput(List<Matrix> inputs, List<Matrix> expected) {
        super(inputs);
        this.expected = expected;
    }

    private List<Matrix> expected ;

    public List<Matrix> getExpected() {
        return expected;
    }

    public void setExpected(List<Matrix> expected) {
        this.expected = expected;
    }
}
