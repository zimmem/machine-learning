package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zimmem on 2016/8/8.
 */
public class CnnTrainContext extends  CnnContext {
    /**
     * 每层每个输出 map 的 delta
     */
    Map<CnnLayer, List<Matrix>> deltas = new HashMap<>();

    private List<Matrix> excepted;

    public List<Matrix> getExcepted() {
        return excepted;
    }

    public void setExcepted(List<Matrix> excepted) {
        this.excepted = excepted;
    }
}
