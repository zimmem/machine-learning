package com.zimmem.neural.network.bp;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * 一层结点
 * <p>
 * Created by zimmem on 2016/7/26.
 */
public class Layer {

    /**
     * 每个神经元的权重
     */
    double[] weights;


    /**
     * 每个神经元计算后的值
     */
    double[] values;

    /**
     * 偏移量
     */
    private int bias;

    private Function<Double, Double> activationFunction;


    /**
     * @param n 有几个神经元
     */
    public Layer(int n, Function<Double, Double> activationFunction) {
        Random random = new Random();
        weights = new double[n];
        for (int i = 0 ; i < n ; i ++ ){
            weights[i] = random.nextDouble() - 0.5d;
        }
        values = new double[n];
        this.activationFunction = activationFunction;
    }

    public void spread(Layer preLayer) {

        for(int i = 0; i < values.length; i ++ ){
            int finalI = i;
            double sum = Arrays.stream(preLayer.values).map(v -> v * weights[finalI]).sum();
            values[i] = activationFunction.apply(sum + bias);
        }

    }
}
