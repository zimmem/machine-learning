package com.zimmem.neural.network.bp;

import com.zimmem.math.Functions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * 一层结点
 * <p>
 * Created by zimmem on 2016/7/26.
 */
public class Layer implements Serializable{

    int size;

    /**
     * 每个神经元的权重对上一层每个输出的权重
     */
    double[][] weights;


    /**
     * 每个神经元计算后的激活值
     */
    //double[] activations;

    /**
     * 偏移量
     */
    double biases[];

    /**
     * 每次训练后的偏差
     */
    //List<double[]> errors;

    /**
     * wx+b 加权输入
     */
    ///List<double[]> weightedInputs;

    //List<double[]> deltas;

    /**
     * 激活函数
     */
    private Function<Double, Double> activationFunction;

    Layer preLayer;

    Layer nextLayer;


    /**
     * @param size 有几个神经元
     */
    public Layer(int size, Function<Double, Double> activationFunction) {
        this.size = size;
        this.activationFunction = activationFunction;
    }

    void init() {
        Random random = new Random();
        if (preLayer != null) {
            weights = new double[size][preLayer.size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < preLayer.size; j++) {
                    weights[i][j] = ( random.nextDouble() - .5) * 0.1;
                }
            }
        }
        biases = new double[size];
        for (int i = 0; i < size; i++) {
            // biases 都为0
            biases[i] = 0d;
        }

    }

    double[] forward(TrainContext context) {

        if (preLayer != null) {
            double[] weightedInputs = new double[size];
            double[] activations = new double[size];

            IntStream.range(0, size).forEach(i -> {
                double weightedInput = IntStream.range(0, preLayer.size).mapToDouble(pi -> context.activations.get(preLayer)[pi] * weights[i][pi]).sum() + biases[i];
                weightedInputs[i] = weightedInput;
                activations[i] = activationFunction.apply(weightedInput);
            });
            context.weightedInputs.put(this, weightedInputs);
            context.activations.put(this, activations);
        }

        if (nextLayer != null) {
            return nextLayer.forward(context);
        } else {
            return context.activations.get(this);
        }

    }

    /**
     * 反向传播偏差
     *
     * @param delta
     */
    void backPropagationDelta(TrainContext context, double[] delta) {
        context.deltas.put(this, delta);

        if (preLayer != null && preLayer.preLayer != null) {
            double[] preDelta = new double[preLayer.size];
            IntStream.range(0, preDelta.length).forEach(i -> {
                IntStream.range(0, size).forEach(j -> {
                    preDelta[i] = weights[j][i] * delta[j] * Functions.SigmoidDerivative.apply(context.weightedInputs.get(preLayer)[i]);
                });
            });
            preLayer.backPropagationDelta(context, preDelta);
        }
    }

    /**
     *
     * @param contexts
     * @param eta 学习速率
     */
    void backPropagationUpdate(List<TrainContext> contexts, double eta) {

        if (preLayer == null ) {
            return;
        }

        // 更新权重
        // 更新 biases
        IntStream.range(0, biases.length).forEach(i -> {
            //biases[i] -= deltas.stream().mapToDouble( a -> a[i]).sum()* p / deltas.size();
            biases[i] -= contexts.stream().mapToDouble(c -> c.deltas.get(this)[i]).sum() * eta / contexts.size();
        });

        IntStream.range(0, weights.length).forEach(i -> {
            IntStream.range(0, preLayer.size).forEach(j -> {
                weights[i][j] -= contexts.stream().mapToDouble(c -> c.deltas.get(this)[i] * c.activations.get(preLayer)[j]).sum() * eta / contexts.size();
            });
        });

        preLayer.backPropagationUpdate(contexts, eta);
    }

    public void cleanTrain() {
//        errors.clear();
//        weightedInputs.clear();
//        deltas.clear();
    }
}
