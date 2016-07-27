package com.zimmem.neural.network.bp;

import com.zimmem.ThreadContext;
import com.zimmem.math.Functions;

import java.io.InputStream;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * 一层结点
 * <p>
 * Created by zimmem on 2016/7/26.
 */
public class Layer {

    int size;

    /**
     * 每个神经元的权重对上一层每个输出的权重
     */
    double[][] weights;


    /**
     * 每个神经元计算后的激活值
     */
    double[] activations;

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
    Map<Integer, double[]> weightedInputs;

    Map<Integer, double[]> deltas;

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
                    weights[i][j] = (random.nextDouble()-0.5) * 0.1d;
                }
            }
        }
        biases = new double[size];
        for (int i = 0; i < size; i++) {
            // biases 都为0
            biases[i] = 0d;
        }
        activations = new double[size];
        //errors = new ArrayList<>();
        weightedInputs = new HashMap<>();
        deltas = new HashMap<>();

    }



    double[] forward() {

        if (preLayer != null) {
            double[] weightedInputs = new double[size];
            IntStream.range(0, size).forEach(i -> {
                double weightedInput = IntStream.range(0, preLayer.size).mapToDouble(pi -> preLayer.activations[pi] * weights[i][pi]).sum() + biases[i];
                weightedInputs[i] = weightedInput;
                activations[i] = activationFunction.apply(weightedInput);
            });
            synchronized (weightedInputs){
                Optional.ofNullable(ThreadContext.get("network.bp.train.batch.index")).ifPresent( o ->{
                    this.weightedInputs.put((Integer) o, weightedInputs);
                });

            }
        }

        if (nextLayer != null) {
            return nextLayer.forward();
        } else {
            return this.activations;
        }

    }

    /**
     * 反向传播偏差
     *
     * @param delta
     */
    void backPropagationDelta(double[] delta) {
        synchronized (deltas){
            int index = (int) ThreadContext.get("network.bp.train.batch.index");
            this.deltas.put (index, delta);
        }

        if (preLayer != null) {
            double[] preDelta = new double[preLayer.size];
            IntStream.range(0, preDelta.length).forEach(i -> {
                IntStream.range(0, size).forEach(j -> {
                    preDelta[i] += weights[j][i] * Functions.SigmoidDerivative.apply(activations[j]);
                });
            });
            preLayer.backPropagationDelta(preDelta);
        }
    }

    /**
     *
     * @param p 训练速度
     */
    void backPropagationUpdate(double p) {

        if(preLayer == null ){
            return ;
        }

        // 更新权重
        // 更新 biases
        IntStream.range(0, size).forEach(i -> {
            biases[i] -= deltas.entrySet().stream().mapToDouble( a -> a.getValue()[i]).sum()* p / deltas.size();
        });

        // i -> 当前层第i个神经元， j -》 上一层第j 个神经元
        IntStream.range(0, size).forEach(i ->{
            IntStream.range(0, preLayer.size).forEach(j ->{
                weights[i][j] -=  deltas.entrySet().stream().mapToDouble(entry -> entry.getValue()[i] * preLayer.activations[j]).sum();
            });
        });

        preLayer.backPropagationUpdate(p);
    }

    public void cleanTrain() {
        //errors.clear();
        weightedInputs.clear();
        deltas.clear();
    }
}
