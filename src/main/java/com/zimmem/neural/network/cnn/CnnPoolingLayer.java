package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class CnnPoolingLayer extends CnnLayer {

    private List<PoolingFilter> filters;

    private int filterRow;

    private int filterColumn;

    private Function<Matrix, Double> strategy;

    public CnnPoolingLayer(int filterRow, int filterColumn, Function<Matrix, Double> strategy) {
        this.filterRow = filterRow;
        this.filterColumn = filterColumn;
        this.strategy = strategy;
    }

    void init() {
        this.outputColumn = preLayer.outputColumn / filterColumn;
        this.outputRow = preLayer.outputRow / filterRow;
        this.outputCount = preLayer.outputCount;
        filters = new ArrayList<>(outputCount);
        for (int i = 0; i < outputCount; i++) {
            filters.add(new PoolingFilter());
        }
    }

    @Override
    protected void propagation(CnnContext context) {
        List<Matrix> output = new ArrayList<>(outputCount);
        List<Matrix> preOutput = context.features.get(preLayer);
        List<Matrix> weightedInputs = new ArrayList<>(outputCount);
        IntStream.range(0, outputCount).forEach(i -> {
            Matrix pooled = filters.get(i).pooling(preOutput.get(i));
            output.add(pooled);
            weightedInputs.add(pooled);
        });
        context.weightedInputs.put(this, weightedInputs);
        context.features.put(this, output);
    }

    @Override
    protected List<Matrix> calculatePreDelta(CnnContext context) {

        List<Matrix> deltas = context.deltas.get(this);
        List<Matrix> preDeltas = new ArrayList<>(preLayer.outputCount);
        // pool 层输出与输入数是一致的， 传播没有交叉， 所以直接把当前层的残卷扩展到上一次特征图大小
        for (int i = 0; i < preLayer.outputCount; i++) {
            Matrix delta = Matrix.zeros(preLayer.outputRow, preLayer.outputColumn);
            for (int r = 0; r < preLayer.outputRow; r++) {
                for (int c = 0; c < preLayer.outputColumn; c++) {
                    delta.setValue(r, c, deltas.get(i).getValue(r / this.filterRow, c / this.filterColumn));
                }
            }
            preDeltas.add(delta);
        }
        return preDeltas;

    }

    @Override
    protected void updateWeightsAndBias(List<CnnContext> contexts, double eta) {

    }


    class PoolingFilter {


        Matrix pooling(Matrix source) {

            Matrix target = Matrix.zeros(outputRow, outputColumn);

            for (int r = 0; r < source.getRow() / filterRow; r++) {
                for (int c = 0; c < source.getColumn() / filterColumn; c++) {
                    double[][] sub = new double[filterRow][filterColumn];
                    for (int tr = 0; tr < filterRow; tr++) {
                        for (int tc = 0; tc < filterColumn; tc++) {
                            sub[tr][tc] = source.getValue(tr + r * filterRow, tc + c * filterColumn);
                        }
                    }
                    target.setValue(r, c, strategy.apply(new Matrix(sub)));
                }
            }
            return target;
        }
    }

    public static class Strategy {

        public static Function<Matrix, Double> Max = m -> {
            double max = m.getValue(0, 0);
            for (int r = 0; r < m.getRow(); r++) {
                for (int c = 0; c < m.getColumn(); c++) {
                    max = Math.max(max, m.getValue(r, c));
                }
            }
            return max;
        };

        static public Function<Matrix, Double> Min = m -> {
            double min = m.getValue(0, 0);
            for (int r = 0; r < m.getRow(); r++) {
                for (int c = 0; c < m.getColumn(); c++) {
                    min = Math.min(min, m.getValue(r, c));
                }
            }
            return min;
        };

        static public Function<Matrix, Double> Sum = m -> {
            double sum = 0;
            for (int r = 0; r < m.getRow(); r++) {
                for (int c = 0; c < m.getColumn(); c++) {
                    sum += m.getValue(r, c);
                }
            }
            return sum;
        };

        static public Function<Matrix, Double> Means = m -> Sum.apply(m) / (m.getColumn() * m.getRow());

    }

    public static void main(String[] args) {
        Matrix random = Matrix.random(8, 8, 10);
        CnnPoolingLayer layer = new CnnPoolingLayer(2, 2, Strategy.Max);
        layer.outputRow = 4;
        layer.outputColumn = 4;
        Matrix pooled = layer.new PoolingFilter().pooling(random);
        System.out.println(random);
        System.out.println(pooled);
    }


}
