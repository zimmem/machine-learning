package com.zimmem.neural.network.cnn;

import com.zimmem.math.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Zimmem on 2016/7/30.
 */
public class CnnConvolutionLayer extends CnnLayer {


    /**
     * @param kernelRow    过滤器宽
     * @param kernelColumn 过滤器高
     * @param count        生成几个结果
     */
    public CnnConvolutionLayer(int kernelRow, int kernelColumn, int count, Function<Double, Double> activationFunction) {
        this(kernelRow, kernelColumn, 0, 1, count, activationFunction);
    }

    public CnnConvolutionLayer(int kernelRow, int kernelColumn, int pad, int step, int count, Function<Double, Double> activationFunction) {
        this.kernelRow = kernelRow;
        this.kernelColumn = kernelColumn;
        this.pad = pad;
        this.step = step;
        this.activationFunction = activationFunction;
        this.outputCount = count;
    }

    private final int kernelRow;
    private final int kernelColumn;
    List<ConvFilter> filters;
    int pad;
    int step;

    private Function<Double, Double> activationFunction;


    public void init() {
        filters = new ArrayList<>(outputCount);
        for (int i = 0; i < outputCount; i++) {
            // 每个输出一个 filter
            // 第个 filter 针对每个输入 一个 kernel
            filters.add(new ConvFilter(kernelRow, kernelColumn, preLayer.outputCount));
        }
        this.outputRow = (preLayer.outputRow + pad * 2 - kernelRow) / step + 1;
        this.outputColumn = (preLayer.outputColumn + pad * 2 - kernelColumn) / step + 1;
    }

    @Override
    protected void propagation(CnnContext context) {

        List<Matrix> features = new ArrayList<>(outputCount);
        for (int i = 0; i < outputCount; i++) {
            features.add(filters.get(i).conv(context.features.get(preLayer)));
        }
        context.features.put(this, features);

    }


    @Override
    protected void recordDelta(CnnContext context) {

        if (nextLayer instanceof CnnPoolingLayer) {

        } else {
            throw new RuntimeException("not yet implemented.");
        }


    }

    @Override
    protected void updateWeightsAndBias(List<CnnContext> contexts, double eta) {
        IntStream.range(0, filters.size()).forEach(fi -> {
            ConvFilter filter = filters.get(fi);
            List<Matrix> deltas = contexts.stream().map(c -> c.deltas.get(this).get(fi)).collect(Collectors.toList());
            filter.update(deltas);
        });
    }


    class ConvFilter {
        double bias;

        List<Matrix> kernels;


        ConvFilter(int kernelRow, int kernelColumn, int kernelCount) {
            kernels = new ArrayList<>(kernelCount);
            for (int i = 0; i < kernelCount; i++) {
                kernels.add(Matrix.random(kernelRow, kernelColumn, 0.05d));
            }
        }

        Matrix conv(List<Matrix> sources) {

            Matrix result = new Matrix(outputRow, outputColumn);
            for (int i = 0; i < sources.size(); i++) {
                Matrix source = sources.get(i);
                Matrix kernel = kernels.get(i);

                for (int stepRow = 0 - pad; stepRow <= source.getRow() + pad - kernel.getRow(); stepRow += step) {

                    for (int stepColumn = 0 - pad; stepColumn <= source.getColumn() + pad - kernel.getColumn(); stepColumn += step) {

                        int resultRow = (stepRow + pad) / step;
                        int resultColumn = (stepColumn + pad) / step;
                        double tempResult = 0;
                        for (int kernelRow = 0; kernelRow < kernel.getRow(); kernelRow++) {
                            for (int kernelColumn = 0; kernelColumn < kernel.getColumn(); kernelColumn++) {
                                int sourceRow = stepRow + kernelRow;
                                int sourceColumn = stepColumn + kernelColumn;
                                double sourceValue = sourceRow < 0 || sourceColumn < 0 || sourceRow >= source.getRow() || sourceColumn >= source.getColumn() ? 0 : source.getValue(sourceRow, sourceColumn);
                                tempResult += sourceValue * kernel.getValue(kernelRow, kernelColumn);
                            }
                        }
                        result.setValue(resultRow, resultColumn, result.getValue(resultRow, resultColumn) + tempResult);

                    }

                }

            }

            result.processUnits(d -> activationFunction.apply(d + bias));
            return result;
        }


        public void update(List<Matrix> deltas) {



        }
    }
}
