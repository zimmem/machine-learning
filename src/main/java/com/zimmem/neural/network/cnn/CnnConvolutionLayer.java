package com.zimmem.neural.network.cnn;

import com.zimmem.math.Functions;
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

    CnnConvolutionLayer(int kernelRow, int kernelColumn, int pad, int step, int count, Function<Double, Double> activationFunction) {
        this.kernelRow = kernelRow;
        this.kernelColumn = kernelColumn;
        this.pad = pad;
        this.step = step;
        this.activationFunction = activationFunction;
        this.outputCount = count;
    }

    private final int kernelRow;
    private final int kernelColumn;
    private List<ConvFilter> filters;
    private int pad;
    private int step;

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
        List<Matrix> weightedInputs = new ArrayList<>(outputCount);
        for (int i = 0; i < outputCount; i++) {
            Matrix weighted = filters.get(i).filter(context.features.get(preLayer));
            weightedInputs.add(weighted);
            features.add(weighted.processUnits(d -> activationFunction.apply(d)));
        }
        context.weightedInputs.put(this, weightedInputs);
        context.features.put(this, features);

    }


    @Override
    protected List<Matrix> calculatePreDelta(CnnContext context) {


        List<Matrix> deltas = context.deltas.get(this);
        List<Matrix> preDeltas = new ArrayList<>(preLayer.outputCount);
        for (int i = 0; i < preLayer.outputCount; i++) {
            preDeltas.add(Matrix.zeros(preLayer.outputRow, preLayer.outputColumn));
        }
        for (int i = 0; i < outputCount; i++) {
            List<Matrix> featureDelta = filters.get(i).calculateDeltas(deltas.get(i));
            for (int j = 0; j < preLayer.outputCount; j++) {
                preDeltas.set(j, preDeltas.get(j).plus(featureDelta.get(j)));
            }
        }

        return preDeltas;

    }

    @Override
    protected void updateWeightsAndBias(List<CnnContext> contexts, double eta) {
        IntStream.range(0, filters.size()).forEach(fi -> {
            ConvFilter filter = filters.get(fi);

            filter.update(contexts, fi, eta);
        });
    }


    private class ConvFilter {
        double bias;

        List<Matrix> kernels;


        ConvFilter(int kernelRow, int kernelColumn, int kernelCount) {
            kernels = new ArrayList<>(kernelCount);
            for (int i = 0; i < kernelCount; i++) {
                kernels.add(Matrix.random(kernelRow, kernelColumn, 0, 0.05d));
            }
        }

        Matrix filter(List<Matrix> sources) {

            Matrix result = sources.get(0).conv(kernels.get(0), 0, 1);

            for (int i = 1; i < sources.size(); i++) {
                result = result.plus(sources.get(i).conv(kernels.get(i), 0, 1));
            }

            return result;
        }


        List<Matrix> calculateDeltas(Matrix delta) {

            List<Matrix> preDeltas = new ArrayList<>(preLayer.outputCount);
            for (int i = 0; i < preLayer.outputCount; i++) {
                // 暂不考虑 正向 conv 时 step > 1 的情况
                // 为什么转180度？
                preDeltas.add(delta.conv(kernels.get(i).rotate180(), (preLayer.outputRow + kernels.get(i).getRow() - 1 - delta.getRow()) / 2, 1));
            }
            return preDeltas;

        }

        void update(List<CnnContext> contexts, int index, double eta) {
            List<Matrix> deltas = contexts.stream().map(c -> c.deltas.get(CnnConvolutionLayer.this).get(index)).collect(Collectors.toList());
            this.bias += deltas.stream().mapToDouble(dm -> {
                double sum = 0;
                for (int r = 0; r < dm.getRow(); r++) {
                    for (int c = 0; c < dm.getColumn(); c++) {
                        sum += dm.getValue(r, c);
                    }
                }
                return sum;
            }).sum() * eta / contexts.size();
            for (int j = 0; j < kernels.size(); j++) {
                Matrix kernelWeight = Matrix.zeros(kernelRow, kernelColumn);
                for (CnnContext context : contexts) {
                    List<Matrix> preFeatures = context.features.get(preLayer);
                    kernelWeight = kernelWeight.plus(preFeatures.get(j).conv(context.deltas.get(CnnConvolutionLayer.this).get(index), 0, 1));
                }
                //kernelWeight.processUnits(d -> Functions.SigmoidDerivative.apply(d)/ contexts.size() * eta);
                kernelWeight.processUnits(d -> Functions.SigmoidDerivative.apply(d) / contexts.size() * eta);
                //kernelWeight.processUnits(d -> d / contexts.size() * eta);
                kernels.set(j, kernels.get(j).plus(kernelWeight));

            }


        }


    }
}
