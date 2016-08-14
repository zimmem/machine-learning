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


    private int filterRow;

    private int filterColumn;

    private Strategy strategy;

    public CnnPoolingLayer(int filterRow, int filterColumn, Strategy strategy) {
        this.filterRow = filterRow;
        this.filterColumn = filterColumn;
        this.strategy = strategy;
    }

    void init() {
        this.outputColumn = preLayer.outputColumn / filterColumn;
        this.outputRow = preLayer.outputRow / filterRow;
        this.outputCount = preLayer.outputCount;

    }

    @Override
    protected void propagation(CnnContext context) {
        List<Matrix> output = new ArrayList<>(outputCount);
        List<Matrix> preOutput = context.features.get(preLayer);
        IntStream.range(0, outputCount).forEach(i -> {
            Matrix pooled = strategy.pooling(preOutput.get(i), filterRow, filterColumn);
            output.add(pooled);
        });
        context.features.put(this, output);
    }

    @Override
    protected List<Matrix> calculatePreDelta(CnnTrainContext context) {

        List<Matrix> deltas = context.deltas.get(this);
        List<Matrix> preDeltas = new ArrayList<>(preLayer.outputCount);
        // pool 层输出与输入数是一致的， 传播没有交叉， 所以直接把当前层的残卷扩展到上一次特征图大小
        for (int i = 0; i < preLayer.outputCount; i++) {
//            Matrix delta = Matrix.zeros(preLayer.outputRow, preLayer.outputColumn);
//            for (int r = 0; r < preLayer.outputRow; r++) {
//                for (int c = 0; c < preLayer.outputColumn; c++) {
//                    //delta.setValue(r, c, deltas.get(i).getValue(r / this.filterRow, c / this.filterColumn) / (this.filterRow * this.filterColumn));
//                    delta.setValue(r, c, deltas.get(i).getValue(r / this.filterRow, c / this.filterColumn));
//                }
//            }
            Matrix preDelta = strategy.unPooling(deltas.get(i), context.features.get(preLayer).get(i));
            preDeltas.add(preDelta);
        }
        return preDeltas;

    }

    @Override
    protected void updateWeightsAndBias(List<CnnTrainContext> contexts, double eta) {
        //do nothing
    }


    public static abstract class Strategy {

        public abstract Matrix pooling(Matrix source, int poolRow, int poolColumn);

        public abstract Matrix unPooling(Matrix source, Matrix template);

        public static Strategy Max = new Strategy() {
            @Override
            public Matrix pooling(Matrix source, int poolRow, int poolColumn) {
                Matrix target = Matrix.zeros(source.getRow() / poolColumn, source.getRow() / poolColumn);

                for (int r = 0; r < target.getRow(); r++) {
                    for (int c = 0; c < target.getColumn(); c++) {

                        double max = source.getValue(r * poolRow, c * poolColumn);
                        int srbegin = r * poolRow;
                        int srend = srbegin + poolRow;
                        int scbegin = c * poolColumn;
                        int scend = scbegin + poolColumn;
                        for (int sr = srbegin; sr < srend; sr++) {
                            for (int sc = scbegin; sc < scend; sc++) {
                                max = Math.max(max, source.getValue(sr, sc));
                            }
                        }
                        target.setValue(r, c, max);

                    }
                }
                return target;
            }

            @Override
            public Matrix unPooling(Matrix source, Matrix template) {

                Matrix result = Matrix.zeros(template.getRow(), template.getRow());

                int poolRow = template.getRow() / source.getRow();
                int poolColumn = template.getColumn() / source.getColumn();

                for (int r = 0; r < source.getRow(); r++) {
                    for (int c = 0; c < source.getColumn(); c++) {



                        int trBegin = r * poolRow;
                        int trEnd = trBegin + poolRow;
                        int tcBegin = c * poolColumn;
                        int tcEnd = tcBegin + poolColumn;
                        double max = template.getValue(trBegin, tcBegin);
                        int maxRow = trBegin;
                        int maxColumn = tcBegin;
                        for (int tr = trBegin; tr < trEnd; tr++) {
                            for (int tc = tcBegin; tc < tcEnd; tc++) {
                                if (max < template.getValue(tr, tc)) {
                                    max = template.getValue(tr, tc);
                                    maxRow = tr;
                                    maxColumn = tc;
                                }
                                //TODO 批量训练时直接 Kronecker 到上层
                                //result.setValue(tr, tc, source.getValue(r, c ));
                            }
                        }
                        result.setValue(maxRow, maxColumn, source.getValue(r, c ));

                    }
                }
                return result;

            }
        };

        public static Strategy Means = new Strategy() {
            @Override
            public Matrix pooling(Matrix source, int poolRow, int poolColumn) {
                Matrix target = Matrix.zeros(source.getRow() / poolColumn, source.getRow() / poolColumn);

                for (int r = 0; r < target.getRow(); r++) {
                    for (int c = 0; c < target.getColumn(); c++) {

                        double sum = 0;
                        int srbegin = r * poolRow;
                        int srend = srbegin + poolRow;
                        int scbegin = c * poolColumn;
                        int scend = scbegin + poolColumn;
                        for (int sr = srbegin; sr < srend; sr++) {
                            for (int sc = scbegin; sc < scend; sc++) {
                                sum += source.getValue(sr, sc);
                            }
                        }
                        target.setValue(r, c, sum / (poolRow * poolColumn));

                    }
                }
                return target;
            }

            @Override
            public Matrix unPooling(Matrix source, Matrix template) {

                Matrix result = Matrix.zeros(template.getRow(), template.getRow());

                int poolRow = template.getRow() / source.getRow();
                int poolColumn = template.getColumn() / source.getColumn();

                for (int r = 0; r < source.getRow(); r++) {
                    for (int c = 0; c < source.getColumn(); c++) {

                        int trBegin = r * poolRow;
                        int trEnd = trBegin + poolRow;
                        int tcBegin = c * poolColumn;
                        int tcEnd = tcBegin + poolColumn;
                        for (int tr = trBegin; tr < trEnd; tr++) {
                            for (int tc = tcBegin; tc < tcEnd; tc++) {
                                //TODO 批量训练时直接 Kronecker 到上层， 单个时均分偏差
                                //result.setValue(tr, tc, source.getValue(r, c));
                                result.setValue(tr, tc, source.getValue(r, c) / (poolRow * poolColumn));
                            }
                        }

                    }
                }
                return result;

            }
        };


    }


}