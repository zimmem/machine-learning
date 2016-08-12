package com.zimmem.neural.network.cnn;

import com.alibaba.fastjson.JSON;
import com.zimmem.Stat2LogListener;
import com.zimmem.math.Matrix;
import com.zimmem.neural.network.NetworkBuilder;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by zimmem on 2016/8/12.
 */
public class SoftmaxTest {
    static ConvolutionNeuralNetwork network = null;
    static Logger log = LoggerFactory.getLogger("b");

    public static void main(String[] args) throws InterruptedException {
        network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(6, 6, 1))
                .addLayer(new CnnPoolingLayer(3, 3, CnnPoolingLayer.Strategy.Max))
                .addLayer(new CnnConvolutionLayer(2, 2, 3))
                .addLayer(new CnnSoftmaxLayer())
                .addListener(new Stat2LogListener())
                .addListener(new CnnTrainListener() {
                    @Override
                    public void onForwardFinish(CnnTrainContext context, List<Matrix> output) {
                        CnnLayer current = SoftmaxTest.network.inputLayer;
                        while (current!=null){
                            log.info("{}_output = {}", current.getClass().getSimpleName(), JSON.toJSONString( context.features.get(current)));
                            current = current.nextLayer;
                        }
                    }

                    @Override
                    public void onBatchFinish(List<CnnTrainContext> contexts) {
                        for (CnnTrainContext context : contexts) {
                            CnnLayer current = SoftmaxTest.network.outputLayer;
                            while (current!=null){
                                log.info("{}_delta = {}", current.getClass().getSimpleName(), JSON.toJSONString( context.deltas.get(current)));
                                current = current.preLayer;
                            }
                        }
                    }

                    @Override
                    public void onEpochFinish(int epoch) {
                        CnnLayer current = SoftmaxTest.network.inputLayer;
                        while (current != null) {
                            if(current instanceof  CnnConvolutionLayer){
                                printKernel((CnnConvolutionLayer) current, "after");
                            }
                            current = current.nextLayer;
                        }


                    }
                })
                .build();

        CnnLayer current = SoftmaxTest.network.inputLayer;
        while (current != null) {
            if(current instanceof  CnnConvolutionLayer){
                printKernel((CnnConvolutionLayer) current, "before");
            }
            current = current.nextLayer;
        }

        Matrix inputValue = Matrix.random(6, 6, -1, 1);
        log.info("var input = {}", JSON.toJSONString(matrix2list(Arrays.asList(inputValue))));
        List<Matrix> expected = Stream.of(0, 0, 1).map(Matrix::single).collect(Collectors.toList());
        CnnTrainInput input = new CnnTrainInput(Arrays.asList(inputValue), expected);
        network.train(Arrays.asList(input), 1, 1);

        network.shutdown();
        LogManager.shutdown();


    }

    private static List<Double> matrix2list(List<Matrix> matrices) {
        List<Double> w = new ArrayList<>();
        matrices.forEach(k -> {
            Arrays.stream(k.getValues()).forEach(a -> {
                Arrays.stream(a).forEach(w::add);
            });
        });
        return w;
    }

    private static void printKernel(CnnConvolutionLayer layer, String key) {
        List<Double> bias = layer.filters.stream().map(f -> f.bias).collect(Collectors.toList());
        List<List<Double>> f = new ArrayList<>();
        List<Map<String, Object>> filters = new ArrayList<>();
        layer.filters.forEach(fi -> {
            Map<String, Object> filter = new HashMap<>();
            filter.put("sx", layer.kernelRow);
            filter.put("sy", layer.kernelColumn);
            filter.put("depth", layer.preLayer.outputCount);
            filter.put("w",matrix2list(fi.kernels));
            filters.add(filter);
        });
        log.info("{}_bias = {}", key, JSON.toJSONString(bias));
        log.info("{}_filter = {}", key, JSON.toJSONString(filters));
        System.out.println(JSON.toJSONString(f));

    }
}


