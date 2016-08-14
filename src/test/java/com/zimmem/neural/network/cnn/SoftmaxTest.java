package com.zimmem.neural.network.cnn;

import com.alibaba.fastjson.JSON;
import com.zimmem.Stat2LogListener;
import com.zimmem.math.ActivationFunction;
import com.zimmem.math.Matrix;
import com.zimmem.mnist.Mnist;
import com.zimmem.mnist.MnistImage;
import com.zimmem.mnist.MnistLabel;
import com.zimmem.neural.network.NetworkBuilder;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/8/12.
 */
public class SoftmaxTest {
    static ConvolutionNeuralNetwork network = null;
    static Logger log = LoggerFactory.getLogger("b");

    public static void main(String[] args) throws InterruptedException, IOException {
        network = NetworkBuilder.cnn()
                .addLayer(new CnnInputLayer(28, 28, 1))
                .addLayer(new CnnConvolutionLayer(5, 5, 1))
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max))
                .addLayer(new CnnConvolutionLayer(5, 5, 1))
                .addLayer(new CnnActivationLayer(ActivationFunction.Relu))
                .addLayer(new CnnPoolingLayer(2, 2, CnnPoolingLayer.Strategy.Max))
                .addLayer(new CnnConvolutionLayer(4, 4, 10))
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
        int i = 0;
        log.info("var filters = []");
        while (current != null) {
            if(current instanceof  CnnConvolutionLayer){
                log.info("var before_filter = {}");
                printKernel((CnnConvolutionLayer) current, "before_filter.");
                log.info("filters.push(before_filter)");
            }
            current = current.nextLayer;
        }

        List<MnistImage> trainImages = Mnist.loadImages("/mnist/train-images.idx3-ubyte");
        List<MnistLabel> trainLabels = Mnist.loadLabels("/mnist/train-labels.idx1-ubyte");
        Matrix inputValue = trainImages.get(0).asMatrix();
        log.info("var input = {}", JSON.toJSONString(matrix2list(Collections.singletonList(inputValue))));
        log.info("var expected  = {}",trainLabels.get(0).getValue()  );
        List<Matrix> expected = IntStream.range(0,10).mapToObj(a -> a == trainLabels.get(0).getValue() ? Matrix.single(1) : Matrix.single(0)).collect(Collectors.toList());
        CnnTrainInput input = new CnnTrainInput(Collections.singletonList(inputValue), expected);
        network.train(Collections.singletonList(input), 1, 1, 0.85);

        CnnContext context = new CnnContext();
        context.setInputs(Collections.singletonList(inputValue));
        System.out.println(network.forward(context));

        current = network.inputLayer;
        while(current !=null ){
            System.out.println("========"+current+"========");
            List<Matrix> matrices = context.features.get(current);
            matrices.stream().forEach(m ->{
                System.out.println(m);
                System.out.println("----------------");
            });
            current = current.nextLayer;
        }

        network.shutdown();
        LogManager.shutdown();


    }

    private static List<Double> matrix2list(List<Matrix> matrices) {
        List<Double> w = new ArrayList<>();
        Matrix first = matrices.get(0);
        IntStream.range(0,first.getRow()).forEach(r -> {
            IntStream.range(0,first.getColumn()).forEach(c -> {
                matrices.forEach(k -> {
                    w.add(k.getValue(r,c));
                });
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


