package com.zimmem.neural.network.bp;

import com.zimmem.mnist.*;
import com.zimmem.neural.network.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.jar.Pack200;
import java.util.stream.Collectors;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BPNetwork implements Network {

    private Logger log = LoggerFactory.getLogger(BPNetwork.class);

    List<Layer> layers;

    @Override
    public void train(List<MnistImage> images, List<MnistLabel> labels , int batchSize , int repeat) {
        long start = System.currentTimeMillis();
        log.info("begin to train at {}" , start);

        Random random = new Random();
        while (repeat -- > 0 ){

            Set<Integer> set = new HashSet<>();
            while (set.size() < batchSize){
                set.add(random.nextInt(images.size()));
            }

            List<MnistImage> imageBatch = set.stream().map(images::get).collect(Collectors.toList());
            imageBatch.forEach((image) -> {
                double[] output = classify(image);
            });
        }

        log.info("train finish, cast {} ms", System.currentTimeMillis() - start);
    }

    /**
     * 分类
     * @param image
     * @return
     */
    public double[] classify(MnistImage image){
        Layer inputLayer = layers.get(0);

        for (int i = 0; i < image.getValues().length; i++) {
            inputLayer.values[i] = (double) image.getValues()[i];
        }

        Layer preLayer = inputLayer;
        for(int i = 1; i < layers.size(); i++){
            layers.get(i).spread(preLayer);
            preLayer = layers.get(i);
        }

        double[] output = layers.get(layers.size() - 1).values;
        log.debug(Arrays.toString(output));
        return output;
    }

}
