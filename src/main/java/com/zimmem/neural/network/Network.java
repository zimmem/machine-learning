package com.zimmem.neural.network;

import com.zimmem.mnist.MnistImage;

import java.util.List;

/**
 * Created by zimmem on 2016/7/26.
 */
public interface Network {

    /**
     * 训练
     *
     * @param mnistDataSet 训练数据集
     * @param batchSize  每批几个， （整个训练集会切成一批一批）
     * @param repeat 重复几次
     */
    void train(List<MnistImage> images, int batchSize , int repeat);
}
