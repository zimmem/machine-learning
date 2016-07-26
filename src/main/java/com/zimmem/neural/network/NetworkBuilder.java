package com.zimmem.neural.network;

import com.zimmem.neural.network.bp.BPNetWorkBuilder;

/**
 * Created by zimmem on 2016/7/26.
 */
public abstract class NetworkBuilder<T extends  Network> {

    public static BPNetWorkBuilder bp(){
        return new BPNetWorkBuilder();
    }

    public abstract  T build();
}
