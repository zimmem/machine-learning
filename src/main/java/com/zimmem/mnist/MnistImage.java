package com.zimmem.mnist;

/**
 * Created by zimmem on 2016/7/26.
 */
public class MnistImage {

    private byte[] values;

    private int index;

    public MnistImage(byte[] values) {
        this.values = values;
    }

    public byte[] getValues() {
        return values;
    }

    public void setValues(byte[] values) {
        this.values = values;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
