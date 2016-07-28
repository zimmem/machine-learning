package com.zimmem.mnist;

/**
 * Created by zimmem on 2016/7/26.
 */
public class MnistImage {

    private byte[] values;

    private int label;

    public MnistImage(byte[] values) {
        this.values = values;
    }

    public byte[] getValues() {
        return values;
    }

    public void setValues(byte[] values) {
        this.values = values;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }
}
