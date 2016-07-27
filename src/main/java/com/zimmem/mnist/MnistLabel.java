package com.zimmem.mnist;

/**
 * Created by zimmem on 2016/7/26.
 */
public class MnistLabel {

    private int value;

    public MnistLabel(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return String.valueOf(value);
    }
}
