package com.zimmem.mnist;

import com.zimmem.math.Matrix;

import java.awt.image.BufferedImage;

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

    public BufferedImage asImage() {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int c = 0; c < 28; c++) {
            for (int r = 0; r < 28; r++) {
                image.setRGB(r, c, values[28 * c + r]);
            }
        }
        return image;
    }

    public Matrix asMatrix() {
        Matrix m = new Matrix(28, 28);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                m.setValue(r, c, (values[28 * r + c] & 0xff) > 0 ? 1 : 0d);
                //m.setValue(r, c, (values[28 * c + r] & 0xff));
            }
        }
        return m;
    }
}
