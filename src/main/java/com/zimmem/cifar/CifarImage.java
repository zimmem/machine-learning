package com.zimmem.cifar;

import com.zimmem.math.Matrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zimmem on 2016/8/8.
 */
public class CifarImage {

    private int label;

    private byte[] redBytes;

    private byte[] greenBytes;

    private byte[] blueBytes;

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public byte[] getRedBytes() {
        return redBytes;
    }

    public void setRedBytes(byte[] redBytes) {
        this.redBytes = redBytes;
    }

    public byte[] getGreenBytes() {
        return greenBytes;
    }

    public void setGreenBytes(byte[] greenBytes) {
        this.greenBytes = greenBytes;
    }

    public byte[] getBlueBytes() {
        return blueBytes;
    }

    public void setBlueBytes(byte[] blueBytes) {
        this.blueBytes = blueBytes;
    }

    public List<Matrix> asMatrices(){
        return Arrays.asList(array2matrix(redBytes), array2matrix(greenBytes), array2matrix(blueBytes));
    }

    private Matrix array2matrix(byte[] values){
        Matrix m = new Matrix(32, 32);
        for (int c = 0; c < 32; c++) {
            for (int r = 0; r < 32; r++) {
                m.setValue(r, c, (double)(values[32 * c + r] & 0xff) / 255  );
            }
        }
        return m;
    }

    public static void main(String[] args) throws IOException {
        CifarImage ci = Cifar.loadTrandImages().get(2);
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_3BYTE_BGR);
        System.out.println(ci.getLabel());
        for(int c = 0 ; c < 32 ; c++){
            for (int r = 0 ; r < 32 ; r++){
                image.setRGB(r, c , (ci.redBytes[c * 32 + r ] << 16)  + (ci.greenBytes[c * 32 + r ] << 8) + (ci.blueBytes[c * 32 + r ] & 0xff)  );
            }
        }
        ImageIO.write(image, "jpg", new File("D:\\test.jpg"));
    }
}
