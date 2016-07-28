package com.zimmem.mnist;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zimmem on 2016/7/26.
 */
public class Mnist {

    public static MnistDataSet trainData() {
        return null;
    }

    public static MnistDataSet validData() {
        return null;
    }

    public static MnistDataSet testData() {
        return null;
    }


    public static List<MnistImage> loadImages(String file) throws IOException {

        InputStream is = Mnist.class.getResourceAsStream(file);
        DataInputStream dis = new DataInputStream(is);
        int magic = dis.readInt();
        assert magic == 0x00000803;
        int count = dis.readInt();
        List<MnistImage> images = new ArrayList<>(count);
        int row = dis.readInt();
        int column = dis.readInt();
        while (count-- > 0) {
            byte[] value = new byte[row * column];
            dis.read(value);

            images.add(new MnistImage(value));

        }
        return images;
    }

    public static List<MnistLabel> loadLabels(String file) throws IOException {
        InputStream is = Mnist.class.getResourceAsStream(file);
        DataInputStream dis = new DataInputStream(is);
        int magic = dis.readInt();
        assert magic == 0x00000801;
        int count = dis.readInt();
        List<MnistLabel> labels = new ArrayList<>(count);
        while (count-- > 0) {
            labels.add(new MnistLabel(dis.readByte()));
        }
        return labels;
    }

    public static void main(String[] args) throws URISyntaxException, IOException {
        List<MnistLabel> labels = loadLabels("/mnist/t10k-labels.idx1-ubyte");
        //System.out.println(labels.size());

        List<MnistImage> images = loadImages("/mnist/t10k-images.idx3-ubyte");
//        System.out.println(labels.size());
//        images.forEach(i -> System.out.println(Arrays.toString(i.getValues())));

        for(int i = 0 ; i < images.size() ;i ++ ){
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for(int c = 0 ; c < 28 ; c++){
                for (int r = 0 ; r < 28 ; r++){
                    image.setRGB(r, c , images.get(i).getValues()[28 * c + r ]);
                }
            }
            new File("D:\\code\\mnist-picture\\test\\").mkdirs();
            ImageIO.write(image, "jpg", new File("D:\\code\\mnist-picture\\test\\" +  i + "_"+ labels.get(i).getValue()+".jpg"));
        }

    }
}
