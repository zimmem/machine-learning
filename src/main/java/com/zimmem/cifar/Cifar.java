package com.zimmem.cifar;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zimmem on 2016/8/8.
 */
public class Cifar {
    private static Logger log = LoggerFactory.getLogger(Cifar.class);

    public  static List<CifarImage> loadTrandImages() {
        File dir = new File("cifar-10-batches-bin");
        if (!dir.exists()) {
            log.error("cifar-10 dataset not exist! please download at http://www.cs.toronto.edu/~kriz/cifar.html");
        }

        File[] dataBatches = dir.listFiles((dir1, name) -> name.startsWith("data_batch_"));
        return loadImagesFormFile(dataBatches);
    }

    public static  List<CifarImage> loadTestImages() {
        File file = new File("cifar-10-batches-bin/test_batch.bin");
        if (!file.exists()) {
            log.error("cifar-10 dataset not exist! please download at http://www.cs.toronto.edu/~kriz/cifar.html");
        }

        List<CifarImage> images = loadImagesFormFile(new File[]{file});
        return images;
    }


    private static List<CifarImage> loadImagesFormFile(File[] dataBatches) {
        List<CifarImage> images = new ArrayList<>();
        Arrays.stream(dataBatches).forEach(f -> {
            FileInputStream inputStream = null;

            try {
                inputStream = new FileInputStream(f);
            } catch (FileNotFoundException e) {
                log.error("load cifar data failure.", e);
                throw new RuntimeException(e);
            }

            try {
                int length = inputStream.available();
                while (length > 0) {
                    CifarImage image = new CifarImage();
                    image.setLabel(inputStream.read());
                    byte[] buffer = new byte[1024];
                    inputStream.read(buffer);
                    image.setRedBytes(buffer);
                    buffer = new byte[1024];
                    inputStream.read(buffer);
                    image.setGreenBytes(buffer);
                    buffer = new byte[1024];
                    inputStream.read(buffer);
                    image.setBlueBytes(buffer);
                    images.add(image);
                    length -= 3073;
                }


            } catch (IOException e) {
                log.error("load cifar data failure.", e);
                throw new RuntimeException(e);
            }
        });
        return images;
    }

    public static void main(String[] args){
        System.out.println(loadTrandImages().size());
        System.out.println(loadTestImages().size());

    }

}
