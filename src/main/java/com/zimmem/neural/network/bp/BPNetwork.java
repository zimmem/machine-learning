package com.zimmem.neural.network.bp;

import com.zimmem.math.Functions;
import com.zimmem.mnist.*;
import com.zimmem.neural.network.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by zimmem on 2016/7/26.
 */
public class BPNetwork implements Network, Serializable {

    private Logger log = LoggerFactory.getLogger(BPNetwork.class);

    private static ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    Layer inputLayer;

    Layer outputLayer;

    @Override
    public void train(List<MnistImage> images, List<MnistLabel> labels, int batchSize, int repeat) {
        long start = System.currentTimeMillis();
        log.info("begin to train at {}", start);

        for (int i = 0; i < images.size(); i++) {
            images.get(i).setLabel(labels.get(i).getValue());
        }

        for (int epoch = 1; epoch <= repeat; epoch++) {
            //重复 repeat 次

            Collections.shuffle(images);

            int correct = 0;
            AtomicInteger batchCorrect = new AtomicInteger(0);
            double verifyRate = 0d;
            for (int batch = 0; batch < images.size(); batch += batchSize) {

                if (batch % 1000 == 0) {
                    // 每训练1000个数据， 拿最后1000个数据做下验证
                    verifyRate = verify(images.subList(images.size() - 1000, images.size()));
                }

                batchCorrect.set(0);
                List<TrainContext> contexts = new ArrayList<>(batchSize);
                CountDownLatch latch = new CountDownLatch(batchSize);
                for (int index = batch; index < batch + batchSize; index++) {
                    MnistImage image = images.get(index);
                    executor.execute(() -> {
                        TrainContext context = new TrainContext();
                        synchronized (contexts) {
                            contexts.add(context);
                        }
                        double[] output = forward(context, image);

                        if (!Double.isNaN(output[image.getLabel()]) && Objects.equals(Arrays.stream(output).max().getAsDouble(), output[image.getLabel()])) {
                            //System.out.println(Arrays.toString(output));
                            batchCorrect.getAndIncrement();
                        }

                        // 输出与期望的偏差
                        double[] expect = new double[output.length];
                        expect[image.getLabel()] = 1;
                        double[] error = new double[output.length];
                        for (int i = 0; i < output.length; i++) {
                            error[i] = output[i] - expect[i];
                        }

                        // 计算 output 层偏差
                        double[] deltas = new double[outputLayer.size];
                        IntStream.range(0, outputLayer.size).forEach(i -> deltas[i] = error[i] * Functions.SigmoidDerivative.apply(context.weightedInputs.get(outputLayer)[i]));
                        outputLayer.backPropagationDelta(context, deltas);
                        latch.countDown();
                    });
                }
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                correct += batchCorrect.get();
                log.debug("batch {} : {}/{} - total {}/{} = {} ", batch / batchSize + 1, batchCorrect, batchSize, correct, batch + batchSize, (double) correct / (batch + batchSize));

                outputLayer.backPropagationUpdate(contexts, 3);
//                Random r = new Random();
//                outputLayer.backPropagationUpdate(contexts, (1 - verifyRate) * r.nextDouble());
                //resetTrainData();


            }
            present(epoch);
            log.info("epoch {}:  {} / {} ", epoch, correct, images.size());

        }

        log.info("train finish, cast {} ms", System.currentTimeMillis() - start);
    }

    private double verify(List<MnistImage> mnistImages) {
        AtomicInteger collect = new AtomicInteger(0);

        CountDownLatch latch = new CountDownLatch(mnistImages.size());
        IntStream.range(0, mnistImages.size()).forEach(i -> {
            executor.execute(() -> {
                double[] output = forward(new TrainContext(), mnistImages.get(i));
                double max = Arrays.stream(output).max().getAsDouble();
                if (Objects.equals(max, output[mnistImages.get(i).getLabel()]) && !Double.isNaN(max)) {
                    collect.incrementAndGet();
                }
                latch.countDown();
            });
        });
        try {
            latch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        double rate = collect.doubleValue() / mnistImages.size();
        log.info("verified {}/{} = {}", collect, mnistImages.size(), rate);
        return rate;
    }


    /**
     * 分类
     *
     * @param context
     * @param image
     * @return
     */
    public double[] forward(TrainContext context, MnistImage image) {

        double[] input = new double[image.getValues().length];
        for (int i = 0; i < image.getValues().length; i++) {
            input[i] = (double) (0xff & image.getValues()[i]) > 100 ? 1d : 0d;
        }
        context.activations.put(inputLayer, input);
        return inputLayer.forward(context);

    }

    void present(int epoch) {

        double[][] imgs = new double[outputLayer.size][];
        IntStream.range(0, outputLayer.size).forEach(oi -> {
            Layer current = outputLayer.preLayer;
            Layer next = outputLayer;
            double[] next_ws = new double[outputLayer.size];
            next_ws[oi] = 1d;
            while (current != null) {
                double[] ws = new double[current.size];
                for (int ci = 0; ci < current.size; ci++) {
                    for (int ni = 0; ni < next.size; ni++) {
                        ws[ci] += next_ws[ni] * next.weights[ni][ci];
                    }
                }
                next = current;
                current = current.preLayer;
                next_ws = ws;

            }
            imgs[oi] = next_ws;
        });

        long timestamp = System.currentTimeMillis();
        for (int i = 0; i < imgs.length; i++) {
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_3BYTE_BGR);
            double max = Arrays.stream(imgs[i]).map(Math::abs).max().getAsDouble();
            for (int c = 0; c < 28; c++) {
                for (int r = 0; r < 28; r++) {
                    double weight = imgs[i][28 * c + r];
                    int bgr = weight > 0 ? (int) (255 * weight / max) << 16 : (int) (-255 * weight / max);
                    image.setRGB(r, c, bgr);
                }
            }
            try {
                ImageIO.write(image, "jpg", new File("present\\" + timestamp + "_" + epoch + "_" + i + ".jpg"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }


}
