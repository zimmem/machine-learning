package com.zimmem;

import com.zimmem.math.Matrix;
import com.zimmem.neural.network.cnn.CnnTrainContext;
import com.zimmem.neural.network.cnn.CnnTrainListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class Stat2LogListener implements CnnTrainListener {

    protected static Logger log = LoggerFactory.getLogger(Stat2LogListener.class);

    protected AtomicInteger batchCollect = new AtomicInteger(0);
    protected AtomicInteger totalCollect = new AtomicInteger(0);
    protected AtomicInteger batchTrained = new AtomicInteger(0);
    protected AtomicInteger totalTrained = new AtomicInteger(0);
    protected AtomicInteger batch = new AtomicInteger(0);

    @Override
    public void onForwardFinish(CnnTrainContext context, List<Matrix> output) {
        if (maxLabel(context.getExcepted()) == maxLabel(output)) {
            batchCollect.incrementAndGet();
            totalCollect.incrementAndGet();
            //System.out.println(output);
        }
        totalTrained.incrementAndGet();
        batchTrained.incrementAndGet();
    }

    @Override
    public void onBatchFinish(List<CnnTrainContext> contexts) {
        batch.incrementAndGet();
        log.debug("batch {} : {}/{} - total {}/{} = {}",batch, batchCollect, batchTrained, totalCollect, totalTrained, totalCollect.doubleValue() / totalTrained.doubleValue());
        batchCollect.set(0);
        batchTrained.set(0);
    }

    @Override
    public void onEpochFinish(int epoch) {
        log.info("epoch {} :  total {}/{} = {}", epoch, totalCollect, totalTrained, totalCollect.doubleValue() / totalTrained.doubleValue());
        totalCollect.set(0);
        totalTrained.set(0);
        batch.set(0);
    }

    protected int maxLabel(List<Matrix> matrices) {
        int label = 0;
        double max = matrices.get(0).getValue(0, 0);
        for (int i = 1; i < matrices.size(); i++) {
            if (max < matrices.get(i).getValue(0, 0)) {
                max = matrices.get(i).getValue(0, 0);
                label = i;
            }
        }
        return label;
    }
}