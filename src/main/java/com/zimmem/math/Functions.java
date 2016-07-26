package com.zimmem.math;

import java.util.function.Function;

/**
 * Created by zimmem on 2016/7/26.
 */
public class Functions {

    public static Function<Double, Double> sigmoid = d -> 1 / (1 + Math.pow(Math.E, -d));
}
