package com.zimmem.math;

import java.util.logging.XMLFormatter;

/**
 * Created by zimmem on 2016/8/4.
 */
public interface ActivationFunction {

    double run(double value);

    double runDerivative(double value);

    ActivationFunction Sigmoid = new ActivationFunction() {
        @Override
        public double run(double x) {
            return 1 / (1 + Math.pow(Math.E, -x));
        }

        @Override
        public double runDerivative(double x) {
            return run(x) * (1 - run(x));
        }
    };

    ActivationFunction Relu = new ActivationFunction() {
        @Override
        public double run(double x) {
            return x > 0 ? x : 0;
        }

        @Override
        public double runDerivative(double x) {
            return x <= 0 ? 0 : 1;
        }
    };

}
