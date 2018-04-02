package com.zimmem.math;

import java.io.Serializable;
import java.util.logging.XMLFormatter;

/**
 * Created by zimmem on 2016/8/4.
 */
public interface ActivationFunction extends Serializable {

    double run(double value);

    double runDerivative(double value);

    ActivationFunction Sigmoid = new ActivationFunction() {
        @Override
        public double run(double x) {
            return 1 / (1 + Math.exp(-x));
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

    public static void main(String[] args){
        System.out.println(Sigmoid.runDerivative(-8.148001762954165E-5));

        System.out.println(Functions.SigmoidDerivative.apply(-8.148001762954165E-5));
    }


}
