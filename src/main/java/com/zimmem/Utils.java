package com.zimmem;

import com.zimmem.math.Matrix;

import java.util.List;

/**
 * Created by Zimmem on 2016/9/4.
 */
public class Utils {

    public static  int maxLabel(List<Matrix> matrices) {
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
