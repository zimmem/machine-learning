package com.zimmem.math;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * Created by zimmem on 2016/7/26.
 */
public class Matrix {

    public double[][] values;

    private static Random random = new Random();
    private int row;
    private int column;

    public Matrix(int row, int column) {
        this.row = row;
        this.column = column;
        values = new double[row][column];
    }

    public Matrix(double[][] values) {

        if (values.length == 0 || values[0].length == 0) {
            throw new IllegalArgumentException("values can not be empty");
        }

        this.row = values.length;
        this.column = values[0].length;
        this.values = values;
    }


    public static Matrix random(int row, int column, double min, double max) {
        Matrix matrix = new Matrix(row, column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                matrix.values[r][c] = random.nextDouble() * (max - min) + min;
            }
        }
        return matrix;
    }

    public static Matrix single(double value){
        Matrix matrix = new Matrix(1, 1);
        matrix.values[0][0] = value;
        return matrix;
    }


    public double getValue(int row, int column) {
        return values[row][column];
    }

    /**
     * 旋转 180 度
     *
     * @return
     */
    public Matrix rotate180() {
        Matrix target = new Matrix(this.row, this.column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                target.values[row - 1 - r][column - 1 - c] = this.values[r][c];
            }
        }
        return target;
    }

    public Matrix multiply(Matrix other) {
        //TODO
        throw new RuntimeException();
    }

    public Matrix plus(Matrix other) {
        if (this.row != other.row || this.column != other.column) {
            throw new IllegalArgumentException("row or column not match when Matrix plus.");
        }
        Matrix matrix = Matrix.zeros(row, column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                matrix.values[r][c] = this.values[r][c] + other.values[r][c];
            }
        }
        return matrix;
    }

    public Matrix minus(Matrix other) {
        if (this.row != other.row || this.column != other.column) {
            throw new IllegalArgumentException("row or column not match when Matrix minus.");
        }
        Matrix matrix = Matrix.zeros(row, column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                matrix.values[r][c] = this.values[r][c] - other.values[r][c];
            }
        }
        return matrix;
    }



    public Matrix T() {
        Matrix t = new Matrix(this.column, this.row);
        for (int r = 0; r < this.row; r++) {
            for (int c = 0; c < this.column; c++) {
                t.values[c][r] = this.values[r][c];
            }
        }
        return t;
    }

    public Matrix conv(Matrix kernel, int pad, int step) {

        int targetRows = (this.row + pad * 2 - kernel.row) / step + 1;
        int targetColumns = (this.column + pad * 2 - kernel.column) / step + 1;

        Matrix result = new Matrix(targetRows, targetColumns);

        for (int stepRow = 0 - pad; stepRow <= this.getRow() + pad - kernel.getRow(); stepRow += step) {

            for (int stepColumn = 0 - pad; stepColumn <= this.getColumn() + pad - kernel.getColumn(); stepColumn += step) {

                int resultRow = (stepRow + pad) / step;
                int resultColumn = (stepColumn + pad) / step;
                double tempResult = 0;
                for (int kernelRow = 0; kernelRow < kernel.getRow(); kernelRow++) {
                    for (int kernelColumn = 0; kernelColumn < kernel.getColumn(); kernelColumn++) {
                        int sourceRow = stepRow + kernelRow;
                        int sourceColumn = stepColumn + kernelColumn;
                        double sourceValue = sourceRow < 0 || sourceColumn < 0 || sourceRow >= this.row || sourceColumn >= this.column ? 0 : this.values[sourceRow][sourceColumn];
                        tempResult += sourceValue * kernel.getValue(kernelRow, kernelColumn);
                    }
                }
                result.setValue(resultRow, resultColumn, result.getValue(resultRow, resultColumn) + tempResult);

            }

        }
        return result;
    }


    public Matrix processUnits(DoubleFunction<Double> function) {
        Matrix m = Matrix.zeros(row, column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                m.values[r][c] = function.apply(this.values[r][c]);
            }
        }
        return m;
    }

    public static Matrix zeros(int targetRow, int targetColumn) {
        return new Matrix(targetRow, targetColumn);
    }

    public static Matrix ones(int targetRow, int targetColumn) {
        Matrix matrix = new Matrix(targetRow, targetColumn);
        matrix = matrix.processUnits(d -> 1d);
        return matrix;
    }


    public int getRow() {
        return row;
    }

    public int getColumn() {
        return column;
    }


    public Matrix setValue(int row, int column, double value) {
        this.values[row][column] = value;
        return this;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Arrays.stream(values).forEach(a -> sb.append(Arrays.toString(a)).append("\n"));
        return sb.toString();
    }

    public static void main(String[] args) {
        Matrix random = Matrix.random(3, 3, -1, 1);
        System.out.println(random);
        //System.out.println(random.conv(Matrix.ones(2, 2), 5, 1));
    }
}
