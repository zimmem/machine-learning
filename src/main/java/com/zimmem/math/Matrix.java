package com.zimmem.math;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * Created by zimmem on 2016/7/26.
 */
public class Matrix {

    private double[][] values;

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


    public static Matrix random(int row, int column, double v) {
        Matrix matrix = new Matrix(row, column);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                matrix.values[r][c] = random.nextDouble() * v;
            }
        }
        return matrix;
    }


    public double getValue(int row, int column) {
        return values[row][column];
    }

    public Matrix multiply(Matrix other) {
        //TODO
        throw new RuntimeException();
    }

    public Matrix plus(Matrix other) {
        //TODO
        throw new RuntimeException();
    }

    public Matrix T(){
        Matrix t = new Matrix(this.column, this.row);
        for(int r = 0 ; r < this.row ; r ++) {
            for (int c = 0; c < this.column; c++) {
                t.values[c][r] = this.values[r][c];
            }
        }
        return t;
    }


    public  void processUnits(DoubleFunction<Double> function){
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                this.values[r][c] = function.apply(this.values[r][c]);
            }
        }
    }

    public static Matrix zeros(int targetRow, int targetColumn) {
        return new Matrix(targetRow, targetColumn);
    }

    public static Matrix ones(int targetRow, int targetColumn) {
        Matrix matrix = new Matrix(targetRow, targetColumn);
        matrix.processUnits(d -> 1d);
        return matrix;
    }



    public int getRow() {
        return row;
    }

    public int getColumn() {
        return column;
    }


    public void setValue(int row, int column, double value) {
        this.values[row][column] = value;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Arrays.stream(values).forEach(a -> sb.append(Arrays.toString(a)).append("\n"));
        return sb.toString();
    }

    public static void main(String[] args) {
        Matrix random = Matrix.random(3, 1, .05);
        System.out.println(random);
        System.out.println(random.T());
    }
}
