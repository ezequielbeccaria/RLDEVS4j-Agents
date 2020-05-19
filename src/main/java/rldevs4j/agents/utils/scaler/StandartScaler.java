package rldevs4j.agents.utils.scaler;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

public class StandartScaler {
    private double mean;
    private double s;
    private double n;

    public StandartScaler() {
        mean = 0D;
        s = 0D;
        n = 0;
    }

    /**
     * Compute the mean and std to be used for later scaling.
     * @param v
     */
    public void fit(double[] v){
        INDArray a = Nd4j.create(v);
        n = v.length;
        mean = Arrays.stream(v).sum()/n;
        s = 0D;
        for(int i=0;i<v.length;i++)
            s += Math.pow(v[i] - mean, 2);
    }

    /**
     * Online computation of mean and std on X for later scaling.
     * @param v
     */
    public void partialFit(double[] v){
        for(int i=0;i<v.length;i++){
            double meanPrev = mean;
            n += 1;
            mean = meanPrev + (v[i] - meanPrev)/n;
            s = s + (v[i] - meanPrev)*(v[i] - mean);
        }
    }

    /**
     * Perform standardization by centering and scaling
     * @param v
     * @return
     */
    public double[] transform(double[] v){
        double[] scaled = new double[v.length];
        for(int i=0;i<v.length;i++){
            scaled[i]=s!=0D?(v[i]-mean)/Math.sqrt(s/n):(v[i]-mean);
        }
        return scaled;
    }

    /**
     * Fit to data, then transform it.
     * @param v
     * @return
     */
    public double[] fitTransform(double[] v){
        fit(v);
        return transform(v);
    }

    /**
     * Online computation of mean and std on X, then transform it.
     * @param v
     * @return
     */
    public double[] partialFitTransform(double[] v){
        partialFit(v);
        return transform(v);
    }

    public double partialFitTransform(double v){
        partialFit(new double[]{v});
        return transform(new double[]{v})[0];
    }

    public double getMean() {
        return mean;
    }

    public double getStd(){
        return Math.sqrt(s/n);
    }
}
