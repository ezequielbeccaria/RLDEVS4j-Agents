package rldevs4j.agents.utils.scaler;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

public class StandartScaler {
    private static StandartScaler INSTANCE;
    private double mean;
    private double s;
    private double n;
    private boolean useMean;
    private boolean useStd;

    private StandartScaler() {
        this(true, true);
    }

    private StandartScaler(boolean useMean, boolean useStd) {
        mean = 0D;
        s = 0D;
        n = 0;
        this.useMean = useMean;
        this.useStd = useStd;
    }

    public static StandartScaler getInstance(boolean useMean, boolean useStd){
        if (INSTANCE == null) {
            // Thread Safe. Might be costly operation in some case
            synchronized (StandartScaler.class) {
                if (INSTANCE == null) {
                    INSTANCE = new StandartScaler(useMean, useStd);
                }
            }
        }
        return INSTANCE;
    }

    /**
     * Compute the mean and std to be used for later scaling.
     * @param v
     */
    public synchronized void fit(double[] v){
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
    public synchronized void partialFit(double[] v){
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
    public synchronized double[] transform(double[] v){
        double[] scaled = new double[v.length];
        double cm = useMean?mean:0;
        double cs = useStd?s:0;
        for(int i=0;i<v.length;i++){
            scaled[i]= cs!=0D? (v[i]-cm)/Math.sqrt(cs/n) : (v[i]-cm);
        }
        return scaled;
    }

    /**
     * Fit to data, then transform it.
     * @param v
     * @return
     */
    public synchronized double[] fitTransform(double[] v){
        fit(v);
        return transform(v);
    }

    /**
     * Online computation of mean and std on X, then transform it.
     * @param v
     * @return
     */
    public synchronized double[] partialFitTransform(double[] v){
        partialFit(v);
        return transform(v);
    }

    public synchronized double partialFitTransform(double v){
        partialFit(new double[]{v});
        return transform(new double[]{v})[0];
    }

    public synchronized double getMean() {
        return mean;
    }

    public synchronized double getStd(){
        return Math.sqrt(s/n);
    }
}
