package rldevs4j.agents.utils.distribution;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class CategoricalTest {
    double deltaError = 0.0001D;
    double[] probs = new double[]{0.1D, 0.0D, 0.2D, 0.7D};
    double[][] probs2 = new double[][]{{0.1D, 0.0D, 0.2D, 0.7D}, {0.1D, 0.0D, 0.2D, 0.7D}};
    private Categorical dist = new Categorical(Nd4j.create(probs));
    private Categorical dist2 = new Categorical(Nd4j.create(probs2));

    @Test
    void sample(){
        INDArray sample = dist.sample();
        int s = sample.getInt(0);
        assertTrue(s < probs.length && s >= 0);
    }

    @Test
    void sample2(){
        INDArray sample = dist2.sample();
        long[] shape = sample.shape();
        assertArrayEquals(new long[]{2, 1}, shape);
    }

    @Test
    void logProb() {
        INDArray sample = Nd4j.ones(1).muli(3);
        INDArray logprob = dist.logProb(sample);
        assertEquals(-0.3567D, logprob.getDouble(0), deltaError);
    }

    @Test
    void entropy() {
        assertEquals(0.8018D, dist.entropy().getDouble(0), deltaError);
    }

    @Test
    void logProb2() {
        double[][] expected = new double[][]{{-0.3567D},{-0.3567D}};
        INDArray sample = Nd4j.ones(2,1).muli(3);
        INDArray logprob = dist2.logProb(sample);
        for(int i=0;i<expected.length;i++)
            assertArrayEquals(expected[i], logprob.toDoubleMatrix()[i], deltaError);
    }

    @Test
    void entropy2() {
        double[] expected = new double[]{0.8018D, 0.8018D};
        INDArray entropy = dist2.entropy();
        assertArrayEquals(expected, entropy.toDoubleVector(), deltaError);
    }
}