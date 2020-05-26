package rldevs4j.agents.utils.distribution;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class CategoricalTest {
    float deltaError = 0.0001F;
    float[] probs = new float[]{0.1F, 0.0F, 0.2F, 0.7F};
    float[][] probs2 = new float[][]{{0.1F, 0.0F, 0.2F, 0.7F}, {0.1F, 0.0F, 0.2F, 0.7F}};
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
        assertEquals(-0.3567F, logprob.getDouble(0), deltaError);
    }

    @Test
    void entropy() {
        assertEquals(0.8018F, dist.entropy().getDouble(0), deltaError);
    }

    @Test
    void logProb2() {
        float[][] expected = new float[][]{{-0.3567F},{-0.3567F}};
        INDArray sample = Nd4j.ones(2,1).muli(3);
        INDArray logprob = dist2.logProb(sample);
        for(int i=0;i<expected.length;i++)
            assertArrayEquals(expected[i], logprob.toFloatMatrix()[i], deltaError);
    }

    @Test
    void entropy2() {
        float[] expected = new float[]{0.8018F, 0.8018F};
        INDArray entropy = dist2.entropy();
        assertArrayEquals(expected, entropy.toFloatVector(), deltaError);
    }
}