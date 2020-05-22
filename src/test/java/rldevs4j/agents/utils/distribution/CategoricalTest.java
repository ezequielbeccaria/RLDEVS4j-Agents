package rldevs4j.agents.utils.distribution;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class CategoricalTest {
    double[] probs = new double[]{0.1D, 0.0D, 0.2D, 0.7D};
    private Categorical dist = new Categorical(Nd4j.create(probs));

    @Test
    void logProb() {
        INDArray sample = Nd4j.ones(1).muli(3);
        INDArray logprob = dist.logProb(sample);
        assertEquals(-0.3567D, logprob.getDouble(0));
    }

    @Test
    void entropy() {
        assertEquals(0.8018D, dist.entropy().getDouble(0));
    }
}