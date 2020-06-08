package rldevs4j.agents.utils.scaler;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class StandartScalerTest {
    double[] v = new double[]{1D, 1D, -1D, -1D};
    StandartScaler ss = StandartScaler.getInstance(true, true);

    @Test
    void fit() {
        ss.fit(v);
        assertEquals(0D, ss.getMean());
        assertEquals(1D, ss.getStd());
    }

    @Test
    void partialFit() {
        ss.partialFit(v);
        assertEquals(0D, ss.getMean());
        assertEquals(1D, ss.getStd());
    }

    @Test
    void transform() {
        ss.fit(v);
        assertArrayEquals(ss.transform(v), v);
    }

    @Test
    void fitTransform() {
        assertArrayEquals(ss.fitTransform(v), v);
    }

    @Test
    void partialFitTransform() {
        assertArrayEquals(ss.partialFitTransform(v), v);
    }
}