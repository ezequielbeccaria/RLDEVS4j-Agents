package rldevs4j.agents.utils.distribution;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Distribution {
    public INDArray sample();
    public INDArray logProb(INDArray sample);
    public INDArray entropy();
}
