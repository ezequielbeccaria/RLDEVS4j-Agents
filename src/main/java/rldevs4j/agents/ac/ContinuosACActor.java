package rldevs4j.agents.ac;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ContinuosACActor extends ACActor{
    public float[] action(INDArray obs);
}
