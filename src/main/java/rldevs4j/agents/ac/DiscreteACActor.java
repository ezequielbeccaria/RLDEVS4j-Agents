package rldevs4j.agents.ac;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface DiscreteACActor extends ACActor{
    public int action(INDArray obs);
    public int actionMax(INDArray obs);
}
