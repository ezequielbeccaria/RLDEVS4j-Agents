package rldevs4j.agents.ppov2;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface DiscretePPOActor extends PPOActor {
    public int action(INDArray obs);
}
