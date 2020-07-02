package rldevs4j.agents.ppov2;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ContinuosPPOActor extends PPOActor {
    public float[] action(INDArray obs);
    public float[] actionOnlyMean(INDArray obs);
}
