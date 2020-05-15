package rldevs4j.agents.ppo;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public interface Actor {
    public void saveModel(String path) throws IOException;
    public INDArray[] output(INDArray obs, INDArray act);
    public double[] action(INDArray obs);
    public double train(INDArray states, INDArray actions, INDArray advantages, INDArray logOldPi, int iteration, int epoch);
}
