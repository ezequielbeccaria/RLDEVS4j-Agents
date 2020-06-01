package rldevs4j.agents.ppo;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public interface PPOActor {
    public void saveModel(String path) throws IOException;
    public void loadModel(String path) throws IOException;
    public INDArray[] output(INDArray obs, INDArray act);
    public float[] action(INDArray obs);
    public double train(INDArray states, INDArray actions, INDArray advantages, INDArray logOldPi, int iteration, int epoch);
}
