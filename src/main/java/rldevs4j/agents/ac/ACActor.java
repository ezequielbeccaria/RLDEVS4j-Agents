package rldevs4j.agents.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public interface ACActor {
    public void saveModel(String path) throws IOException;
    public void loadModel(String path) throws IOException;
    public INDArray[] output(INDArray obs, INDArray act);
    public Gradient gradient(INDArray states , INDArray actions, INDArray advantages);
    public void applyGradient(Gradient gradient, int batchSize);
    public INDArray getParams();
    public void setParams(INDArray p);
    public ACActor clone();
}
