package rldevs4j.agents.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public interface ACCritic {
    public void saveModel(String path) throws IOException;
    public void loadModel(String path) throws IOException;
    public INDArray output(INDArray obs);
    public Gradient gradient(INDArray states, INDArray returns);
    public void applyGradient(Gradient gradient, int batchSize);
    public INDArray getParams();
    public void setParams(INDArray p);
    public ACCritic clone();
}
