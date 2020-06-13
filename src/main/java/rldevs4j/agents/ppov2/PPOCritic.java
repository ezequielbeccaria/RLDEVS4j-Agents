package rldevs4j.agents.ppov2;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public interface PPOCritic {
    public void saveModel(String path) throws IOException;
    public void loadModel(String path) throws IOException;
    public INDArray output(INDArray obs);
    public Gradient gradient(INDArray states, INDArray oldValues, INDArray returns);
    public void applyGradient(INDArray gradient, int batchSize);
    public INDArray getParams();
    public void setParams(INDArray p);
    public ComputationGraph getModel();
    public PPOCritic clone();
}
