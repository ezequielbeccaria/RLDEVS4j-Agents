package rldevs4j.agents.ppov2;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

public class FFCritic implements PPOCritic {
    private ComputationGraph model;
    private final double paramClamp = 0.5D;
    private float epsilonClip;

    public FFCritic(ComputationGraph model){
        this.model = model;
        this.model.init();
    }

    public FFCritic(int obsDim, Double learningRate, Double l2, float epsilonClip, int hSize){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(l2!=null?l2:0.001D)
                .graphBuilder()
                .addInputs("in")
                .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.TANH).build(), "in")
                .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "h1")
                .addLayer("value", new DenseLayer.Builder().activation(Activation.IDENTITY).nIn(hSize).nOut(1).build(), "h2")
                .setOutputs("value")
                .build();
        model = new ComputationGraph(conf);
        model.init();

        this.epsilonClip = epsilonClip;
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"critic_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"critic_model");
        this.model = ComputationGraph.load(file, true);
    }

    @Override
    public INDArray output(INDArray obs) {
        return model.output(obs)[0];
    }

    public INDArray loss(INDArray states, INDArray oldValues, INDArray returns){
        INDArray v = model.output(states)[0];
        INDArray vClipped = oldValues.add(AgentUtils.clamp(v.dup().subi(oldValues), epsilonClip, epsilonClip));
        INDArray lossV = Transforms.pow(v.subColumnVector(returns), 2);
        INDArray lossVClipped = Transforms.pow(vClipped.subColumnVector(returns), 2);
        //Why max? -> https://github.com/openai/baselines/issues/445#issuecomment-408835567
        INDArray loss = Transforms.max(lossV, lossVClipped).muli(0.5);
        return loss;
    }

    @Override
    public Gradient gradient(INDArray states, INDArray oldValues, INDArray returns) {
        INDArray lossPerPoint = loss(states, oldValues, returns);
        model.feedForward(new INDArray[]{states}, true, false);
        Gradient g = model.backpropGradient(lossPerPoint);
        model.params().subi(gradientsClipping(g.gradient()));
        return g;
    }

    private INDArray gradientsClipping(INDArray output){
        INDArray clipped = output.dup();
        BooleanIndexing.replaceWhere(clipped, paramClamp, Conditions.greaterThan(paramClamp));
        BooleanIndexing.replaceWhere(clipped, -paramClamp, Conditions.lessThan(-paramClamp));
        return clipped;
    }

    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize
     */
    @Override
    public void applyGradient(Gradient gradient, int batchSize) {
        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        model.getUpdater().update(gradient, iterationCount, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        //Get a row vector gradient array, and apply it to the parameters to update the model
        INDArray updateVector = gradientsClipping(gradient.gradient());
        model.params().subi(updateVector);
        Collection<TrainingListener> iterationListeners = model.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            iterationListeners.forEach((listener) -> {
                listener.iterationDone(model, iterationCount, epochCount);
            });
        }
        cgConf.setIterationCount(iterationCount + 1);
    }

    @Override
    public INDArray getParams() {
        return model.params();
    }

    @Override
    public void setParams(INDArray p) {
        model.setParams(p);
    }

    @Override
    public PPOCritic clone() {
        return new FFCritic(model.clone());
    }
}
