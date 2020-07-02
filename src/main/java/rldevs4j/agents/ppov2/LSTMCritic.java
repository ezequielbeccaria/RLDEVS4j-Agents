package rldevs4j.agents.ppov2;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

public class LSTMCritic implements PPOCritic {
    private ComputationGraph model;
    private final double paramClamp = 1D;
    private float epsilonClip;

    public LSTMCritic(ComputationGraph model){
        this.model = model;
        this.model.init();
    }

    public LSTMCritic(int obsDim, Double learningRate, Double l2, float epsilonClip, int hSize, StatsStorage statsStorage){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(paramClamp)
                .l2(l2)
                .graphBuilder()
                .addInputs("in")
                .addLayer("lstm1", new LSTM.Builder().nIn(obsDim).nOut(hSize).activation(Activation.TANH).build(), "in")
                .addLayer("lstm2", new LSTM.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "lstm1")
                .addVertex("lastStep", new LastTimeStepVertex("in"), "lstm2")
                .addLayer("value", new DenseLayer.Builder().nIn(hSize).nOut(1).activation(Activation.IDENTITY).build(), "lastStep")
                .setOutputs("value")
                .build();
        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }
        this.epsilonClip = epsilonClip;
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"LSTMCritic_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"LSTMCritic_model");
        this.model = ComputationGraph.load(file, true);
    }

    @Override
    public INDArray output(INDArray obs) {
        return model.output(obs.reshape(new int[]{obs.rows(), obs.columns(), 1}))[0];
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
        model.rnnClearPreviousState();
        INDArray reshapedState = states.reshape(new int[]{states.rows(), states.columns(), 1});
        INDArray lossPerPoint = loss(reshapedState, oldValues, returns);
        model.feedForward(new INDArray[]{reshapedState}, true, false);
        Gradient g = model.backpropGradient(lossPerPoint);
        model.setScore(lossPerPoint.meanNumber().doubleValue());
        model.rnnClearPreviousState();

        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        this.model.getUpdater().update(g, iterationCount, epochCount, states.rows(), LayerWorkspaceMgr.noWorkspaces());
        this.model.update(g);

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
    public void applyGradient(INDArray gradient, int batchSize) {
        //Get a row vector gradient array, and apply it to the parameters to update the model
        model.params().subi(gradient);
    }

    @Override
    public INDArray getParams() {
        return model.params();
    }

    @Override
    public void setParams(INDArray p) {
        model.setParams(p.dup());
    }

    @Override
    public ComputationGraph getModel() {
        return model;
    }

    @Override
    public PPOCritic clone() {
        return new LSTMCritic(model.clone());
    }
}
