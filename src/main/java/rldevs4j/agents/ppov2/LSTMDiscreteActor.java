package rldevs4j.agents.ppov2;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
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
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;
import rldevs4j.agents.utils.distribution.Categorical;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

public class LSTMDiscreteActor implements DiscretePPOActor {
    private final double paramClamp = 1D;
    private float entropyFactor;
    private float epsilonClip;
    private ComputationGraph model;
    private Random rnd;
    private float currentApproxKL;

    public LSTMDiscreteActor(String modelPath) throws IOException {
        this.rnd = Nd4j.getRandom();
        this.loadModel(modelPath);
        this.model.init();
    }

    public LSTMDiscreteActor(ComputationGraph model, float entropyFactor, float epsilonClip){
        this.rnd = Nd4j.getRandom();
        this.model = model;
        this.model.init();

        this.entropyFactor = entropyFactor;
        this.epsilonClip = epsilonClip;
    }

    public LSTMDiscreteActor(int obsDim, int actionDim, Double learningRate, Double l2, float entropyFactor, float epsilonClip, int hSize, StatsStorage statsStorage) {
        this.rnd = Nd4j.getRandom();
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
                .addLayer("policy",new DenseLayer.Builder().nIn(hSize).nOut(actionDim).activation(Activation.SOFTMAX).build(), "lastStep")
                .setOutputs("policy")
                .build();

        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }
        this.entropyFactor = entropyFactor;
        this.epsilonClip = epsilonClip;
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"LSTMDiscreteActor_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"LSTMDiscreteActor_model");
        this.model = ComputationGraph.load(file, true);
    }

    @Override
    public INDArray[] output(INDArray obs, INDArray act) {
        INDArray probs = this.model.output(obs.reshape(new int[]{obs.rows(), obs.columns(), 1}))[0];
        Categorical dist = new Categorical(probs);
        INDArray sample = dist.sample();
        INDArray logProbs = dist.logProb(act);
        INDArray entropy = dist.entropy();
        return new INDArray[]{sample, probs, logProbs, entropy};
    }

    @Override
    public int action(INDArray obs) {
        INDArray prob = this.model.output(obs.reshape(new int[]{1, obs.columns(), 1}))[0];
        Categorical dist = new Categorical(prob, null);
        return dist.sample().getInt(0);
    }

    @Override
    public int actionMax(INDArray obs) {
        INDArray prob = this.model.output(obs.reshape(new int[]{1, obs.columns(), 1}))[0];
        int idx = prob.argMax(1).getInt(0);
        return idx;
    }

    private INDArray loss(INDArray states , INDArray actions, INDArray advantages, INDArray probOld, INDArray logProbOld){
        //output[0] -> sample, output[1] -> probs, output[2] -> logProb, output[3] -> entropy
        INDArray[] output = this.output(states, actions);
        INDArray logPi = Transforms.log(output[1]);
        INDArray ratio = Transforms.exp(logPi.sub(logProbOld));
//        INDArray ratio = Transforms.exp(Transforms.log(output[1]).sub(logProbOld));
        INDArray clipAdv = ratio.dup();
        AgentUtils.clamp(clipAdv, 1D-epsilonClip, 1D+epsilonClip);
        clipAdv.muliColumnVector(advantages);
        INDArray lossPerPoint = Transforms.min(ratio.mulColumnVector(advantages), clipAdv);
        lossPerPoint.negi();
        lossPerPoint.addiColumnVector(output[3].mul(this.entropyFactor));
        //Extra info
        currentApproxKL = (logProbOld.sub(output[2])).mean().getFloat(0);
        return lossPerPoint;
    }

    @Override
    public Gradient gradient(INDArray states, INDArray actions, INDArray advantages, INDArray probOld, INDArray logProbOld) {
        model.rnnClearPreviousState();
        INDArray lossPerPoint = loss(states, actions, advantages, probOld, logProbOld);
        model.feedForward(new INDArray[]{states.reshape(new int[]{states.rows(), states.columns(), 1})}, true, false);
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

    @Override
    public void applyGradient(INDArray gradient, int batchSize) {
        //Get a row vector gradient array, and apply it to the parameters to update the model
        model.params().subi(gradient);
    }

    private INDArray gradientsClipping(INDArray output){
        INDArray clipped = output.dup();
        BooleanIndexing.replaceWhere(clipped, paramClamp, Conditions.greaterThan(paramClamp));
        BooleanIndexing.replaceWhere(clipped, -paramClamp, Conditions.lessThan(-paramClamp));
        return clipped;
    }

    @Override
    public INDArray getParams() {
        return model.params();
    }

    @Override
    public void setParams(INDArray p){
        model.setParams(p.dup());
    }

    @Override
    public ComputationGraph getModel() {
        return model;
    }

    @Override
    public PPOActor clone() {
        return new LSTMDiscreteActor(model.clone(), entropyFactor, epsilonClip);
    }

    @Override
    public double getCurrentApproxKL() {
        return currentApproxKL;
    }
}
