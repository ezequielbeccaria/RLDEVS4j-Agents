package rldevs4j.agents.ac;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.distribution.Categorical;

import java.io.File;
import java.io.IOException;

public class FFDiscreteActor implements DiscreteACActor {
    private final double entropyFactor;
    private final double paramClip = 0.5D;
    private ComputationGraph model;
    private Random rnd;

    public FFDiscreteActor(ComputationGraph model, double entropyFactor){
        this.rnd = Nd4j.getRandom();
        this.model = model;
        this.model.init();

        this.entropyFactor = entropyFactor;
    }

    public FFDiscreteActor(int obsDim, int actionDim, Double learningRate, Double l2, double entropyFactor, int hSize, StatsStorage statsStorage) {
        this.rnd = Nd4j.getRandom();
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.UNIFORM)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(paramClip)
                .l2(l2)
                .l2Bias(l2)
                .graphBuilder()
                .addInputs("in")
                .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.TANH).build(), "in")
                .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "h1")
                .addLayer("policy",new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.SOFTMAX).nIn(hSize).nOut(actionDim).build(), "h2")
                .setOutputs("policy")
                .build();

        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }

        this.entropyFactor = entropyFactor;
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"actor_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"actor_model");
        this.model = ComputationGraph.load(file, true);
    }

    @Override
    public INDArray[] output(INDArray obs, INDArray act) {
        INDArray probs = this.model.output(obs)[0];
        Categorical dist = new Categorical(probs);
        INDArray sample = dist.sample();
        INDArray logProbs = dist.logProb(act);
        INDArray entropy = dist.entropy();
        return new INDArray[]{sample, probs, logProbs, entropy};
    }

    @Override
    public int action(INDArray obs) {
        INDArray prob = this.model.output(obs.reshape(new int[]{1, obs.columns()}))[0];
        INDArray cumsum = prob.cumsum(0);
        double rndProb = rnd.nextDouble();
        int idx = BooleanIndexing.firstIndex(cumsum, Conditions.greaterThanOrEqual(rndProb)).getInt(0);
        return idx;
    }

    private INDArray loss(INDArray states , INDArray actions, INDArray advantages){
        //output[0] -> sample, output[1] -> probs, output[2] -> logProb, output[3] -> entropy
        INDArray[] output = this.output(states, actions);
        INDArray logProb = Transforms.log(output[1]);
//        INDArray lossPerPoint = logProb.mulColumnVector(advantages);
        INDArray lossPerPoint = output[2].mulColumnVector(advantages);
        INDArray entropyLoss = output[3].mul(entropyFactor);
        lossPerPoint.addiColumnVector(entropyLoss);
        //Extra info
        return lossPerPoint.negi().dup();
    }

    @Override
    public Gradient gradient(INDArray states , INDArray actions, INDArray advantages) {
        INDArray lossPerPoint = loss(states, actions, advantages);
        model.feedForward(new INDArray[]{states}, true, false);
        Gradient g = model.backpropGradient(lossPerPoint);

        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        this.model.getUpdater().update(g, iterationCount, epochCount, states.rows(), LayerWorkspaceMgr.noWorkspaces());
        this.model.update(g);

        return g;
    }

    @Override
    public void applyGradient(Gradient gradient, int batchSize, double score) {
        model.params().addi(gradient.gradient().dup());
    }

    @Override
    public ComputationGraph getModel() {
        return model;
    }

    @Override
    public double getScore() {
        return model.score();
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
    public ACActor clone() {
        return new FFDiscreteActor(model.clone(), entropyFactor);
    }
}
