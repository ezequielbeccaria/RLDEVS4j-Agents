package rldevs4j.agents.ppov2;

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

import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;
import rldevs4j.agents.utils.distribution.Categorical;

import java.io.File;
import java.io.IOException;

public class FFDiscreteActor implements DiscretePPOActor {
    private float entropyFactor;
    private float epsilonClip;
    private ComputationGraph model;
    private float currentApproxKL;

    public FFDiscreteActor(String modelPath) throws IOException {
        this.loadModel(modelPath);
        this.model.init();
    }

    public FFDiscreteActor(ComputationGraph model, float entropyFactor, float epsilonClip){
        this.model = model;
        this.model.init();

        this.entropyFactor = entropyFactor;
        this.epsilonClip = epsilonClip;
    }

    public FFDiscreteActor(int obsDim, int actionDim, Double learningRate, Double l2, float entropyFactor, float epsilonClip, int hSize, Activation hact, StatsStorage statsStorage) {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(learningRate))
                .weightInit(WeightInit.UNIFORM)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5D)
                .l2(l2)
                .graphBuilder()
                .addInputs("in")
                .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(hact).build(), "in")
                .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(hact).build(), "h1")
//                .addLayer("h3", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "h2")
                .addLayer("policy",new DenseLayer.Builder().nIn(hSize).nOut(actionDim).activation(Activation.SOFTMAX).build(), "h2")
//                .addLayer("policy", new OutputLayer.Builder().lossFunction(new PpoFFDiscreteActorLoss(entropyFactor)).nIn(hSize).nOut(actionDim).activation(Activation.SOFTMAX).build(), "h2")
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
        File file = new File(path+"FFDiscreteActor_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"FFDiscreteActor_model");
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
//        System.out.println(prob);
//        Logger.getGlobal().info(prob.toString());
        Categorical dist = new Categorical(prob, null);
        return dist.sample().getInt(0);
    }

    @Override
    public int actionMax(INDArray obs) {
        INDArray prob = this.model.output(obs.reshape(new int[]{1, obs.columns()}))[0];
        return prob.argMax(1).getInt(0);
    }

    private INDArray loss(INDArray states , INDArray actions, INDArray advantages, INDArray probOld, INDArray logProbOld){
        //output[0] -> sample, output[1] -> probs, output[2] -> logProb, output[3] -> entropy
        INDArray[] output = this.output(states, actions);
//        INDArray logPi = output[2];
        INDArray logPi = Transforms.log(output[1]);
        INDArray ratio = Transforms.exp(logPi.sub(logProbOld), true);
//        INDArray ratio = Transforms.exp(logPi.sub(Transforms.log(probOld)), true);
        INDArray clipAdv = ratio.dup();
        AgentUtils.clamp(clipAdv, 1D-epsilonClip, 1D+epsilonClip);
        clipAdv.muliColumnVector(advantages);
        INDArray lossPerPoint = Transforms.min(ratio.mulColumnVector(advantages), clipAdv, true);
        lossPerPoint.addiColumnVector(output[3].mul(this.entropyFactor));
        lossPerPoint.negi();
        //Extra info
        currentApproxKL = (logPi.sub(logProbOld)).mul(output[1]).sum(1).mean().getFloat(0);
        return lossPerPoint;
    }

    @Override
    public Gradient gradient(INDArray states , INDArray actions, INDArray advantages, INDArray probOld, INDArray logProbOld) {
        INDArray lossPerPoint = loss(states, actions, advantages, probOld, logProbOld);

        model.feedForward(new INDArray[]{states}, true, false);
        Gradient g = model.backpropGradient(lossPerPoint);

        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        this.model.getUpdater().update(g, iterationCount, epochCount, states.rows(), LayerWorkspaceMgr.noWorkspaces());
        this.model.update(g);
        this.model.clear();
        
        return g;
    }

    @Override
    public synchronized void applyGradient(INDArray gradient, int batchSize) {
        model.params().subi(gradient.dup());
    }

    @Override
    public synchronized INDArray getParams() {
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
        return new FFDiscreteActor(model.clone(), entropyFactor, epsilonClip);
    }

    @Override
    public double getCurrentApproxKL() {
        return currentApproxKL;
    }
}
