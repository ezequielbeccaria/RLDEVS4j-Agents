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
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;
import rldevs4j.agents.utils.distribution.Normal;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;

/**
 *
 * @author Ezequiel Beccaria
 */
public class ContinuousActorFixedStd implements ContinuosPPOActor {
    private final double paramClamp = 1D;
    private float entropyFactor;
    private float epsilonClip;
    private ComputationGraph model;
    private final float LOG_STD = -0.5F; // std = e^-20 = 0.000000002
    private float tahnActionLimit; //max sample value
    private float currentApproxKL;

    public ContinuousActorFixedStd(String modelPath) throws IOException {
        this.loadModel(modelPath);
        this.model.init();
    }

    public ContinuousActorFixedStd(ComputationGraph model, float entropyFactor, float epsilonClip, float tahnActionLimit) {
        this.model = model;
        this.model.init();
        this.entropyFactor = entropyFactor;
        this.epsilonClip = epsilonClip;
        this.tahnActionLimit = tahnActionLimit;
    }

    public ContinuousActorFixedStd(
            int obsDim,
            int actionDim,
            Double learningRate,
            Double l2,
            float tahnActionLimit,
            float epsilonClip,
            float entropyFactor,
            int hSize,
            StatsStorage statsStorage){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(paramClamp)
            .l2(l2)
            .graphBuilder()
            .addInputs("in")
            .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.RELU).build(), "in")
            .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.RELU).build(), "h1")
            .addLayer("mean", new DenseLayer.Builder().nIn(hSize).nOut(actionDim).activation(Activation.RELU).build(), "h2")
            .setOutputs("mean")
            .build();

        this.model = new ComputationGraph(conf);
        this.model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }

        this.tahnActionLimit = tahnActionLimit;
        this.epsilonClip = epsilonClip;
        this.entropyFactor = entropyFactor;
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"FFContinuousActor_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"FFContinuousActor_model");
        this.model = ComputationGraph.load(file, true);
    }
    
    public INDArray[] output(INDArray obs, INDArray act){                
        Normal pi = distribution(obs);
        INDArray sample = pi.sample();
        INDArray logPi = pi.logProb(act);
        INDArray entropy = pi.entropy();
        return new INDArray[]{sample, logPi, entropy};
    }
    
    public float[] action(INDArray obs){
        Normal pi = distribution(obs.reshape(new int[]{1, obs.columns()}));
        INDArray sample = pi.sample();
        INDArray tanhSample = Transforms.tanh(sample);
        tanhSample = tanhSample.muli(this.tahnActionLimit);
        tanhSample = Transforms.max(tanhSample, 0);
        return tanhSample.toFloatVector();
    }

    @Override
    public float[] actionOnlyMean(INDArray obs) {
        INDArray[] output = model.output(obs);
        INDArray mean = output[0];
        return mean.toFloatVector();
    }

    private Normal distribution(INDArray obs){
        INDArray[] output = model.output(obs);
        INDArray mean = output[0];
        //Clamp LogStd
        INDArray std = Transforms.exp(Nd4j.ones(mean.shape()).muli(LOG_STD));
        return new Normal(mean, std);
    }

    private INDArray loss(INDArray states , INDArray actions, INDArray advantages, INDArray logOldPi){
        //output[0] -> sample, output[1] -> logProb, output[2] -> entropy
        INDArray[] output = this.output(states, actions);
        INDArray ratio = Transforms.exp(output[1].sub(logOldPi));
        INDArray clipAdv = ratio.dup();
        AgentUtils.clamp(clipAdv, 1D-epsilonClip, 1D+epsilonClip);
        clipAdv.muliColumnVector(advantages);
        INDArray lossPerPoint = Transforms.min(ratio.mulColumnVector(advantages), clipAdv);
        lossPerPoint.negi();
        lossPerPoint.add(output[2].mul(this.entropyFactor));
        //Extra info
        currentApproxKL = (logOldPi.sub(output[2])).mean().getFloat(0);
        return lossPerPoint;
    }

    @Override
    public Gradient gradient(INDArray states , INDArray actions, INDArray advantages, INDArray logProbOld) {
        INDArray lossPerPoint = loss(states, actions, advantages, logProbOld);
        model.feedForward(new INDArray[]{states}, true, false);
        Gradient g = model.backpropGradient(lossPerPoint);
        model.setScore(lossPerPoint.meanNumber().doubleValue());

        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        this.model.getUpdater().update(g, iterationCount, epochCount, states.rows(), LayerWorkspaceMgr.noWorkspaces());
        this.model.update(g);

        return g;
    }
    
    @Override
    public void applyGradient(INDArray gradient, int batchSize) {
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
    public PPOActor clone() {
        return new ContinuousActorFixedStd(model.clone(), entropyFactor, epsilonClip, tahnActionLimit);
    }

    @Override
    public double getCurrentApproxKL() {
        return currentApproxKL;
    }
}
