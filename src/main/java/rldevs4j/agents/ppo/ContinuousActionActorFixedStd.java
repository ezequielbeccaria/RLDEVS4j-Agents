package rldevs4j.agents.ppo;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
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
import rldevs4j.agents.ppov2.ContinuosPPOActor;
import rldevs4j.agents.ppov2.PPOActor;
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
public class ContinuousActionActorFixedStd implements ContinuosPPOActor {
    private ComputationGraph model;
    private final double LOG_STD = -0.5D; // std = e^-20 = 0.000000002
    private final double epsilonClip;
    private final double tahnActionLimit; //max sample value
    private final double entropyCoef;

    public ContinuousActionActorFixedStd(
            int obsDim,
            int actionDim,
            Double learningRate,
            Double l2,
            int hSize,
            double tahnActionLimit,
            double epsilonClip,
            double entropyCoef,
            StatsStorage statsStorage){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .weightInit(WeightInit.XAVIER)
            .l2(l2!=null?l2:0.001D)
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
            this.model.setListeners(new ScoreIterationListener(1));
            this.model.setListeners(new StatsListener(statsStorage));
        }
        this.tahnActionLimit = tahnActionLimit;
        this.epsilonClip = epsilonClip;
        this.entropyCoef = entropyCoef;
    }

    public void saveModel(String path) throws IOException{
        File file = new File(path+"actor_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"actor_model");
        this.model = ComputationGraph.load(file, true);
    }

    public ContinuousActionActorFixedStd(Map<String,Object> params){
        this((int) params.get("OBS_DIM"),
            (int) params.get("ACTION_DIM"),
            (double) params.getOrDefault("LEARNING_RATE", 1e-3),
            (double) params.getOrDefault("L2", 1e-2),
            (int) params.getOrDefault("HIDDEN_SIZE", 128),
            (double) params.get("TAHN_ACTION_LIMIT"),
            (double) params.getOrDefault("EPSILON_CLIP", 0.2D),
            (double) params.getOrDefault("ENTROPY_COEF", 0.02D),
            (StatsStorage) params.getOrDefault("STATS_STORAGE", null));
    }
    
    public INDArray[] output(INDArray obs, INDArray act){                
        Normal pi = distribution(obs);
        INDArray sample = pi.sample();
        INDArray logPi = pi.logProb(sample);        
        INDArray entropy = pi.entropy();
        return new INDArray[]{sample, logPi, entropy};
    }
    
    public float[] action(INDArray obs){
        Normal pi = distribution(obs.reshape(new int[]{1, obs.columns()}));
        INDArray sample = pi.sample();
        INDArray tanhSample = Transforms.tanh(sample);
//        sample = Transforms.max(sample, 0);
        tanhSample = tanhSample.muli(this.tahnActionLimit);
//        sample = Transforms.round(sample);
        return tanhSample.toFloatVector();
    }

    @Override
    public float[] actionOnlyMean(INDArray obs) {
        return new float[0];
    }

    private Normal distribution(INDArray obs){
        INDArray[] output = model.output(obs);
        INDArray mean = output[0];
        //Clamp LogStd
        INDArray std = Transforms.exp(Nd4j.ones(mean.shape()).muli(LOG_STD));
        return new Normal(mean, std);
    }
    
    public double train(INDArray states, INDArray actions, INDArray advantages, INDArray logOldPi, int iteration, int epoch){
        //Calculate gradient with respect to an external error
        INDArray lossPerPoint = loss(states, actions, advantages, logOldPi);
        //Do forward pass, but don't clear the input activations in each layers - we need those set so we can calculate
        // gradients based on them
        this.model.feedForward(new INDArray[]{states}, true, false);
        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        //Update the gradient: apply learning rate, momentum, etc
        //This modifies the Gradient object in-place
        Gradient g = model.backpropGradient(lossPerPoint);
        model.getUpdater().update(g, iteration, epoch, states.rows(), LayerWorkspaceMgr.noWorkspaces());
        //Get a row vector gradient array, and apply it to the parameters to update the model
        INDArray updateVector = g.gradient();
        model.params().subi(updateVector);

        Collection<TrainingListener> iterationListeners = model.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            iterationListeners.forEach((listener) -> {
                listener.iterationDone(model, iterationCount, epochCount);
            });
        }
        cgConf.setIterationCount(iterationCount + 1);

        return lossPerPoint.meanNumber().doubleValue();
    }

    @Override
    public Gradient gradient(INDArray states , INDArray actions, INDArray advantages, INDArray probOld, INDArray logProbOld) {
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

    }

    @Override
    public INDArray getParams() {
        return null;
    }

    @Override
    public void setParams(INDArray p) {

    }

    @Override
    public ComputationGraph getModel() {
        return null;
    }

    @Override
    public PPOActor clone() {
        return null;
    }

    @Override
    public double getCurrentApproxKL() {
        return 0;
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
        lossPerPoint.add(output[2].mul(this.entropyCoef));
        //Extra info
        return lossPerPoint;
    }
}
