package rldevs4j.agents.ppo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.distribution.NormalDistribution;
import rldevs4j.agents.utils.AgentUtils;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Ezequiel Beccaria
 */
public class ContinuousActionActorTest extends Agent {
    private ComputationGraph model;
    private final double LOG_STD_MIN = -20D; // std = e^-20 = 0.000000002
    private final double LOG_STD_MAX = 1; // std = e^1 = 2.7183
    private double tahnActionLimit; //max sample value
    protected Map<Double, double[]> appliedActions;
    private double cumReward;


    public ContinuousActionActorTest(String name, Preprocessing preprocessing, String modelPath, double tahnActionLimit) throws IOException {
        super(name, preprocessing, 1D);
        this.model = loadModel(modelPath);
        this.model.init();

        this.tahnActionLimit = tahnActionLimit;
        this.appliedActions = new HashMap<>();
    }

    public ComputationGraph loadModel(String path) throws IOException{
        File file = new File(path+"Env3PPOTrain_actor_model");
        return ComputationGraph.load(file, false);
    }

    public double[] action(INDArray obs){
        INDArray[] output = model.output(obs.reshape(new int[]{1, obs.columns()}));
        INDArray mean = output[0];
        INDArray sample = Transforms.tanh(mean);
        sample = Transforms.max(sample, 0);
        sample = sample.muli(this.tahnActionLimit);
        sample = Transforms.round(sample);
        return sample.toDoubleVector();
    }

    @Override
    public Event observation(Step step) {
        cumReward += step.getReward();
        double[] action = action(step.getObservation());
        appliedActions.put(step.getFeature(-1), action);
        return new Continuous(100, "action", EventType.action, action);
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void clear() {
        cumReward = 0D;
        appliedActions.clear();
    }

    @Override
    public void saveModel(String path) {

    }

    public Map<Double, double[]> getAppliedActions() {
        return appliedActions;
    }
}
