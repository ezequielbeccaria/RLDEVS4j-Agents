package rldevs4j.agents.ppo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
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
    protected Map<Double, float[]> appliedActions;
    private double cumReward;


    public ContinuousActionActorTest(String name, Preprocessing preprocessing, String modelPath, double tahnActionLimit) throws IOException {
        super(name, preprocessing, 1D);
        File file = new File(modelPath+"Env3PPOTrain_actor_model");
        this.model = ComputationGraph.load(file, false);
        this.model.init();

        this.tahnActionLimit = tahnActionLimit;
        this.appliedActions = new HashMap<>();
    }

    public float[] action(INDArray obs){
        INDArray[] output = model.output(obs.reshape(new int[]{1, obs.columns()}));
        INDArray mean = output[0];
        INDArray sample = Transforms.tanh(mean);
        sample = Transforms.max(sample, 0);
        sample = sample.muli(this.tahnActionLimit);
        sample = Transforms.round(sample);
        return sample.toFloatVector();
    }

    @Override
    public Event observation(Step step) {
        cumReward += step.getReward();
        float[] action = action(step.getObservation());
        appliedActions.put(step.getFeature(-1), action);
        return new Continuous(100, "action", EventType.action, action);
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void trainingFinished() {

    }

    @Override
    public void clear() {
        cumReward = 0D;
        appliedActions.clear();
    }

    @Override
    public void saveModel(String path) {

    }

    @Override
    public void loadModel(String path) {

    }

    public Map<Double, float[]> getAppliedActions() {
        return appliedActions;
    }
}
