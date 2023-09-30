package rldevs4j.agents.ppov2;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestAgent extends Agent {
    private FFDiscreteActor actor;
    private float cumReward;
    private TDTuple currentTuple;
    private final List<TDTuple> trace;
    private float[][] actionSpace;
    protected Map<Double, float[]> appliedActions;

    public TestAgent(String modelPath, Preprocessing prep, float[][] actionSpace){
        super("TestDiscretePPO", prep, 1D);
        try {
            actor = new FFDiscreteActor(modelPath);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
        this.actionSpace = actionSpace;
        cumReward = 0;
        this.trace = new ArrayList<>();
        this.appliedActions = new HashMap<>();
    }

    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservationINDArray();
        System.out.println(state);
        double reward = step.getReward();
        cumReward+= reward;
        //compute new policy
        if(currentTuple!=null){
            currentTuple.addReward(reward);
            currentTuple.setNextState(state.dup()); //set next state
            currentTuple.setDone(step.isDone());
            trace.add(currentTuple);
        }
        int action = actor.actionMax(state);
        //store current td tuple
        currentTuple = new TDTuple(state.dup(), action, null, 0);
        appliedActions.put(step.getFeature(-1), actionSpace[action]);
        return new Continuous(  100, "action", EventType.action, actionSpace[action]);
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void trainingFinished() {
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @Override
    public void clear() {

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
