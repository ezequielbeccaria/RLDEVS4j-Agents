package rldevs4j.agents.ac;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.util.ArrayList;
import java.util.List;

/**
 * PPO worker class.
 * @author Ezequiel Beccaría
 */
public class ACWorkerTest extends Agent {
    private ACActor actor;
    private float[][] actionSpace;
    private final List<TDTuple> trace;
    private TDTuple currentTuple;
    private float cumReward;

    public ACWorkerTest(
            int id,
            ACActor actor,
            Preprocessing preprocessing,
            float[][] actionSpace) {
        super("worker"+id, preprocessing, 0D);
        this.actor = actor;
        this.trace = new ArrayList<>();
        this.cumReward = 0;
        this.actionSpace = actionSpace;
    }
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservationINDArray();
//        System.out.println(state);
        double reward = step.getReward();
        cumReward+= reward;
        if(actor instanceof DiscreteACActor){
            int action = ((DiscreteACActor)actor).actionMax(state);
            INDArray onehotAction = Nd4j.zeros(actionSpace.length);
            onehotAction.putScalar(action, 1D);

            //store current td tuple
            currentTuple = new TDTuple(state.dup(), onehotAction, null, 0);
            return new Continuous(action, "action", EventType.action, actionSpace[action]);
        }else{
            float[] action = ((ContinuosACActor)actor).action(state);
            INDArray contAction = Nd4j.create(action);

            //store current td tuple
            currentTuple = new TDTuple(state.dup(), contAction, null, 0);
            return new Continuous(0, "action", EventType.action, action);
        }
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void episodeFinished() {
        super.episodeFinished();
        //reset worker
        this.cumReward = 0;
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
}
