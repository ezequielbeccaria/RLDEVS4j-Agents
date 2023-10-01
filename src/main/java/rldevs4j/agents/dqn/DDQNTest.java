
package rldevs4j.agents.dqn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.agents.ppo.ProximalPolicyOptimization;
import rldevs4j.agents.utils.memory.ExperienceReplayBuffer;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.io.IOException;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * DDQN Agent implementation with Experience Replay
 * 
 *  
 * @author Ezequiel Beccaría
 */
public class DDQNTest extends Agent {
    private TDTuple<Integer> currentTuple;
    private double cumReward;

    private Model model;

    private final int batchSize;
    private float[][] actionSpace;
    private Random rnd;

    public DDQNTest(
            String name,
            Preprocessing preprocessing,
            Model model,
            Map<String,Object> params) {
        super(name, preprocessing, 0D);

        rnd = Nd4j.getRandom();

        this.actionSpace = (float[][]) params.get("ACTION_SPACE");
        this.model = model;
        this.batchSize = (int) params.getOrDefault("BATCH_SIZE", 64);
    }
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservationINDArray();
        double reward = step.getReward();
        //add step reward
        this.cumReward += reward;

        int action = model.actionMax(state);

        //store current td tuple
        currentTuple = new TDTuple(state.dup(), action, null, 0);

        return new Continuous(action, "action", EventType.action, actionSpace[action]);
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
        currentTuple = null;
    }

    @Override
    public void saveModel(String path) {
        try {
            this.model.saveModel(path);
        } catch (IOException ex) {
            Logger.getLogger(ProximalPolicyOptimization.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void loadModel(String path) throws IOException {
        this.model.loadModel(path);
    }
    
    public Model getModel(){
        return model;
    }
}
