
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
import java.util.Arrays;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * DDQN Agent implementation with Experience Replay
 * 
 *  
 * @author Ezequiel Beccaría
 */
public class DDQN extends Agent {
    private TDTuple currentTuple;
    private double cumReward;

    private Model model;
    private final ExperienceReplayBuffer<TDTuple> memory;
    private final int batchSize;
    private float[][] actionSpace;
    private Random rnd;

    private boolean debug;
    private int iteration;
    private Logger logger;

    public DDQN(
            String name,
            Preprocessing preprocessing,
            Model model,
            Map<String,Object> params) {
        super(name, preprocessing, 0.1D);

        rnd = Nd4j.getRandom();
        memory = new ExperienceReplayBuffer<>((int) params.getOrDefault("MEMORY_SIZE", 10000), rnd);
        this.actionSpace = (float[][]) params.get("ACTION_SPACE");
        this.model = model;
        this.batchSize = (int) params.getOrDefault("BATCH_SIZE", 64);
        debug = (boolean) params.getOrDefault("DEBUG", false);
        logger = Logger.getGlobal();
        iteration = 0;
    }    
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservation();
        double reward = step.getReward();
        //add step reward
        this.cumReward += reward;
        if(currentTuple!=null){
            currentTuple.addReward(reward);
            currentTuple.setNextState(state.dup()); //set next state
            currentTuple.setDone(step.isDone());
            memory.add(currentTuple.copy()); //add current tuple to currentTrace
        }

        int action = model.action(state);

        //store current td tuple
        currentTuple = new TDTuple(state.dup(), actionSpace[action], null, 0);
        //Train the model
        model.train(memory.sample(batchSize), batchSize, iteration); // Experience Replay

        if(debug){ // Debuging
            logger.info(currentTuple.toStringMinimal());
            logger.log(Level.INFO, "Action: {0}", Arrays.toString(actionSpace[action]));
        }

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
        INDArray testObs = Nd4j.zeros(10);
        testObs.putScalar(4, 1);
        INDArray testOutput = model.test(testObs);
        System.out.println("Test Output: "+testOutput);

        cumReward = 0D;
        currentTuple = null;
        iteration++;
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
    public void loadModel(String path) {
        try {
            this.model.loadModel(path);
        } catch (IOException ex) {
            Logger.getLogger(ProximalPolicyOptimization.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
