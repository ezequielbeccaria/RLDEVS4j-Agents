package rldevs4j.agents.ppo;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;
import rldevs4j.agents.memory.TDTuple;
import rldevs4j.agents.memory.TDTupleBatch;

/**
 *
 * @author Ezequiel Beccaria
 */
public class ProximalPolicyOptimization extends Agent{
    private int iteration;
    private Actor actor;
    private Critic critic; 
    private List<TDTuple> trace;
    private TDTuple currentTuple;
    
    private double discountRate;
    private double lambdaGae;
    private double targetKl;
    private final int epochs;
    private int horizon;

    
    private boolean debug;
    private Logger logger;
    
    private double cumReward;

    public ProximalPolicyOptimization(
            String name, 
            Preprocessing preprocessing, 
            Actor actor,
            Critic critic,
            Map<String,Object> params) {
        super(name, preprocessing);
        
        this.actor = actor;
        this.critic = critic;
        
        trace = new ArrayList<>();
        
        discountRate = (double) params.getOrDefault("DISCOUNT_RATE", 0.99D);
        lambdaGae = (double) params.getOrDefault("LAMBDA_GAE", 0.96D);
        targetKl = (double) params.getOrDefault("TARGET_KL", 0.02D);
        epochs = (int) params.getOrDefault("EPOCHS", 5);
        horizon = (int) params.getOrDefault("HORIZON", 100);
        
        debug = (boolean) params.getOrDefault("DEBUG", false);
        logger = Logger.getGlobal();
        iteration = 0;
    }

    @Override
    public Event observation(Step step) {
        //TODO: Train when horizon is reached
        INDArray state = step.getObservation();
        double reward = step.getReward();
        //add reward to currentTuple
        cumReward+= reward;
        //compute new policy               
        if(currentTuple!=null){
            currentTuple.addReward(reward);
            currentTuple.setNextState(state.dup()); //set next state
            trace.add(currentTuple);            
            
            if(trace.size() == horizon)
                train();
        }    
        double[] action = actor.action(state);
        
        //store current td tuple
        currentTuple = new TDTuple(state.dup(), action, null, 0);
        if(debug) // Debuging
            logger.info(currentTuple.toStringMinimal());      
        return new Continuous(100, "action", EventType.action, action);
    }
    
    private double[] train(){        
        TDTupleBatch batch = new TDTupleBatch(trace);        
        //oldPi[0] -> Sample
        //oldPi[1] -> log_oldPi
        INDArray[] oldPi = actor.output(batch.getStates(), batch.getActions());
        INDArray oldValues = critic.output(batch.getStates());
        //gae[0] -> returns
        //gae[1] -> advantages
        INDArray[] gae = gae(oldValues, batch.getRewards(), batch.getDone());
        
        double actorLoss = 0;
        double criticLoss = 0;
        for (int i = 0; i < this.epochs; i++) {
            actorLoss = actor.train(batch.getStates(), batch.getActions(), gae[1], oldPi[1], iteration, epochs);
            criticLoss = critic.train(batch.getStates(), gae[0]);
            //TODO: evaluate KL divergese for early stopping
        }
        
        trace.clear();
        
        return new double[]{actorLoss, criticLoss};
    }    

    /**
     * General Advantage Estimation
     */
    private INDArray[] gae(INDArray values, double[] rewards, double[] mask){
        INDArray returns = Nd4j.zeros(rewards.length);
        INDArray advantages = Nd4j.zeros(rewards.length);
        
        double runningReturn = 0D;
        double previousValue = 0D;
        double runningAdvantage = 0D;
        
        for(int t=rewards.length-1;t>=0;t--){
            runningReturn = rewards[t] + discountRate * runningReturn * mask[t];
            double runningTdError = rewards[t] + discountRate * previousValue * mask[t] - values.getDouble(t);
            runningAdvantage = runningTdError + discountRate * lambdaGae * runningAdvantage * mask[t];
        
            returns.putScalar(t, runningReturn);
            previousValue = values.getDouble(t);
            advantages.putScalar(t, runningAdvantage);
        }
        
        return new INDArray[]{returns, advantages};
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void clear() {
        cumReward = 0D;
        iteration++;
    }
    
}
