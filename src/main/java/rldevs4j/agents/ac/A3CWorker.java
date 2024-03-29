package rldevs4j.agents.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.agents.utils.memory.TDTupleBatch;
import rldevs4j.agents.utils.scaler.StandartScaler;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.*;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A3C worker class.
 * @author Ezequiel Beccaría
 */
public class A3CWorker extends Agent {
    private DiscreteACActor actor;
    private ACCritic critic;
    private StandartScaler scaler;
    private float[][] actionSpace;

    private final int horizon;
    private final A3C global;
    private final List<TDTuple> trace;
    private TDTuple currentTuple;
    private double cumReward;
    private final double discountFactor; //discount rate

    private boolean firstTime;

    private Logger logger;
    private boolean debug;
    
    public A3CWorker(
            int id,
            DiscreteACActor actor,
            ACCritic critic,
            A3C global,
            double discountFactor,
            int horizon,
            Preprocessing preprocessing,
            float[][] actionSpace,
            boolean debug) {
        super("worker"+id, preprocessing, 0D);
        this.actor = actor;
        this.critic = critic;
        this.scaler = StandartScaler.getInstance(false, true);
        this.horizon = horizon;
        this.discountFactor = discountFactor;
        this.global = global;
        this.trace = new ArrayList<>();
        this.cumReward = 0;
        this.logger = Logger.getGlobal();
        this.actionSpace = actionSpace;
        this.firstTime = true;
        this.debug = debug;
    }
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservationINDArray();
        double reward = step.getReward();
        cumReward+= reward;
        //compute new policy
        if(currentTuple!=null){
            currentTuple.addReward(reward);
            currentTuple.setNextState(state.dup()); //set next state
            currentTuple.setDone(step.isDone());
            trace.add(currentTuple);

            if(trace.size() == horizon)
                train();
        }
        int action = actor.action(state);
        INDArray onehotAction = Nd4j.zeros(actionSpace.length);
        onehotAction.putScalar(action, 1D);

        //store current td tuple
        currentTuple = new TDTuple(state.dup(), onehotAction, null, 0);
//        if(debug){ // Debuging
//            logger.info(currentTuple.toStringMinimal());
//            logger.log(Level.INFO, "Action: {0}", Arrays.toString(actionSpace[action]));
//        }
        return new Continuous(action, "action", EventType.action, actionSpace[action]);
//        return new Categorical<Integer>(action, "action", EventType.action, action);
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    private double[] train(){
        if(trace.size()>0){
            TDTupleBatch batch = new TDTupleBatch(trace);

            INDArray oldValues = critic.output(batch.getStates());
            //gae[0] -> returns
            //gae[1] -> advantages
            INDArray[] gae = advantageEStimation(oldValues, scaler.partialFitTransform(batch.getRewards()), batch.getDone());

            Gradient gActor = actor.gradient(batch.getStates(), batch.getActions(), gae[1]);
            Gradient gCritic = critic.gradient(batch.getStates(), gae[0]);

            global.enqueueGradient(new Gradient[]{gCritic, gActor}, trace.size(), new double[]{critic.getScore(), actor.getScore()});
            INDArray[] globalParams = global.getNetsParams();

            critic.setParams(globalParams[0]);
            actor.setParams(globalParams[1]);

            firstTime = false;
            trace.clear();

            if(debug){
                INDArray input = Nd4j.diag(Nd4j.ones(9));
                logger.log(Level.INFO, critic.output(input).toString());
                logger.log(Level.INFO, actor.getModel().output(input)[0].toString( ));
            }
        }
        return new double[]{0};
    }

    /**
     * Advantage Estimation
     */
    private INDArray[] advantageEStimation(INDArray values, double[] rewards, double[] mask){
        INDArray returns = Nd4j.zeros(rewards.length);
        INDArray advantages = Nd4j.zeros(rewards.length);
        double cum_reward = 0F;

        for(int t=rewards.length-1;t>=0;t--){
            cum_reward += rewards[t];
            returns.putScalar(t, rewards[t] - discountFactor * values.getDouble(t) * mask[t]);
            advantages.putScalar(t, cum_reward - discountFactor * values.getDouble(t) * mask[t]);
        }

        return new INDArray[]{returns, advantages};
    }

    @Override
    public void episodeFinished() {
        super.episodeFinished();
        train();
        //reset worker 
        this.trace.clear();
        this.currentTuple = null;
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
