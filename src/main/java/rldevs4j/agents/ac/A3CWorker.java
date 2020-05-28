package rldevs4j.agents.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.agents.utils.memory.TDTupleBatch;
import rldevs4j.agents.utils.scaler.StandartScaler;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.util.ArrayList;
import java.util.Arrays;
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
    private double[][] actionSpace;

    private final int horizon;
    private final A3C global;
    private final List<TDTuple> trace;
    private TDTuple currentTuple;
    private double cumReward;
    private final double discountFactor; //discount rate

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
            double[][] actionSpace) {
        super("worker"+id, preprocessing, 1D);
        this.actor = actor;
        this.critic = critic;
        this.scaler = new StandartScaler();
        this.horizon = horizon;
        this.discountFactor = discountFactor;
        this.global = global;
        this.trace = new ArrayList<>();
        this.cumReward = 0;
        this.logger = Logger.getGlobal();
        this.actionSpace = actionSpace;
    }
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservation();
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

        //store current td tuple
        currentTuple = new TDTuple(state.dup(), action, null, 0);
        if(debug){ // Debuging
            logger.info(currentTuple.toStringMinimal());
            logger.log(Level.INFO, "Action: {0}", Arrays.toString(actionSpace[action]));
        }
        return new Continuous(100, "action", EventType.action, actionSpace[action]);
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    private double[] train(){
        TDTupleBatch batch = new TDTupleBatch(trace, true);

        INDArray oldValues = critic.output(batch.getStates());
        //gae[0] -> returns
        //gae[1] -> advantages
        INDArray[] gae = gae(oldValues, scaler.partialFitTransform(batch.getRewards()), batch.getDone());

        Gradient gActor = actor.gradient(batch.getStates(), batch.getActions(), gae[1]);
        Gradient gCritic = critic.gradient(batch.getStates(), gae[0]);

        global.enqueueGradient(new Gradient[]{gCritic, gActor}, trace.size());
        INDArray[] globalParams = global.getNetsParams();

        actor.setParams(globalParams[0]);
        critic.setParams(globalParams[1]);

        trace.clear();
        return new double[]{0};
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
            runningReturn = rewards[t] + discountFactor * runningReturn * mask[t];
            double runningTdError = rewards[t] + discountFactor * previousValue * mask[t] - values.getDouble(t);
            runningAdvantage = runningTdError + discountFactor * runningAdvantage * mask[t];

            returns.putScalar(t, runningReturn);
            previousValue = values.getDouble(t);
            advantages.putScalar(t, runningAdvantage);
        }

        return new INDArray[]{returns, advantages};
    }

    @Override
    public void episodeFinished() {
        super.episodeFinished();
        train();
        //reset worker 
        this.trace.clear();
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
