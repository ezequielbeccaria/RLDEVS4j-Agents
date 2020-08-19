package rldevs4j.agents.ppov2;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
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
 * PPO worker class.
 * @author Ezequiel Beccaría
 */
public class PPOWorker extends Agent {
    private PPOActor actor;
    private PPOCritic critic;
    private StandartScaler scaler;
    private float[][] actionSpace;

    private final int horizon;
    private final int epochs;
    private final float targetKl;
    private final PPO global;
    private final List<TDTuple> trace;
    private TDTuple currentTuple;
    private float cumReward;
    private final float discountFactor; //discount rate
    private final float lambdaGae;
    private int iteration=0;

    private Logger logger;
    private boolean debug;
    
    public PPOWorker(
            int id,
            PPOActor actor,
            PPOCritic critic,
            PPO global,
            float discountFactor,
            float lambdaGae,
            int horizon,
            int epochs,
            float targetKl,
            Preprocessing preprocessing,
            float[][] actionSpace,
            boolean debug) {
        super("worker"+id, preprocessing, 0D);
        this.actor = actor;
        this.critic = critic;
        this.scaler = StandartScaler.getInstance(true, true);
        this.horizon = horizon;
        this.epochs = epochs;
        this.targetKl = targetKl;
        this.discountFactor = discountFactor;
        this.lambdaGae = lambdaGae;
        this.global = global;
        this.trace = new ArrayList<>();
        this.cumReward = 0;
        this.logger = Logger.getGlobal();
        this.actionSpace = actionSpace;
        this.debug = debug;
    }
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservation();
//        System.out.println(state);
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
        if(actor instanceof DiscretePPOActor){
            int action = ((DiscretePPOActor)actor).action(state);
            INDArray onehotAction = Nd4j.zeros(actionSpace.length);
            onehotAction.putScalar(action, 1D);

            //store current td tuple
            currentTuple = new TDTuple(state.dup(), onehotAction, null, 0);
//            if(debug){ // Debuging
//                logger.info(currentTuple.toStringMinimal());
//                logger.log(Level.INFO, "Action: {0}", Arrays.toString(actionSpace[action]));
//            }
            return new Continuous(action, "action", EventType.action, actionSpace[action]);
        }else{
            float[] action = ((ContinuosPPOActor)actor).action(state);
            INDArray contAction = Nd4j.create(action);

            //store current td tuple
            currentTuple = new TDTuple(state.dup(), contAction, null, 0);
//            if(debug){ // Debuging
//                logger.info(currentTuple.toStringMinimal());
//                logger.log(Level.INFO, "Action: {0}", Arrays.toString(action));
//            }
            return new Continuous(0, "action", EventType.action, action);
        }
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    private double[] train(){
        if(trace.size()>0) {
            TDTupleBatch batch = new TDTupleBatch(trace);
            //oldPi[0] -> sample, oldPi[1] -> probs, oldPi[2] -> logProb, oldPi[3] -> entropy
            INDArray[] oldPi = actor.output(batch.getStates(), batch.getActions());
            INDArray oldValues = critic.output(batch.getStates());
            if(debug && iteration%100==0){
                System.out.println(oldPi[1]);
                System.out.println(oldValues);
            }
            //gae[0] -> returns
            //gae[1] -> advantages
            INDArray[] gae = gae(oldValues, scaler.partialFitTransform(batch.getRewards()), batch.getDone());
//            INDArray[] gae = gae(oldValues, scaler.partialFitTransform(clip(batch.getRewards(),-5F, 5F)), batch.getDone());

            INDArray gActor = null;
            INDArray gCritic = null;

            for (int i = 0; i < epochs; i++) {
                if (gActor == null) {
                    gActor = actor.gradient(batch.getStates(), batch.getActions(), gae[1], oldPi[1], oldPi[2]).gradient();
                    gCritic = critic.gradient(batch.getStates(), oldValues, gae[0]).gradient();
                } else {
                    gActor.addi(actor.gradient(batch.getStates(), batch.getActions(), gae[1], oldPi[1], oldPi[2]).gradient());
                    gCritic.addi(critic.gradient(batch.getStates(), oldValues, gae[0]).gradient());
                }
                if (actor.getCurrentApproxKL() > 1.5 * targetKl) {
                    logger.info(String.format("Early stopping at epoch %d due to reaching max kl: %f", i, actor.getCurrentApproxKL()));
                    break;
                }
            }


            global.enqueueGradient(
                    new INDArray[]{gCritic.dup(), gActor.dup()},
                    trace.size());

            INDArray[] globalParams = global.getNetsParams();

            critic.setParams(globalParams[0]);
            actor.setParams(globalParams[1]);

            trace.clear();
            iteration++;
            if(debug){
                INDArray input = Nd4j.diag(Nd4j.ones(9));
                logger.log(Level.INFO, critic.output(input).toString());
                logger.log(Level.INFO, actor.getModel().output(input)[0].toString( ));
            }
        }
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
            runningAdvantage = runningTdError + discountFactor * lambdaGae * runningAdvantage * mask[t];

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

    private double[] clip(double[] output, float min, float max){
        INDArray clipped = Nd4j.create(output);
        BooleanIndexing.replaceWhere(clipped, max, Conditions.greaterThan(max));
        BooleanIndexing.replaceWhere(clipped, min, Conditions.lessThan(min));
        return clipped.toDoubleVector();
    }
}
