package rldevs4j.rldevs4j.agents.a3c;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.Step;
import rldevs4j.rldevs4j.agents.memory.TDTuple;
import rldevs4j.rldevs4j.agents.memory.TDTupleBatch;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 * A3C worker class.
 * @author Ezequiel Beccaría
 */
public class A3CWorker extends AbstractWorker{
    private int stepCounter;
    private final int stepCounterMax;
    private final A3C global;
    private final boolean recurrent;
    private final INDArray initialState;
    private final int stateSize;
    private final int initialActionIdx;    
    private final List<TDTuple> trace;
    private TDTuple currentTuple;
    private double currentReward;
    private Map<String, INDArray> hiddenInitState; //hidden state where training trace starts
    
    private final Policy policy;
    private final double discountFactor; //discount rate
    
    private Logger logger;
    private boolean debug;
    
    public A3CWorker(
            int id, 
            ComputationGraph net, 
            int doNothingActionId, 
            INDArray initialState,
            Policy policy, 
            A3C global, 
            double discountFactor,
            int stepCounterMax,
            Preprocessing preprocessing) {
        
        this(id, net, doNothingActionId, initialState, policy, global, discountFactor, stepCounterMax, preprocessing, false, null);
    }

    public A3CWorker(
            int id, 
            ComputationGraph net, 
            int doNothingActionId, 
            INDArray initialState,
            Policy policy, 
            A3C global, 
            double discountFactor,
            int stepCounterMax,
            Preprocessing preprocessing,
            boolean debug,
            Logger logger) {
        super("worker"+id, doNothingActionId, net, preprocessing);
        
        this.stepCounter = 0;
        this.stepCounterMax = stepCounterMax;
        
        this.initialActionIdx = doNothingActionId;
        this.stateSize = initialState.columns();
        this.policy = policy;
        this.discountFactor = discountFactor;
        this.global = global;
        this.recurrent = global.isRecurrent();
        if(recurrent){
            this.initialState = initialState.reshape(new int[]{ 1, stateSize, 1});
            cg.rnnClearPreviousState();
        }else{
            this.initialState = initialState;
        }    
        this.trace = new ArrayList<>();
        this.currentReward = 0;
        this.currentTuple = new TDTuple(
                this.initialState, 
                this.policy.getActionSpace().get(this.initialActionIdx), 
                null,
                0);
        this.debug = debug;
        this.logger = logger;
    }
    
    @Override
    public Event observation(Step step) {
        if(recurrent && stepCounter == 0){
            hiddenInitState = this.cg.rnnGetPreviousState("lstm");
        }
        INDArray state = step.getObservation();
        List<Event> activeActions = step.getActiveActions();
        double reward = step.getReward();
        //add reward to currentTuple
        currentTuple.addReward(reward);
        //add step reward
        this.currentReward += reward;
        //compute new policy           
        //set next state
        if(recurrent)
            state = state.reshape(new int[]{ 1, stateSize, 1});
        currentTuple.setNextState(state.detach());
        //add to trace
        trace.add(currentTuple.copy());        
        //Select action following policy
        Event defaultAction = policy.chooseAction(state, activeActions);
        if(debug)
            logger.info(currentTuple.toStringMinimal());                                                     
        //store current td tuple
        currentTuple = new TDTuple(state.detach(), defaultAction.copy(), null, 0, step.isDone());            
        stepCounter++;
        if(stepCounter == stepCounterMax)
            updateGradients();        
        return defaultAction;
    }

    @Override
    public double getTotalReward() {
        return currentReward;
    }
    
    private void updateGradients(){        
        int batchSize = trace.size();
        
        if(batchSize>1){
            if(recurrent){
                updateGradientsLSTM(batchSize);
            }else{
                updateGradientsFeedFoward(batchSize);
            }
        }            
    }
    
    private void updateGradientsFeedFoward(int batchSize){
        TDTupleBatch batch = new TDTupleBatch(trace); //zip traces
        //if recurrent then train as a time serie with a batch size of 1            
        INDArray[] output = cg.output(batch.getStates());
        double r = 0D;
        
        if(!trace.get(batchSize-1).isDone()){
            r += cg.output(trace.get(batchSize-1).getNextState().reshape(new int[]{1, stateSize}))[0].getDouble(0);
        }

        double advantage = 0;
        for (int i=batchSize-1;i>=0;i--) {                
            r = batch.getRewards()[i] + discountFactor*r;      

            advantage = r - output[0].getDouble(i);

            //the critic
            output[0].putScalar(i, r);

            //the actor     
            INDArray oneHotAction = Nd4j.zeros(output[1].columns());
            oneHotAction.putScalar(batch.getEvents().get(i).getId(), 1.0); //OneHotAction Array

            double logPolicy = output[1].getDouble(i, batch.getEvents().get(i).getId())*advantage;
            oneHotAction.muli(logPolicy);
            output[1].putRow(i, oneHotAction);
        }        
        
        // calc gradients
        Gradient[] grad = this.gradient(batch.getStates(), output);
        // enqueue gradients
        global.enqueueGradient(grad, batchSize-1, cg);
        // update nets' weigths
        INDArray[] params = global.getNetsParams();
        cg.setParams(params[0]); 
        //trace clear
        this.trace.clear();
        this.stepCounter = 0;      
    }
    
    private void updateGradientsLSTM(int batchSize){
        TDTupleBatch batch = new TDTupleBatch(trace); //zip traces
        INDArray[] output = new INDArray[2];
        output[0] = Nd4j.zeros(new int[]{batchSize, 1});
        output[1] = Nd4j.zeros(new int[]{batchSize, 3});
        Map<String, INDArray> hiddenCurrentState = cg.rnnGetPreviousState("lstm");
        double r = 0D;
        
        // Set hidden state of the begining of the trace
        cg.rnnClearPreviousState();
        //if recurrent then train as a time serie with a batch size of 1     
        for(int i=0;i<batchSize;i++){
            INDArray[] o = cg.rnnTimeStep(trace.get(i).getState().reshape(new int[]{1, stateSize}));
            output[0].putRow(i, o[0]);
            output[1].putRow(i, o[1]);
        }

        if(!trace.get(batchSize-1).isDone()){
            // Get the values and policy for the nextState of the last transition
            INDArray lastNextState = trace.get(batchSize-1).getNextState().reshape(new int[]{1, stateSize});
            r += cg.rnnTimeStep(lastNextState)[0].getFloat(0);
        }
        double advantage = 0;
        for (int i=batchSize-1;i>=0;i--) {                
            r = trace.get(i).getReward() + discountFactor*r;      

            advantage = r - output[0].getFloat(i);

            //the critic
            output[0].putScalar(i, r);

            //the actor     
            INDArray oneHotAction = Nd4j.zeros(output[1].columns());
            oneHotAction.putScalar(trace.get(i).getAction().getId(), 1.0); //OneHotAction Array

            double logPolicy = output[1].getDouble(i, trace.get(i).getAction().getId())*advantage;
            oneHotAction.muli(logPolicy);
            output[1].putRow(i, oneHotAction);
        }                
        
        // calc gradients
        Gradient[] grad = this.gradient(batch.getStates(), output);
        // enqueue gradients
        global.enqueueGradient(grad, batchSize-1, cg);
        // update nets' weigths
        INDArray[] params = global.getNetsParams();
        cg.setParams(params[0]); 
        //trace clear
        this.trace.clear();
        this.stepCounter = 0;      
        // Set pre-replay hidden state to continue with the training episode
        cg.rnnSetPreviousState("lstm", hiddenCurrentState);
    }

    @Override
    public void episodeFinished() {
        super.episodeFinished();
        updateGradients();    

        //reset worker 
        this.trace.clear();
        this.stepCounter = 0;            
        this.currentReward = 0;        
        this.currentTuple = new TDTuple(initialState, this.policy.getActionSpace().get(this.initialActionIdx), null, 0);
    }
    
    private double clipping(double value, double min, double max){
        if(value<min){
            return min;
        }else if (value>max){
            return max;
        }
        return value;
    }

    /**
     *
     * @param value
     */
    @Override
    public void setDebugMode(boolean value) {
        this.debug = value;
    }

    @Override
    public void clearMemory() {
        cg = null;       
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }
}
