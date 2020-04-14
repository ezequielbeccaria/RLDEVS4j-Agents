package rldevs4j.rldevs4j.agents.drqn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.PersistModel;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.Step;
import rldevs4j.rldevs4j.agents.memory.ExperienceReplayBuffer;
import rldevs4j.rldevs4j.agents.memory.TDTuple;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 * DRQN Agent implementation with Experience Replay, Double Model and Clipped Reward. 
 * https://www.aaai.org/ocs/index.php/FSS/FSS15/paper/viewPaper/11673
 *  
 * @author Ezequiel Beccaría
 */
public class DRQN extends Agent implements PersistModel{
    private final INDArray initialState;
    private final int initialActionIdx;
    private Logger logger;
    private TDTuple currentTuple;
    private double currentReward;
    
    private final double discountFactor; //gamma
    private ComputationGraph vf;
    private ComputationGraph vfTarget;
    private final boolean clipReward;     
    private final Policy policy;
    private final ExperienceReplayBuffer<List<TDTuple>> memory;
    private List<TDTuple> currentTrace;    
    private final int batchSize;
    private final int c;
    private int j;
    
    private boolean debug;
    
    public DRQN(
            String name, 
            INDArray initialState,                    
            double discountFactor,      
            ComputationGraph vf,
            Policy policy,
            int memorySize,
            int batchSize,
            int c,
            int initialActionIdx,            
            boolean clipReward,
            Preprocessing preprocessing,
            Random rnd,
            Logger logger,            
            boolean debug) {        
        super(name, preprocessing); //pass doNothing action ID              
        this.initialState = initialState;
        this.initialActionIdx = initialActionIdx;
        this.memory = new ExperienceReplayBuffer<>(memorySize, rnd);     
        this.currentTrace = new ArrayList<>();
        this.vf = vf;
        this.vfTarget = vf.clone();
        this.policy = policy;              
        this.batchSize = batchSize;
        this.discountFactor = discountFactor;     
        this.c = c;
        this.j = 0;
        this.clipReward = clipReward;
        this.logger = logger;
        this.debug = debug;        
        
        currentTuple = new TDTuple(this.initialState.dup(), this.policy.getActionSpace().get(this.initialActionIdx), null, 0);
    }    
    
    @Override
    public Event observation(Step step) {
        INDArray state = step.getObservation();
        List<Event> activeActions = step.getActiveActions();
        double reward = step.getReward();
        //add reward to currentTuple
        currentTuple.addReward(reward);
        //add step reward
        this.currentReward += reward;
        //compute new policy                 
        INDArray rstate = state.reshape(new int[]{ 1, state.columns(), 1});
        currentTuple.setNextState(state.dup()); //set next state
        currentTrace.add(currentTuple.copy()); //add current tuple to currentTrace        
        Event defaultAction = policy.chooseAction(rstate, activeActions); //Select action following policy        
        if(this.memory.size()>=this.batchSize){            
            replay(); // Experience Replay
        }
            
        // Update weights to duplicated model
        this.j++;
        if(j%c==0){
            vfTarget.setParams(vf.params());
        }                
        //store current td tuple
        currentTuple = new TDTuple(state.dup(), defaultAction.copy(), null, 0);
        if(debug){ // Debuging
            logger.info(currentTuple.toStringMinimal());      
            logger.info(policy.test(rstate, activeActions).toString());      
        }
        return defaultAction;
    }
    
    private void replay(){
        //Select tuples from memory to train the LSTM
        List<TDTuple> replayTuples = memory.sample(batchSize).get(0);
        
        LSTM lstmLayerVf = ((LSTM)vf.getLayer("lstm"));
        LSTM lstmLayerVfTarget = ((LSTM)vfTarget.getLayer("lstm"));
        
        Map<String, INDArray> vfCurrentState = lstmLayerVf.rnnGetPreviousState();
        
        lstmLayerVf.rnnClearPreviousState();
//        lstmLayerVfTarget.rnnClearPreviousState();
        
        List<INDArray> batchXList = new ArrayList<>();
        List<INDArray> batchYList = new ArrayList<>();
        
        for(TDTuple t : replayTuples){            
            INDArray stateReshaped = t.getState().reshape(new int[]{1, t.getState().columns(), 1});
            
            INDArray qsa = vf.output(stateReshaped)[0];
            int qsa_max_action_arg = Nd4j.getExecutioner().execAndReturn(new IMax(qsa)).getFinalResult().intValue();
            
            //Set vf current hidden state to target network to get next state qsa
            lstmLayerVfTarget.rnnSetPreviousState(lstmLayerVf.rnnGetPreviousState());
            INDArray qsa_prime = vfTarget.output(t.getNextState().reshape(new int[]{1, t.getNextState().columns(), 1}))[0]; //qsa' from dual model
            
            double reward = clipReward?clipReward(t.getReward()):t.getReward(); //Reward clipping
            qsa.putScalar(t.getAction().getId(), reward+discountFactor*qsa_prime.getDouble(qsa_max_action_arg));
            batchXList.add(stateReshaped);
            batchYList.add(qsa);
        }
        
        INDArray[] batchXArray = new INDArray[1];
        batchXArray[0] = Nd4j.vstack(batchXList);
        INDArray[] batchYArray = new INDArray[1];
        batchYArray[0] = Nd4j.vstack(batchYList);
        vf.fit(batchXArray, batchYArray);   
        
        // Set pre-replay hidden state to continue with the training episode
        lstmLayerVf.rnnSetPreviousState(vfCurrentState);
    }

    @Override
    public double getTotalReward() {
        return currentReward;
    }

    @Override
    public void episodeFinished() {
        super.episodeFinished();
        memory.add(new ArrayList<>(currentTrace));
        currentTrace.clear();
        currentTuple = new TDTuple(this.initialState.dup(), this.policy.getActionSpace().get(this.initialActionIdx), null, 0);
        currentReward = 0D;        
    }
    
    private double clipReward(double reward){
        if(reward<-1D){
            return -1D;
        }else if (reward>1D){
            return 1D;
        }
        return reward;
    }
    
    @Override
    public boolean saveNetwork(String path) {
        try {
            File valueModelFile = new File(path);
            //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
            boolean saveUpdater = true;         
            ModelSerializer.writeModel(vf, valueModelFile, saveUpdater);
        } catch (IOException ex) {
            Logger.getLogger(DRQN.class.getName()).log(Level.SEVERE, null, ex);
            return false;
        }
        return true;
    }

    @Override
    public void loadNetwork(String path) {
        try {
            //Load the model
            this.vf = ModelSerializer.restoreComputationGraph(path);            
            this.vfTarget = vf.clone();
        } catch (IOException ex) {
            Logger.getLogger(DRQN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void setDebugMode(boolean value) {
        this.debug = value;
    }

    @Override
    public void clear() {
        memory.clear();
        vf = null;
        vfTarget = null;
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }
}
