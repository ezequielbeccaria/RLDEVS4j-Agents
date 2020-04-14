package rldevs4j.rldevs4j.agents.dqn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.PersistModel;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.Step;
import rldevs4j.rldevs4j.agents.memory.ExperienceReplayBuffer;
import rldevs4j.rldevs4j.agents.memory.TDTuple;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 * DQN Agent implementation with Experience Replay, Double Model and Clipped Reward. 
 * http://www.nature.com/articles/nature14236
 *  
 * @author Ezequiel Beccaría
 */
public class DQN extends Agent implements PersistModel{
    private final INDArray initialState;
    private final int initialActionIdx;
    private Logger logger;
    private TDTuple currentTuple;
    private double currentReward;
    
    private final double discountFactor; //gamma
    private MultiLayerNetwork vf;
    private MultiLayerNetwork vfTarget;
    private final boolean clipReward;     
    private final double clipEpsilon = 1;
    private final Policy policy;
    private final ExperienceReplayBuffer<TDTuple> memory;
    private final int batchSize;
    private final int c;
    private int j;
    
    private boolean debug;

    public DQN(
            String name, 
            INDArray initialState,                    
            double discountFactor,      
            MultiLayerNetwork vf,
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
        memory = new ExperienceReplayBuffer<>(memorySize, rnd);       
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
        currentTuple.setNextState(state.dup()); //set next state
        memory.add(currentTuple.copy()); //add current tuple to currentTrace
        Event defaultAction = policy.chooseAction(state, activeActions); //Select action following policy
        if(debug) // Debuging
            logger.info(currentTuple.toStringMinimal());      
        replay(); // Experience Replay
        // Update weights to duplicated model
        this.j++;
        if(j%c==0){
            vfTarget.setParams(vf.params());
        }                
        //store current td tuple
        currentTuple = new TDTuple(state.dup(), defaultAction.copy(), null, 0);
//        }
        return defaultAction;
    }
    
    private void replay(){
        //Select tuples from memory to train the DQN
        List<TDTuple> replayTuples = memory.sample(batchSize);
        
        List<INDArray> batchXList = new ArrayList<>();
        List<INDArray> batchYList = new ArrayList<>();
        
        for(TDTuple t : replayTuples){            
            INDArray qsa = vf.output(t.getState().reshape(new int[]{1, t.getState().columns()}));
            INDArray qsa_prime = vfTarget.output(t.getNextState().reshape(new int[]{1, t.getNextState().columns()})); //qsa' from dual model
//            double reward = clipReward?clipReward(t.getReward()):t.getReward(); //Reward clipping
            qsa.putScalar(t.getAction().getId(), t.getReward()+discountFactor*Nd4j.max(qsa_prime).getDouble(0));
            batchXList.add(t.getState());
            batchYList.add(qsa);
        }
        Gradient g = this.gradient(Nd4j.vstack(batchXList), Nd4j.vstack(batchYList));
        this.applyGradient(g, batchSize);       
//        vf.fit(Nd4j.vstack(batchXList), Nd4j.vstack(batchYList));   
    }
    
    public Gradient gradient(INDArray input, INDArray labels) {
        vf.setInput(input);
        vf.setLabels(labels);
        vf.computeGradientAndScore();
        Collection<TrainingListener> valueIterationListeners = vf.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener l : valueIterationListeners) {
                    l.onGradientCalculation(vf);
            }
        }
        
        return vf.gradient();
    }
    
    private INDArray gradientsClipping(INDArray output){
        BooleanIndexing.replaceWhere(output, clipEpsilon, Conditions.greaterThan(clipEpsilon));
        BooleanIndexing.replaceWhere(output, -clipEpsilon, Conditions.lessThan(-clipEpsilon));
        return output;
    }
    
    private void applyGradient(Gradient gradient, int batchSize) {
        MultiLayerConfiguration valueConf = vf.getLayerWiseConfigurations();
        int valueIterationCount = valueConf.getIterationCount();
        int valueEpochCount = valueConf.getEpochCount();        
        vf.getUpdater().update(vf, gradient, valueIterationCount, valueEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces()); 
        vf.params().subi(gradientsClipping(gradient.gradient()));        
        valueConf.setIterationCount(valueIterationCount + 1);
    }

    @Override
    public double getTotalReward() {
        return currentReward;
    }

    @Override
    public void episodeFinished() {
        super.episodeFinished();
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
            Logger.getLogger(DQN.class.getName()).log(Level.SEVERE, null, ex);
            return false;
        }
        return true;
    }

    @Override
    public void loadNetwork(String path) {
        try {
            //Load the model
            this.vf = ModelSerializer.restoreMultiLayerNetwork(path);            
            this.vfTarget = vf.clone();
        } catch (IOException ex) {
            Logger.getLogger(DQN.class.getName()).log(Level.SEVERE, null, ex);
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
