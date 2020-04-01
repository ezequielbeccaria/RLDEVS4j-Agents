package rldevs4j.rldevs4j.agents.policy;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import static org.nd4j.linalg.ops.transforms.Transforms.*;
import rldevs4j.base.env.msg.Event;

/**
 * epsilon-greedy policy implementation with epsilon lineal decay. 
 * @author Ezequiel Beccaría
 */
public class Softmax implements Policy{
    private final Random rnd;
    private final List<Event> actionSpace;
    private final MultiLayerNetwork v;
    private final ComputationGraph g;
    private final double tau;
    
    public Softmax(
            Random rnd, 
            List<Event> actionSpace, 
            MultiLayerNetwork v){
        this(rnd, actionSpace, v, 1D);
            
    }
    
    public Softmax(
            Random rnd, 
            List<Event> actionSpace, 
            MultiLayerNetwork v,
            double tau) {
       
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.v = v;
        this.g = null;
        this.tau = tau;
    }
    
    public Softmax(
            Random rnd, 
            List<Event> actionSpace, 
            ComputationGraph g) {       
        this(rnd, actionSpace, g, 1D);
    }

    public Softmax(
            Random rnd, 
            List<Event> actionSpace, 
            ComputationGraph g,
            double tau) {
       
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.v = null;
        this.g = g;
        this.tau = tau;
    }
    
    private INDArray enabledActionsMask(List<Event> enabledActions){
        INDArray mask = Nd4j.zeros(actionSpace.size());
        enabledActions.forEach((e) -> {
            mask.putScalar(new int[]{e.getId()}, 1);
        });
        return mask;
    }

    @Override
    public Event chooseAction(INDArray state, List<Event> activeActions) { 
        INDArray qsa = null;       
        
        if(activeActions != null){
            INDArray actionMask = enabledActionsMask(activeActions);
            qsa = this.output(state, actionMask);
        }else{
            qsa = this.output(state, null);
        }        
        qsa = exp(qsa.muli(tau));
        Number qsaSum = qsa.sumNumber();
        INDArray qProb = qsa.div(qsaSum);
        INDArray cumsum = qProb.cumsum(0);
        
        double rndProb = rnd.nextDouble();
        int idx = BooleanIndexing.firstIndex(cumsum, Conditions.greaterThan(rndProb)).getInt(0);
        return actionSpace.get(idx);
    }
    
    private INDArray output(INDArray state, INDArray actionMask){
        if(v != null){
            if(actionMask != null){
                INDArray qsa = this.v.output(
                    state.reshape(new int[]{1, state.columns()}), 
                    false, //train = false
                    null, 
                    actionMask.reshape(new int[]{1, -1}));
                return qsa;
            }else{
                return this.v.output(state.reshape(new int[]{1, state.columns()}));
            }            
            
        }else{
            INDArray[] stateArray = new INDArray[1];
            INDArray[] actionMaskArray = new INDArray[1];
            
            if(state.shape().length>2)
                stateArray[0] = state;
            else
                stateArray[0] = state.reshape(new int[]{1, state.columns()});
            
            if(actionMask != null){
                actionMaskArray[0] = actionMask.reshape(new int[]{1, -1});
                INDArray[] qsa = this.g.output(
                    false, //train = false,
                    stateArray,                 
                    null, 
                    actionMaskArray);
                return qsa[0];
            }else{
                INDArray[] qsa = this.g.output(
                    false, //train = false,
                    stateArray);                
                return qsa[0];
            }          
        }        
    }
    
    @Override
    public List<Event> getActionSpace() {
        return actionSpace;
    }

    @Override
    public INDArray test(INDArray state, List<Event> activeActions) {
        INDArray qsa = null;       
        
        if(activeActions != null){
            INDArray actionMask = enabledActionsMask(activeActions);
            qsa = this.output(state, actionMask);
        }else{
            qsa = this.output(state, null);
        }        
        qsa = exp(qsa.muli(tau));    
        Number qsaSum = qsa.sumNumber();
        INDArray qProb = qsa.div(qsaSum);
        return qProb;
    }
}
