
package rldevs4j.rldevs4j.agents.policy;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import rldevs4j.base.env.msg.Event;

/**
 * epsilon-greedy policy implementation 
 * @author Ezequiel Beccaría
 */
public class EGreedy implements Policy{
    private final Random rnd;
    private final List<Event> actionSpace;
    private final MultiLayerNetwork v;
    private final ComputationGraph g;
    private final double epsilon;

    public EGreedy(double epsilon, Random rnd, List<Event> actionSpace, MultiLayerNetwork v) {
        this.epsilon = epsilon;
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.v = v;
        this.g = null;
    }    
    
     public EGreedy(double epsilon, Random rnd, List<Event> actionSpace, ComputationGraph g) {
        this.epsilon = epsilon;
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.v = null;
        this.g = g;
    }
    
    @Override
    public synchronized Event chooseAction(INDArray state, List<Event> activeActions) {
        double r = rnd.nextDouble();
        if(r<=epsilon){
            // Explore
            return actionSpace.get(rnd.nextInt(actionSpace.size()));
        }
        // Explote
        INDArray qsa = this.output(state, null).detach();        
       
        int idx = Nd4j.getExecutioner().execAndReturn(new IMax(qsa)).getFinalResult().intValue();
        return actionSpace.get(idx);
    }
    
    private INDArray enabledActionsMask(List<Event> enabledActions){
        INDArray mask = Nd4j.zeros(actionSpace.size());
        enabledActions.forEach((e) -> {
            mask.putScalar(new int[]{e.getId()}, 1);
        });
        return mask;
    }

    @Override
    public List<Event> getActionSpace() {
        return actionSpace;
    }
    
    private INDArray output(INDArray state, INDArray actionMask){
        if(v != null){
            if(actionMask != null){
                INDArray qsa = this.v.output(
                    state.reshape(new int[]{1, state.columns()}), 
                    false, //train = false
                    null, 
                    actionMask.reshape(new int[]{1, -1}));
                BooleanIndexing.replaceWhere(qsa, -999999, Conditions.equals(0));
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
                BooleanIndexing.replaceWhere(qsa[0], -999999, Conditions.equals(0));
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
    public INDArray test(INDArray state, List<Event> activeActions) {
        INDArray qsa = this.output(state, null).detach();        
        return qsa;
    }
}
