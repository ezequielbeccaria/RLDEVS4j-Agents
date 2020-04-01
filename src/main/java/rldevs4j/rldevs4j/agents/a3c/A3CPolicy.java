
package rldevs4j.rldevs4j.agents.a3c;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import rldevs4j.base.env.msg.Event;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 *
 * @author Ezequiel Beccaría
 */
public class A3CPolicy implements Policy{
    private final Random rnd;
    private final List<Event> actionSpace;
    private final ComputationGraph cg;
    private final MultiLayerNetwork mln;
    private Double softmaxClipEps;

    public A3CPolicy(Random rnd, List<Event> actionSpace, ComputationGraph cg, Double softmaxClipEps) {
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.cg = cg;
        this.mln = null;
        
        if(softmaxClipEps != null)
            if(softmaxClipEps < 0 || softmaxClipEps > 0.5){
                throw new IllegalArgumentException("Invalid clipping epsilon: epsilon should be >= 0 (but near zero). Got: "
                    + softmaxClipEps);
            }
    }
    
    public A3CPolicy(Random rnd, List<Event> actionSpace, MultiLayerNetwork mln, Double softmaxClipEps) {
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.cg = null;
        this.mln = mln;
        
        if(softmaxClipEps != null)
           if(softmaxClipEps < 0 || softmaxClipEps > 0.5){
               throw new IllegalArgumentException("Invalid clipping epsilon: epsilon should be >= 0 (but near zero). Got: "
                   + softmaxClipEps);
           }

    }

    @Override
    public List<Event> getActionSpace() {
        return actionSpace;
    }

    @Override
    public Event chooseAction(INDArray state, List<Event> activeActions) {
        INDArray prob = null;
        if(cg != null){
            if(state.shape().length>2)
                prob = this.cg.rnnTimeStep(state)[1];
            else
                prob = this.cg.output(state.reshape(new int[]{1, state.columns()}))[1];                
        }else{
            prob = this.mln.output(state.reshape(new int[]{1, state.columns()}));
        }
        
        if(softmaxClipEps!=null){
            BooleanIndexing.replaceWhere(prob, softmaxClipEps, Conditions.lessThan(softmaxClipEps));
            BooleanIndexing.replaceWhere(prob, 1.0-softmaxClipEps, Conditions.greaterThan(1.0-softmaxClipEps));
        }
            
        INDArray cumsum = prob.cumsum(0);
        
        double rndProb = rnd.nextDouble();
        int idx = BooleanIndexing.firstIndex(cumsum, Conditions.greaterThanOrEqual(rndProb)).getInt(0);
        
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
    public INDArray test(INDArray state, List<Event> activeActions) {
        INDArray prob = null;
        if(cg != null){
            if(state.shape().length>2)
                prob = this.cg.rnnTimeStep(state)[1];
            else
                prob = this.cg.output(state.reshape(new int[]{1, state.columns()}))[1];                
        }else{
            prob = this.mln.output(state.reshape(new int[]{1, state.columns()}));
        }
        return prob;
    }
    
}
