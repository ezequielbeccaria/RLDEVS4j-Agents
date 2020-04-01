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
 * epsilon-greedy policy implementation with epsilon exponential decay. 
 * @author Ezequiel Beccaría
 */
public class EGreedyExponentialDecay implements Policy{
    private final Random rnd;
    private final List<Event> actionSpace;
    private final MultiLayerNetwork v;
    private final ComputationGraph cg;
    private final double epsilonMin;
    private final double epsilonDecay;
    private final double epsilon;
    private double t;
    

    public EGreedyExponentialDecay(
            double initEpsilon, 
            double epsilonMin, 
            double epsilonDecay, 
            Random rnd, 
            List<Event> actionSpace, 
            MultiLayerNetwork v) {
        
        this.epsilon = initEpsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.v = v;
        this.cg = null;
        t = 0;
    }
    
    public EGreedyExponentialDecay(
            double initEpsilon, 
            double epsilonMin, 
            double epsilonDecay, 
            Random rnd, 
            List<Event> actionSpace, 
            ComputationGraph cg) {
        
        this.epsilon = initEpsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.v = null;
        this.cg = cg;
        t = 0;
    }
    
    private INDArray enabledActionsMask(List<Event> enabledActions){
        INDArray mask = Nd4j.zeros(actionSpace.size());
        enabledActions.forEach((e) -> {
            mask.putScalar(new int[]{e.getId()}, 1);
        });
        return mask;
    }

    /**
     *
     * @param state
     * @param activeActions
     * @return
     */
    @Override
    public synchronized Event chooseAction(INDArray state, List<Event> activeActions) {
        t++; // increase computational temperature
        if(rnd.nextDouble() <= Math.max(epsilonMin, epsilon*Math.exp(-epsilonDecay*t))){
            // Explore
            return activeActions.get(rnd.nextInt(activeActions.size()));
        }
        // Explote
        INDArray qsa = null;
        if(cg == null){
            qsa = v.output(state);
        }else{
            qsa = cg.output(state)[0];
        }
        
        INDArray actionMask = enabledActionsMask(activeActions);
        
        BooleanIndexing.replaceWhere(actionMask, qsa, Conditions.equals(1));
        int idx = Nd4j.getExecutioner().execAndReturn(new IMax(actionMask)).getFinalResult().intValue();
        return actionSpace.get(idx);
    }
    
    @Override
    public List<Event> getActionSpace() {
        return actionSpace;
    }
    
    @Override
    public INDArray test(INDArray state, List<Event> activeActions) {
        INDArray qsa = null;
        if(cg == null){
            qsa = v.output(state);
        }else{
            qsa = cg.output(state)[0];
        }
        return qsa;
    }
}
