package rldevs4j.rldevs4j.agents.policy.factory;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.rng.Random;
import rldevs4j.base.env.msg.Event;
import rldevs4j.rldevs4j.agents.policy.EGreedyExponentialDecay;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 *
 * @author Ezequiel Beccaría
 */
public class EGreedyExponentialDecayFactory implements PolicyFactory{
    private final double initEpsilon;
    private final double epsilonMin;
    private final double epsilonDecay;
    private final Random rnd;
    private final List<Event> actionSpace;

    public EGreedyExponentialDecayFactory(double initEpsilon, double epsilonMin, double epsilonDecay, Random rnd, List<Event> actionSpace) {
        this.initEpsilon = initEpsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        this.rnd = rnd;
        this.actionSpace = actionSpace;
    }
    
    @Override
    public Policy create(MultiLayerNetwork v){
        return new EGreedyExponentialDecay(initEpsilon, epsilonMin, epsilonDecay, rnd, actionSpace, v);
    }
    
    @Override
    public Policy create(ComputationGraph v){
        return new EGreedyExponentialDecay(initEpsilon, epsilonMin, epsilonDecay, rnd, actionSpace, v);
    }

    @Override
    public Policy create() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
