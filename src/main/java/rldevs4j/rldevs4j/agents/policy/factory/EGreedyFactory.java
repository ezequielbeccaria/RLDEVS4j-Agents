package rldevs4j.rldevs4j.agents.policy.factory;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.rng.Random;
import rldevs4j.base.env.msg.Event;
import rldevs4j.rldevs4j.agents.policy.EGreedy;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 *
 * @author Ezequiel Beccaría
 */
public class EGreedyFactory implements PolicyFactory{
    private final double epsilon;
    private final Random rnd;
    private final List<Event> actionSpace;

    public EGreedyFactory(double epsilon, Random rnd, List<Event> actionSpace) {
        this.epsilon = epsilon;
        this.rnd = rnd;
        this.actionSpace = actionSpace;
    }
    
    @Override
    public Policy create(MultiLayerNetwork v){
        return new EGreedy(epsilon, rnd, actionSpace, v);
    }
    
    @Override
    public Policy create(ComputationGraph cg){
        return new EGreedy(epsilon, rnd, actionSpace, cg);
    }

    @Override
    public Policy create() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
