package rldevs4j.rldevs4j.agents.policy.factory;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.rng.Random;
import rldevs4j.base.env.msg.Event;
import rldevs4j.rldevs4j.agents.policy.Policy;
import rldevs4j.rldevs4j.agents.policy.Softmax;

/**
 *
 * @author Ezequiel Beccaría
 */
public class SoftmaxFactory implements PolicyFactory{
    private final Random rnd;
    private final List<Event> actionSpace;

    public SoftmaxFactory(Random rnd, List<Event> actionSpace) {
        this.rnd = rnd;
        this.actionSpace = actionSpace;
    }
    
    @Override
    public Policy create(MultiLayerNetwork v){
        return new Softmax(rnd, actionSpace, v);
    }

    @Override
    public Policy create(ComputationGraph c) {
        return new Softmax(rnd, actionSpace, c);
    }

    @Override
    public Policy create() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
