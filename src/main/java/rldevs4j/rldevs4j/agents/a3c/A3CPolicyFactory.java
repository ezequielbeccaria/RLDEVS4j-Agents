package rldevs4j.rldevs4j.agents.a3c;

import java.util.List;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.rng.Random;
import rldevs4j.base.env.msg.Event;
import rldevs4j.rldevs4j.agents.policy.Policy;
import rldevs4j.rldevs4j.agents.policy.factory.PolicyFactory;

/**
 *
 * @author Ezequiel Beccaría
 */
public class A3CPolicyFactory implements PolicyFactory{
    private final Random rnd;
    private final List<Event> actionSpace;
    private final Double softmaxClipEps;

    public A3CPolicyFactory(Random rnd, List<Event> actionSpace, Double softmaxClipEps) {
        this.rnd = rnd;
        this.actionSpace = actionSpace;
        this.softmaxClipEps = softmaxClipEps;
    }
    
    @Override
    public Policy create(MultiLayerNetwork v) {
        return new A3CPolicy(rnd, actionSpace, v, softmaxClipEps);
    }

    @Override
    public Policy create(ComputationGraph c) {
        return new A3CPolicy(rnd, actionSpace, c, softmaxClipEps);
    }

    @Override
    public Policy create() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
