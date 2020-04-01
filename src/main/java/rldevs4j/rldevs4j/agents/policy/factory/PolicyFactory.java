package rldevs4j.rldevs4j.agents.policy.factory;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import rldevs4j.rldevs4j.agents.policy.Policy;

/**
 *
 * @author Ezequiel Beccaría
 */
public interface PolicyFactory {
    public Policy create(MultiLayerNetwork v);
    public Policy create(ComputationGraph c);
    public Policy create();
}
