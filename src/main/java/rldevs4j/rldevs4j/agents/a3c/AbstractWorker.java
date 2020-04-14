package rldevs4j.rldevs4j.agents.a3c;

import java.util.Collection;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;

/**
 * 
 * @author Ezequiel Beccaría
 */
public abstract class AbstractWorker extends Agent{
    protected ComputationGraph cg;
    private final double clipEpsilon = 1;

    public AbstractWorker(String name, ComputationGraph net, Preprocessing preprocessing) {
        super(name, preprocessing);
        this.cg = net;
    }
    
    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        cg.setInput(0, input);
        cg.setLabels(labels);
        cg.computeGradientAndScore();
        Collection<TrainingListener> iterationListeners = cg.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener l : iterationListeners) {
                l.onGradientCalculation(cg);
            }
        }
        return new Gradient[] {cg.gradient()};
    }
    
}
