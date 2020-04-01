
package rldevs4j.rldevs4j.agents.policy;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import rldevs4j.base.env.msg.Event;

/**
 * Random policy implementation.
 * @author Ezequiel Beccaría
 */
public class RandomPolicy implements Policy{
    private final Random rnd;
    private final List<Event> actionSpace;

    public RandomPolicy(Random rnd, List<Event> actionSpace) {
        this.rnd = rnd;        
        this.actionSpace = actionSpace;
    }    
    
    @Override
    public synchronized Event chooseAction(INDArray state, List<Event> activeActions) {
        return activeActions.get(rnd.nextInt(activeActions.size()));
    }
    
    @Override
    public List<Event> getActionSpace() {
        return actionSpace;
    }

    @Override
    public INDArray test(INDArray state, List<Event> activeActions) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
