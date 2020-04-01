package rldevs4j.rldevs4j.agents.policy;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import rldevs4j.base.env.msg.Event;

/**
 * General interface for al policies
 * @author Ezequiel Beccaría
 */
public interface Policy {
    public List<Event> getActionSpace();
    public Event chooseAction(INDArray state, List<Event> activeActions);
    public INDArray test(INDArray state, List<Event> activeActions);
}
