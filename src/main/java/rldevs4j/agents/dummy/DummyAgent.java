
package rldevs4j.agents.dummy;

import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

/**
 *
 * @author Ezequiel Beccaria
 */
public class DummyAgent extends Agent{
    private double cumReward = 0D;

    public DummyAgent(String name, Preprocessing preprocessing) {
        super(name, preprocessing);
    }

    @Override
    public Event observation(Step step) {
        cumReward += step.getReward();
               
        Continuous action = new Continuous(
                100, 
                "action", 
                EventType.action, 
                new double[]{0D, (step.getFeature(-1)>30D && step.getFeature(-1)>35D)?1D:0D});
        
        return action;
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void clear() {
        cumReward = 0D;
    }
    
}
