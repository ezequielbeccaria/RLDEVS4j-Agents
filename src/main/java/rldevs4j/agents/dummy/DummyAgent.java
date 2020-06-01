
package rldevs4j.agents.dummy;

import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Continuous;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.util.Arrays;

/**
 *
 * @author Ezequiel Beccaria
 */
public class DummyAgent extends Agent{
    private double cumReward = 0D;
    private int actionSize;
    private boolean flag = true;


    public DummyAgent(String name, Preprocessing preprocessing, int actionSize) {
        super(name, preprocessing, 1D);
        this.actionSize = actionSize;
    }

    @Override
    public Event observation(Step step) {
        cumReward += step.getReward();

        float[] a = new float[actionSize];
        Arrays.fill(a, 0F);
//        if(step.getFeature(-1) == 0D)
//            a = new double[]{0D, 500D};
//        if(step.getFeature(2)<1000 && flag){
//            flag = false;
//            a = new double[]{0D, 1000D};
//        }

        Continuous action = new Continuous(
                100, 
                "action", 
                EventType.action,
                a);
//                new double[]{0D, (step.getFeature(-1)>30D && step.getFeature(-1)>35D)?1D:0D});
        
        return action;
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void trainingFinished() {

    }

    @Override
    public void clear() {
        flag = true;
        cumReward = 0D;
    }

    @Override
    public void saveModel(String path) {        
    }

    @Override
    public void loadModel(String path) {

    }

}
