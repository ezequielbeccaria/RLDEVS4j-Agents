package rldevs4j.rldevs4j.agents.memory;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;
import rldevs4j.base.env.msg.Event;

/**
 * Class that represent each execution step done by the agent (Decision epoch).
 * This class is Serializable for cloning purposes.
 * @author Ezequiel Beccaría
 * @date 25/07/2017  
 */
public class TDTuple implements Serializable {
    private INDArray state;
    private Event action;    
    private INDArray nextState;    
    private double reward;    
    private boolean done;

    public TDTuple(INDArray state, Event action, INDArray nextState, double reward) {
        this(state, action, nextState, reward, false);
    }   

    public TDTuple(INDArray state, Event action, INDArray nextState, double reward, boolean done) {
        this.state = state;
        this.action = action;
        this.nextState = nextState;
        this.reward = reward;
        this.done = done;
    }

    public INDArray getState() {
        return state;
    }

    public void setState(INDArray state) {
        this.state = state;
    }

    public Event getAction() {
        return action;
    }

    public void setAction(Event action) {
        this.action = action;
    }

    public INDArray getNextState() {
        return nextState;
    }

    public void setNextState(INDArray nextState) {
        this.nextState = nextState;
    }

    public double getReward() {
        return reward;
    }
    
    public double getReward(boolean clipped) {
        if(!clipped)
            return reward;
        else
            return clipReward(reward);
    }

    public void setReward(double reward) {
        this.reward = reward;
    }    
    
    public void addReward(double reward){
        this.reward += reward;
    }

    public boolean isDone() {
        return done;
    }

    public void setDone(boolean done) {
        this.done = done;
    }
    
    public TDTuple copy(){
        return new TDTuple(state.dup(), action.copy(), nextState.dup(), reward, done);
    }

    @Override
    public String toString() {
        return "TDTuple{" + "state=" + state + ", action=" + action + ", nextState=" + nextState + ", reward=" + reward + ", done=" + done + '}';
    }
    

    public String toStringMinimal() {
        return "TDTuple{" + "state=" + state + ", action=" + action + '}';
    }
    
    private double clipReward(double reward){
        if(reward<-1D){
            return -1D;
        }else if (reward>1D){
            return 1D;
        }
        return reward;
    }
}
