package rldevs4j.rldevs4j.agents.memory;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import rldevs4j.base.env.msg.Event;

/**
 *
 * @author Ezequiel Beccaria
 */
public class TDTupleBatch {
    private final INDArray states;
    private final List<Event> events;
    private final INDArray nextStates;
    private final double[] rewards;
    private final boolean[] doneFlags;

    public TDTupleBatch(List<TDTuple> tuples) {
        int batchSize = tuples.size();  
        boolean recursive = tuples.get(0).getState().shape().length > 2;
        if(recursive){
            int stateSize = (int) tuples.get(0).getState().shape()[1];
            states = Nd4j.create(batchSize, stateSize, 1);
            nextStates = Nd4j.create(batchSize, stateSize, 1);
        }else{
            int stateSize = tuples.get(0).getState().columns();
            states = Nd4j.create(batchSize, stateSize);
            nextStates = Nd4j.create(batchSize, stateSize);
        }
        
        events = new ArrayList<>(batchSize);        
        rewards = new double[batchSize];
        doneFlags = new boolean[batchSize];
        
        for(int i=0;i<batchSize;i++){
            TDTuple t = tuples.get(i);
            if(recursive){
                states.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(t.getState());
                nextStates.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(t.getNextState());
            }else{
                states.putRow(i, t.getState());
                nextStates.putRow(i, t.getNextState());
            }           
            events.add(t.getAction());            
            rewards[i] = t.getReward();
            doneFlags[i] = t.isDone();
        }
    }

    public INDArray getStates() {
        return states;
    }

    public List<Event> getEvents() {
        return events;
    }

    public INDArray getNextStates() {
        return nextStates;
    }

    public double[] getRewards() {
        return rewards;
    }

    public boolean[] getDoneFlags() {
        return doneFlags;
    }
    
    
}
