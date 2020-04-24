package rldevs4j.agents.memory;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author Ezequiel Beccaria
 */
public class TDTupleBatch {
    private final INDArray states;
    private final INDArray actions;
    private final INDArray nextStates;
    private final double[] rewards;
    private final double[] done;

    public TDTupleBatch(List<TDTuple> tuples) {
        int batchSize = tuples.size();  
        boolean recursive = tuples.get(0).getState().shape().length > 2;
        int actionSize = tuples.get(0).getAction().length;
        if(recursive){
            int stateSize = (int) tuples.get(0).getState().shape()[1];
            states = Nd4j.create(batchSize, stateSize, 1);
            nextStates = Nd4j.create(batchSize, stateSize, 1);
        }else{
            int stateSize = tuples.get(0).getState().columns();
            states = Nd4j.create(batchSize, stateSize);
            nextStates = Nd4j.create(batchSize, stateSize);
        }
        
        actions = Nd4j.zeros(batchSize, actionSize);
        rewards = new double[batchSize];
        done = new double[batchSize];
        
        for(int i=0;i<batchSize;i++){
            TDTuple t = tuples.get(i);
            if(recursive){
                states.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(t.getState());
                nextStates.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(t.getNextState());
            }else{
                states.putRow(i, t.getState());
                nextStates.putRow(i, t.getNextState());
            }           
            actions.putRow(i, Nd4j.create(t.getAction()));
            rewards[i] = t.getReward();
            done[i] = t.isDone()?0D:1D;
        }
    }

    public INDArray getStates() {
        return states;
    }

    public INDArray getActions() {
        return actions;
    }

    public INDArray getNextStates() {
        return nextStates;
    }

    public double[] getRewards() {
        return rewards;
    }

    public double[] getDone() {
        return done;
    }
    
    
}
