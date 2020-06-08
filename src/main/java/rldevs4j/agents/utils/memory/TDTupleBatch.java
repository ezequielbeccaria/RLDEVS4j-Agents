package rldevs4j.agents.utils.memory;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
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
        int stateSize = tuples.get(0).getState().columns();
        states = Nd4j.create(batchSize, stateSize);
        nextStates = Nd4j.create(batchSize, stateSize);

        long actDim = 0;
        if(tuples.get(0).getAction() instanceof NDArray){
            INDArray a = (INDArray) tuples.get(0).getAction();
            actDim = a.length();
        } else if (tuples.get(0).getAction() instanceof Float[]){
            Float[] a = (Float[]) tuples.get(0).getAction();
            actDim = a.length;
        } else {
            actDim = 1;
        }
        actions = Nd4j.zeros(batchSize, actDim);
        rewards = new double[batchSize];
        done = new double[batchSize];
        
        for(int i=0;i<batchSize;i++){
            TDTuple t = tuples.get(i);

            states.putRow(i, t.getState());
            nextStates.putRow(i, t.getNextState());

            INDArray a = null;
            if(t.getAction() instanceof NDArray){
                a = (INDArray) t.getAction();
            } else if (t.getAction() instanceof Float[] || t.getAction() instanceof float[]){
                a = Nd4j.create((float[])t.getAction());
            } else {
                a = Nd4j.create(new float[]{(int)t.getAction()});
            }
            actions.putRow(i, a);
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
