package rldevs4j.agents.utils.memory;

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

    public TDTupleBatch(List<TDTuple> tuples, boolean discrete) {
        int batchSize = tuples.size();
        int stateSize = tuples.get(0).getState().columns();
        states = Nd4j.create(batchSize, stateSize);
        nextStates = Nd4j.create(batchSize, stateSize);

        actions = Nd4j.zeros(batchSize, discrete?1:((double[])tuples.get(0).getAction()).length);
        rewards = new double[batchSize];
        done = new double[batchSize];
        
        for(int i=0;i<batchSize;i++){
            TDTuple t = tuples.get(i);

            states.putRow(i, t.getState());
            nextStates.putRow(i, t.getNextState());

            actions.putRow(i, t.getAction() instanceof Integer?Nd4j.create(new double[]{(int)t.getAction()}):Nd4j.create((double[])t.getAction()));
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
