package rldevs4j.agents.utils.distribution;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;

/**
 * Creates a categorical distribution parameterized by either @param `probs` or @param `logits` (but not both).
 */
public class Categorical implements Distribution{
    private INDArray probs;
    private INDArray logits;
    private final double eps = Nd4j.EPS_THRESHOLD;
    private final Random rnd;

    public Categorical(INDArray probs) {
        this(probs, null);
    }

    public Categorical(INDArray probs, INDArray logits) {
        if((probs == null) && (logits == null))
            throw new RuntimeException("Either `probs` or `logits` must be specified, but not both.");
        if(probs != null){
            if(probs.shape().length < 1)
                throw new RuntimeException("`probs` parameter must be at least one-dimensional.");
            if(probs.rank()==1)
                probs = probs.reshape(1, -1);
            this.probs = clampProbs(probs);
        }else{
            if(logits.shape().length < 1)
                throw new RuntimeException("`logits` parameter must be at least one-dimensional.");
            if(logits.rank()==1)
                logits = logits.reshape(1, -1);
            this.logits = logits.sub(AgentUtils.logSumExp(logits));
        }
        rnd = Nd4j.getRandomFactory().getNewRandomInstance();
    }

    @Override
    public INDArray sample() {
        if(logits != null){
            probs = Transforms.softmax(logits);
        }
        INDArray cumProbs = probs.cumsum(-1);
        long[][] sample = new long[probs.rows()][1];
        for(int i=0;i<probs.rows();i++){
            float rndProb = rnd.nextFloat();
            int idx = BooleanIndexing.firstIndex(cumProbs.getRow(i), Conditions.greaterThan(rndProb)).getInt(0);
            sample[i][0] = idx;
        }
        return Nd4j.create(sample);
    }

    @Override
    public INDArray logProb(INDArray sample) {
        if(sample.rank()==1)
            sample = sample.reshape(-1, 1);
        if(logits==null){
            logits = Transforms.log(probs);
        }
//        float[][] output = new float[probs.rows()][1];
//        for(int i=0;i<probs.rows();i++){
//            int idx = sample.getInt(i, 0);
//            output[i][0] = logits.getFloat(i, idx);
//        }
//        return Nd4j.create(output);
        return logits.mul(sample);
    }

    @Override
    public INDArray entropy() {
        if(logits==null) {
            logits = Transforms.log(probs);
        }
        return logits.mul(probs).sum(-1).muli(-1);
    }

    private INDArray clampProbs(INDArray p){
        INDArray clampProbs = p.dup();
        AgentUtils.clamp(clampProbs, eps, 1-eps);
        return clampProbs;
    }
}
