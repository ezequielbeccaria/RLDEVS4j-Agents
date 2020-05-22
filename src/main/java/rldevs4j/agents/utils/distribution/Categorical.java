package rldevs4j.agents.utils.distribution;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;

/**
 * Creates a categorical distribution parameterized by either @param `probs` or @param `logits` (but not both).
 */
public class Categorical implements Distribution{
    private INDArray probs;
    private INDArray logits;

    public Categorical(INDArray probs) {
        this(probs, null);
    }

    public Categorical(INDArray probs, INDArray logits) {
        if((probs == null) && (logits == null))
            throw new RuntimeException("Either `probs` or `logits` must be specified, but not both.");
        this.probs = probs;
        this.logits = logits;
        if(probs != null){
            if(probs.shape().length < 1)
                throw new RuntimeException("`probs` parameter must be at least one-dimensional.");
            this.probs = probs.div(probs.sum(-1));
        }else{
            if(logits.shape().length < 1)
                throw new RuntimeException("`logits` parameter must be at least one-dimensional.");
            this.logits = logits.sub(AgentUtils.logSumExp(logits));
        }
    }

    @Override
    public INDArray sample() {
        if(logits != null){
            probs = Transforms.softmax(logits);
        }
        INDArray cumProbs = probs.cumsum(-1);
        double rndProb = Nd4j.getRandom().nextDouble();
        return BooleanIndexing.firstIndex(cumProbs, Conditions.greaterThan(rndProb));
    }

    @Override
    public INDArray logProb(INDArray sample) {
        if(logits==null)
            logits = Transforms.log(probs);
        return logits.getScalar(sample.getInt(0));
    }

    @Override
    public INDArray entropy() {
        if(logits==null) {
            float eps = 1.1920928955078125e-07f;
            INDArray clampProbs = probs.dup();
            AgentUtils.clamp(clampProbs, eps, 1-eps);
            logits = Transforms.log(clampProbs);
        }
        return logits.mul(probs).sum(-1).muli(-1);
    }
}
