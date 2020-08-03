package rldevs4j.agents.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author Ezequiel Beccaria
 */
public class AgentUtils {
    
    public static INDArray clamp(INDArray input, double min, double max){
        BooleanIndexing.replaceWhere(input, min, Conditions.lessThan(min));
        BooleanIndexing.replaceWhere(input, max, Conditions.greaterThan(max));
        return input;
    }

    public static INDArray logSumExp(INDArray a) {
        INDArray aMax = a.max(-1);
        INDArray logSumExp = Transforms.log(Transforms.exp(a.sub(aMax)).sum(-1)).add(aMax);
        BooleanIndexing.replaceWhere(logSumExp, aMax, Conditions.isNan());
        return logSumExp;
    }

    public static float sum(float[] a){
        float sum = 0F;
        for(int i=0;i<a.length;i++)
            sum += a[i];
        return sum;
    }
}
