package rldevs4j.agents.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 *
 * @author Ezequiel Beccaria
 */
public class AgentUtils {
    
    public static void clamp(INDArray input, double min, double max){
        BooleanIndexing.replaceWhere(input, min, Conditions.lessThan(min));
        BooleanIndexing.replaceWhere(input, max, Conditions.greaterThan(max));
    }
    
}
