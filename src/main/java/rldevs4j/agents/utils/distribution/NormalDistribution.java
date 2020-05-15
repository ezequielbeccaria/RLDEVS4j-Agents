package rldevs4j.agents.utils.distribution;

import java.util.Iterator;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author Ezequiel Beccaria
 */
public class NormalDistribution {
    private final INDArray mean;
    private final INDArray std;

    public NormalDistribution(double[] mean, double[] std) {
        this.mean = Nd4j.create(mean);
        this.std = Nd4j.create(std);
    }
    
    public NormalDistribution(INDArray mean, INDArray std) {
        this.mean = mean;
        this.std = std;
    }
    
    public INDArray sample(){        
        Random random = Nd4j.getRandom();
        Iterator<long[]> idxIter = new NdIndexIterator(mean.shape()); //For consistent values irrespective of c vs. fortran ordering        
        
        long len = mean.length();
        INDArray output = Nd4j.zeros(mean.shape());
        
        for (int i = 0; i < len; i++) {
            long[] idx = idxIter.next();

            output.putScalar(idx, std.getDouble(idx) * random.nextGaussian() + mean.getDouble(idx));
        }        
        return output;
    }
    
    public INDArray logProb(INDArray sample){
        //Taken from pytorch.Normal.entropy
        INDArray var = Transforms.pow(std, 2);
        INDArray logStd = Transforms.log(std);
        return (Transforms.pow(sample.sub(mean),2).div(var.mul(2)).sub(logStd).sub(Math.log(Math.sqrt(2*Math.PI)))).mul(-1D);
    }
    
    public INDArray entropy(){            
        //Taken from pytorch.Normal.entropy
        return Transforms.log(std).add(0.5 + 0.5 * Math.log(2 * Math.PI));
    }
}
