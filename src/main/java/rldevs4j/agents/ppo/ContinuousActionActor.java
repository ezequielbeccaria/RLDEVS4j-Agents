package rldevs4j.agents.ppo;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.distribution.NormalDistribution;
import rldevs4j.agents.utils.AgentUtils;

/**
 *
 * @author Ezequiel Beccaria
 */
public class ContinuousActionActor {
    private final ComputationGraph model;
    private final double LOG_STD_MIN = -20D; // std = e^-20 = 0.000000002
    private final double LOG_STD_MAX = 1; // std = e^1 = 2.7183
    private final double epsilonClip;
    private final double tahnActionLimit; //max sample value
    private final double entropyCoef;
    
    public ContinuousActionActor(
            int obsDim, 
            int actionDim, 
            Double learningRate, 
            Double l2, 
            int hSize,
            double tahnActionLimit,
            double epsilonClip,
            double entropyCoef){
        
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()    
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .weightInit(WeightInit.XAVIER)                
            .l2(l2!=null?l2:0.001D)
            .graphBuilder()
            .addInputs("in")
            .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.RELU).build(), "in")                     
            .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.RELU).build(), "h1")            
            .addLayer("mean", new DenseLayer.Builder().nIn(hSize).nOut(actionDim).activation(Activation.RELU).build(), "h2")            
            .addLayer("log_std", new DenseLayer.Builder().nIn(hSize).nOut(actionDim).activation(Activation.RELU).build(), "h2")
            .setOutputs("mean", "log_std")
            .build();

        this.model = new ComputationGraph(conf);
        this.model.init();
        
        this.tahnActionLimit = tahnActionLimit;
        this.epsilonClip = epsilonClip;
        this.entropyCoef = entropyCoef;
    }
    
    public void saveModel(String path) throws IOException{
        File file = new File(path+"actor_model");
        this.model.save(file);
    }
    
    public ContinuousActionActor(Map<String,Object> params){
        this((int) params.get("OBS_DIM"),
            (int) params.get("ACTION_DIM"),
            (double) params.getOrDefault("LEARNING_RATE", 1e-3),
            (double) params.getOrDefault("L2", 1e-2),
            (int) params.getOrDefault("HIDDEN_SIZE", 128),
            (double) params.get("TAHN_ACTION_LIMIT"),
            (double) params.getOrDefault("EPSILON_CLIP", 0.2D),
            (double) params.getOrDefault("ENTROPY_COEF", 0.02D));
    }
    
    public INDArray[] output(INDArray obs, INDArray act){                
        NormalDistribution pi = distribution(obs);
        INDArray sample = pi.sample();
        INDArray logPi = pi.logProb(sample);        
        INDArray entropy = pi.entropy();
        return new INDArray[]{sample, logPi, entropy};
    }
    
    public double[] action(INDArray obs){        
        NormalDistribution pi = distribution(obs.reshape(new int[]{1, obs.columns()}));
        INDArray sample = Transforms.tanh(pi.sample());
        sample = Transforms.max(sample, 0);
        sample = sample.muli(this.tahnActionLimit);
        sample = Transforms.round(sample);
        return sample.toDoubleVector();
    }
    
    private NormalDistribution distribution(INDArray obs){
        INDArray[] output = model.output(obs);
        INDArray mean = output[0];
        //Clamp LogStd
        AgentUtils.clamp(output[1], LOG_STD_MIN, LOG_STD_MAX);        
        INDArray std = Transforms.exp(output[1]);
        
        return new NormalDistribution(mean, std);
    }
    
    public double train(INDArray states, INDArray actions, INDArray advantages, INDArray logOldPi, int iteration, int epoch){
        //Calculate gradient with respect to an external error
        INDArray lossPerPoint = loss(states, actions, advantages, logOldPi);
        //Do forward pass, but don't clear the input activations in each layers - we need those set so we can calculate
        // gradients based on them
        this.model.feedForward(new INDArray[]{states}, true, false);
        //Update the gradient: apply learning rate, momentum, etc
        //This modifies the Gradient object in-place
        Gradient g = model.backpropGradient(lossPerPoint, lossPerPoint);
        model.getUpdater().update(g, iteration, epoch, states.rows(), LayerWorkspaceMgr.noWorkspaces());
        //Get a row vector gradient array, and apply it to the parameters to update the model
        INDArray updateVector = g.gradient();
        model.params().subi(updateVector);
        return lossPerPoint.meanNumber().doubleValue();
    }
    
    private INDArray loss(INDArray states , INDArray actions, INDArray advantages, INDArray logOldPi){   
        //output[0] -> sample, output[1] -> logProb, output[2] -> entropy
        INDArray[] output = this.output(states, actions);
        INDArray ratio = Transforms.exp(output[1].sub(logOldPi));
        INDArray clipAdv = ratio.dup();
        AgentUtils.clamp(clipAdv, 1D-epsilonClip, 1D+epsilonClip);  
        clipAdv.muliColumnVector(advantages);
        INDArray lossPerPoint = Transforms.min(ratio.mulColumnVector(advantages), clipAdv).mul(-1D).add(output[2].mul(this.entropyCoef));        
        //Extra info
        return lossPerPoint;
    }
}
