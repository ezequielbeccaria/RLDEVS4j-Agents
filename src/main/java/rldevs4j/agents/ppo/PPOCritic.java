package rldevs4j.agents.ppo;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Ezequiel Beccaria
 */
public class PPOCritic {
    private final ComputationGraph model;
    private final double paramClamp = 1D;
    
    public PPOCritic(int obsDim, Double learningRate, Double l2, int hSize, StatsStorage statsStorage){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()    
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(paramClamp)
            .l2(l2)
            .graphBuilder()
            .addInputs("in")
            .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.RELU).build(), "in")                     
            .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.RELU).build(), "h1")
            .addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hSize).nOut(1).build(), "h2")           
            .setOutputs("value")
            .build();

        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new ScoreIterationListener(1));
            this.model.setListeners(new StatsListener(statsStorage));
        }
    }
    
    public PPOCritic(Map<String,Object> params){
        this((int) params.get("OBS_DIM"),
            (double) params.getOrDefault("LEARNING_RATE", 1e-3),
            (double) params.getOrDefault("L2", 1e-2),
            (int) params.getOrDefault("HIDDEN_SIZE", 128),
            (StatsStorage) params.getOrDefault("STATS_STORAGE", null));
    }
    
    public INDArray output(INDArray obs){
        return model.output(obs)[0];
    }
    
    public double train(INDArray states, INDArray returns){
        int batchSize = states.rows();
        Gradient g = gradient(states, returns);
        
        applyGradient(g, batchSize);
        return model.score();
    }
    
    private Gradient gradient(INDArray states, INDArray returns) {
        model.fit(new INDArray[]{states}, new INDArray[]{returns.reshape(new int[]{returns.columns(), 1})});
        return model.gradient();
    }

    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize 
     */
    private void applyGradient(Gradient gradient, int batchSize) {
        model.params().subi(gradient.gradient().dup());
    }
    
}
