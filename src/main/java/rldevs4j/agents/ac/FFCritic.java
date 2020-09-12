package rldevs4j.agents.ac;

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
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;

import java.io.File;
import java.io.IOException;


public class FFCritic implements ACCritic {
    private ComputationGraph model;
    private final double paramClamp = 0.5D;

    public FFCritic(ComputationGraph model){
        this.model = model;
        this.model.init();
    }

    public FFCritic(int obsDim, Double learningRate, Double l2, int hSize, Activation act, StatsStorage statsStorage){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.UNIFORM)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(paramClamp)
                .l2(l2)
                .graphBuilder()
                .addInputs("in")
                .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(act).build(), "in")
                .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(act).build(), "h1")
                .addLayer("value", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(hSize).nOut(1).activation(Activation.IDENTITY).build(), "h2")
                .setOutputs("value")
                .build();
        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"critic_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"critic_model");
        this.model = ComputationGraph.load(file, true);
    }

    @Override
    public INDArray output(INDArray obs) {
        return model.output(obs)[0];
    }

    @Override
    public Gradient gradient(INDArray states, INDArray returns) {
        model.fit(new INDArray[]{states}, new INDArray[]{returns.reshape(new int[]{returns.columns(), 1})});
        return model.gradient();
    }

    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize
     */
    @Override
    public void applyGradient(Gradient gradient, int batchSize, double score) {
        //Get a row vector gradient array, and apply it to the parameters to update the model
        model.params().subi(gradient.gradient());
    }

    @Override
    public INDArray getParams() {
        return model.params();
    }

    @Override
    public double getScore() {
        return model.score();
    }

    @Override
    public void setParams(INDArray p) {
        model.setParams(p.dup());
    }

    @Override
    public ACCritic clone() {
        return new FFCritic(model.clone());
    }
}
