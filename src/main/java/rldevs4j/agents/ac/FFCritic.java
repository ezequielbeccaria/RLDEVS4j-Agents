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

import java.io.File;
import java.io.IOException;


public class FFCritic implements ACCritic {
    private ComputationGraph model;
    private final double paramClamp = 1D;

    public FFCritic(ComputationGraph model){
        this.model = model;
        WeightInit wi = WeightInit.XAVIER;
        this.model.setParams(wi.getWeightInitFunction().init(
                model.layerInputSize("h1"),
                model.layerSize("value"),
                model.params().shape(),
                'c',
                model.params()));
        this.model.init();
    }

    public FFCritic(int obsDim, Double learningRate, Double l2, int hSize, StatsStorage statsStorage){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipL2PerParamType)
                .gradientNormalizationThreshold(0.5)
                .graphBuilder()
                .addInputs("in")
                .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.TANH).build(), "in")
                .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "h1")
                .addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hSize).nOut(1).build(), "h2")
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
        model.setInputs(states);
        model.setLabels(returns.reshape(new int[]{returns.columns(), 1}));
        model.computeGradientAndScore();
        return model.gradient();
    }

    private INDArray gradientsClipping(INDArray output){
        BooleanIndexing.replaceWhere(output, paramClamp, Conditions.greaterThan(paramClamp));
        BooleanIndexing.replaceWhere(output, -paramClamp, Conditions.lessThan(-paramClamp));
        return output;
    }

    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize
     */
    @Override
    public void applyGradient(Gradient gradient, int batchSize, double score) {
        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        model.setScore(score);
        model.getUpdater().update(gradient, iterationCount, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        //Get a row vector gradient array, and apply it to the parameters to update the model
        model.params().subi(gradientsClipping(gradient.gradient()));
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
        model.setParams(p);
    }

    @Override
    public ACCritic clone() {
        return new FFCritic(model.clone());
    }
}
