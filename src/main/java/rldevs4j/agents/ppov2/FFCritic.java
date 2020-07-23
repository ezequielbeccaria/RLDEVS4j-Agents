package rldevs4j.agents.ppov2;

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
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.AgentUtils;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

public class FFCritic implements PPOCritic {
    private ComputationGraph model;
    private final double paramClamp = 0.5D;
    private float epsilonClip;

    public FFCritic(ComputationGraph model, float epsilonClip){
        this.model = model;
        WeightInit wi = WeightInit.XAVIER;
        this.model.setParams(wi.getWeightInitFunction().init(
                model.layerInputSize("h1"),
                model.layerSize("value"),
                model.params().shape(),
                'c',
                model.params()));
        this.model.init();
        this.epsilonClip = epsilonClip;
    }

    public FFCritic(int obsDim, Double learningRate, Double l2, float epsilonClip, int hSize, StatsStorage statsStorage){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(paramClamp)
                .l2(l2)
                .graphBuilder()
                .addInputs("in")
                .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.TANH).build(), "in")
                .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "h1")
                .addLayer("value", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(hSize).nOut(1).activation(Activation.IDENTITY).build(), "h2")
                .setOutputs("value")
                .build();
        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }
        this.epsilonClip = epsilonClip;
    }

    @Override
    public void saveModel(String path) throws IOException {
        File file = new File(path+"FFCritic_model");
        this.model.save(file);
    }

    @Override
    public void loadModel(String path) throws IOException {
        File file = new File(path+"FFCritic_model");
        this.model = ComputationGraph.load(file, true);
    }

    @Override
    public INDArray output(INDArray obs) {
        return model.output(obs)[0];
    }

//    public INDArray loss(INDArray states, INDArray oldValues, INDArray returns){
//        model.fit(new INDArray[]{states}, new INDArray[]{returns});
////        INDArray v = model.output(states)[0];
////        INDArray vClipped = oldValues.add(AgentUtils.clamp(v.dup().subi(oldValues), epsilonClip, epsilonClip));
////        INDArray lossV = Transforms.pow(v.subColumnVector(returns), 2);
////        INDArray lossVClipped = Transforms.pow(vClipped.subColumnVector(returns), 2);
//        //Why max? -> https://github.com/openai/baselines/issues/445#issuecomment-408835567
////        INDArray loss = Transforms.max(lossV, lossVClipped).muli(0.5);
//        return lossV;
//    }

    @Override
    public Gradient gradient(INDArray states, INDArray oldValues, INDArray returns) {
//        INDArray lossPerPoint = loss(states, oldValues, returns);
//        model.feedForward(new INDArray[]{states}, true, false);
//        Gradient g = model.backpropGradient(lossPerPoint);
//        model.setScore(lossPerPoint.meanNumber().doubleValue());

         model.fit(new INDArray[]{states}, new INDArray[]{returns.reshape(new int[]{returns.columns(), 1})});

//        ComputationGraphConfiguration cgConf = model.getConfiguration();
//        int iterationCount = cgConf.getIterationCount();
//        int epochCount = cgConf.getEpochCount();
//        this.model.getUpdater().update(g, iterationCount, epochCount, states.rows(), LayerWorkspaceMgr.noWorkspaces());
//        this.model.update(g);

        return model.gradient();
    }

    private INDArray gradientsClipping(INDArray output){
        INDArray clipped = output.dup();
        BooleanIndexing.replaceWhere(clipped, paramClamp, Conditions.greaterThan(paramClamp));
        BooleanIndexing.replaceWhere(clipped, -paramClamp, Conditions.lessThan(-paramClamp));
        return clipped;
    }

    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize
     */
    @Override
    public void applyGradient(INDArray gradient, int batchSize) {
        //Get a row vector gradient array, and apply it to the parameters to update the model
//        model.params().subi(gradientsClipping(gradient));
        model.params().subi(gradient);
    }

    @Override
    public INDArray getParams() {
        return model.params();
    }

    @Override
    public void setParams(INDArray p) {
        model.setParams(p.dup());
    }

    @Override
    public ComputationGraph getModel() {
        return model;
    }

    @Override
    public PPOCritic clone() {
        return new FFCritic(model.clone(), epsilonClip);
    }
}
