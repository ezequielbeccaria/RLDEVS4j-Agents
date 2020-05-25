package rldevs4j.agents.dqn;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.distribution.Categorical;
import rldevs4j.agents.utils.memory.ExperienceReplayBuffer;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.agents.utils.memory.TDTupleBatch;
import rldevs4j.agents.utils.scaler.StandartScaler;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Ezequiel Beccaria
 */
public class Model {
    private StandartScaler scaler;
    private ComputationGraph model;
    private ComputationGraph target;
    private final double paramClamp = 1D;
    private final double discountFactor;
    private final boolean clipReward;
    private final double tau;
    private final Random rnd;
    private final int c;
    private int j;

    public Model(
            int obsDim,
            int outputDim,
            Double learningRate,
            Double l2,
            Double discountFactor,
            boolean clipReward,
            int targetUpdate,
            int hSize,
            StatsStorage statsStorage){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new RmsProp(learningRate))
            .weightInit(WeightInit.XAVIER)
            .l2(l2!=null?l2:0.001D)
            .graphBuilder()
            .addInputs("in")
            .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.RELU).build(), "in")
            .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.RELU).build(), "h1")
            .addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hSize).nOut(outputDim).build(), "h2")
            .setOutputs("value")
            .build();

        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }
//        this.model.setListeners(new ScoreIterationListener(1));
//        this.model.setListeners(new PerformanceListener(1));

        this.target = model.clone();
        this.scaler = new StandartScaler();
        this.discountFactor = discountFactor;
        this.clipReward = clipReward;
        this.tau = 1D;
        this.rnd = Nd4j.getRandom();
        this.c = targetUpdate;
        this.j = 0;
    }

    public Model(Map<String,Object> params){
        this((int) params.get("OBS_DIM"),
            (int) params.get("OUTPUT_DIM"),
            (double) params.getOrDefault("LEARNING_RATE", 1e-3),
            (double) params.getOrDefault("L2", 1e-2),
            (double) params.getOrDefault("DISCOUNT_FACTOR", 1e-3),
            (boolean) params.getOrDefault("CLIP_REWARD", false),
            (int) params.getOrDefault("TARGET_UPDATE", 50),
            (int) params.getOrDefault("HIDDEN_SIZE", 128),
            (StatsStorage) params.getOrDefault("STATS_STORAGE", null));
    }

    public int action(INDArray obs){
        INDArray qsa = model.output(obs.reshape(new int[]{1, obs.columns()}))[0];
        Categorical dist = new Categorical(null, qsa);
        return dist.sample().getInt(0);
    }

    private INDArray gradientsClipping(INDArray output){
        BooleanIndexing.replaceWhere(output, paramClamp, Conditions.greaterThan(paramClamp));
        BooleanIndexing.replaceWhere(output, -paramClamp, Conditions.lessThan(-paramClamp));
        return output;
    }

    public void train(List<TDTuple> replayTuples, int batchSize, int iteration){
        if(replayTuples.size()==batchSize) {
            List<INDArray> batchXList = new ArrayList<>();
            List<INDArray> batchYList = new ArrayList<>();

            for(TDTuple t : replayTuples){
                INDArray qsa = model.output(t.getState().reshape(new int[]{1, t.getState().columns()}))[0];
                int qsa_max_action_arg = Nd4j.getExecutioner().execAndReturn(new IMax(qsa)).getFinalResult().intValue();
                INDArray qsa_prime = target.output(t.getNextState().reshape(new int[]{1, t.getNextState().columns()}))[0]; //qsa' from dual model
                qsa.putScalar(qsa_max_action_arg, scaler.partialFitTransform(t.getReward(clipReward))+discountFactor*qsa_prime.getDouble(qsa_max_action_arg));
                batchXList.add(t.getState());
                batchYList.add(qsa);
            }

            Gradient g = this.gradient(Nd4j.vstack(batchXList), Nd4j.vstack(batchYList));
            this.applyGradient(g, batchSize, iteration);

            this.j++;
            if (j % c == 0) {
                target.setParams(model.params());
            }
        }
    }

    public Gradient gradient(INDArray input, INDArray labels) {
        model.setInputs(input);
        model.setLabels(labels);
        model.computeGradientAndScore();
        Collection<TrainingListener> valueIterationListeners = model.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener l : valueIterationListeners) {
                l.onGradientCalculation(model);
            }
        }
        return model.gradient();
    }

    /**
     * Apply calculated gradients to model parameters
     * @param gradient
     * @param batchSize
     */
    private void applyGradient(Gradient gradient, int batchSize, int iteration) {
        ComputationGraphConfiguration cgConf = model.getConfiguration();
        model.getUpdater().update(gradient, iteration, j, batchSize, LayerWorkspaceMgr.noWorkspaces());
        //Get a row vector gradient array, and apply it to the parameters to update the model
        INDArray updateVector = gradientsClipping(gradient.gradient());
        model.params().subi(updateVector);
        //Notify training listeners
        Collection<TrainingListener> iterationListeners = model.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            iterationListeners.forEach((listener) -> {
                listener.iterationDone(model, iteration, j);
            });
        }
    }

    public void saveModel(String path) throws IOException {
        File file = new File(path+"_model");
        this.model.save(file, true);
    }
    public void loadModel(String path) throws IOException {
        File file = new File(path+"_model");
        this.model = ComputationGraph.load(file, true);
    }
}
