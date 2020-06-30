package rldevs4j.agents.dqn;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import rldevs4j.agents.utils.distribution.Categorical;
import rldevs4j.agents.utils.memory.TDTuple;
import rldevs4j.agents.utils.memory.TDTupleBatch;
import rldevs4j.agents.utils.scaler.StandartScaler;

import java.io.File;
import java.io.IOException;
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
    private final double discountFactor;
    private final int c;
    private int j;

    public Model(
            int obsDim,
            int outputDim,
            Double learningRate,
            double discountFactor,
            double clipReward,
            int targetUpdate,
            int hSize,
            boolean rwdMeanScale,
            boolean rwdStdScale,
            StatsStorage statsStorage){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(learningRate))
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(clipReward)
            .graphBuilder()
            .addInputs("in")
            .addLayer("h1", new DenseLayer.Builder().nIn(obsDim).nOut(hSize).activation(Activation.TANH).build(), "in")
            .addLayer("h2", new DenseLayer.Builder().nIn(hSize).nOut(hSize).activation(Activation.TANH).build(), "h1")
            .addLayer("value", new DenseLayer.Builder().nIn(hSize).nOut(outputDim).activation(Activation.IDENTITY).build(), "h2")
            .setOutputs("value")
            .build();

        model = new ComputationGraph(conf);
        model.init();
        if(statsStorage!=null) {
            this.model.setListeners(new StatsListener(statsStorage));
        }
//        this.model.setListeners(new ScoreIterationListener(10));
//        this.model.setListeners(new PerformanceListener(1));

        this.target = model.clone();
        this.scaler = StandartScaler.getInstance(rwdMeanScale, rwdStdScale);
        this.discountFactor = discountFactor;
        this.c = targetUpdate;
        this.j = 0;
    }

    public Model(Map<String,Object> params){
        this((int) params.get("OBS_DIM"),
            (int) params.get("OUTPUT_DIM"),
            (double) params.get("LEARNING_RATE"),
            (double) params.get("DISCOUNT_RATE"),
            (double) params.get("CLIP_REWARD"),
            (int) params.get("TARGET_UPDATE"),
            (int) params.get("HIDDEN_SIZE"),
            (boolean) params.get("RWD_MEAN_SCALE"),
            (boolean) params.get("RWD_STD_SCALE"),
            (StatsStorage) params.getOrDefault("STATS_STORAGE", null));
    }

    public INDArray test(INDArray obs){
        return model.output(obs.reshape(new int[]{1, obs.columns()}))[0];
    }

    public int action(INDArray obs){
        INDArray qsa = model.output(obs.reshape(new int[]{1, obs.columns()}))[0];
        Categorical dist = new Categorical(null, qsa);
        return dist.sample().getInt(0);
    }

    public void train(List<TDTuple> replayTuples, int batchSize, int iteration){
        if(replayTuples.size()==batchSize) {

            TDTupleBatch batch = new TDTupleBatch(replayTuples);

            // Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            // columns of actions taken. These are the actions which would've been taken
            // for each batch state according to policy_net
            INDArray state_action_values = model.output(batch.getStates())[0];

            // Compute V(s_{t+1}) for all next states.
            // Expected values of actions for non_final_next_states are computed based
            // on the "older" target_net; selecting their best reward with max(1)[0].
            // This is merged based on the mask, such that we'll have either the expected
            // state value or 0 in case the state was final.
            INDArray qsa = model.output(batch.getNextStates())[0];
            int[] max_act = qsa.argMax(1).toIntVector();
            INDArray qsa_prime = target.output(batch.getNextStates())[0];
            double[] delta = new double[batchSize];
            double[] scaledRwd = scaler.partialFitTransform(batch.getRewards());
            for(int i=0;i<batchSize;i++){
                delta[i] = scaledRwd[i] + batch.getDone()[i] * discountFactor * qsa_prime.getFloat(i, max_act[i]);
                state_action_values.putScalar(new int[]{i, max_act[i]}, delta[i]);
            }

            Gradient g = this.gradient(batch.getStates(), state_action_values);
            this.applyGradient(g, batchSize, iteration);

            this.j++;
            if (j % c == 0) {
                target.setParams(model.params());
            }
        }
    }

    public INDArray loss(INDArray states, INDArray target){
        INDArray v = model.output(states)[0];
        INDArray loss = Transforms.pow(v.sub(target), 2);
        return loss;
    }

    public Gradient gradient(INDArray input, INDArray labels) {
        INDArray lossPerPoint = loss(input, labels);
        model.feedForward(new INDArray[]{input}, true, false);
        Gradient g = model.backpropGradient(lossPerPoint);
        model.setScore(lossPerPoint.meanNumber().doubleValue());

        return g;
    }

    /**
     * Apply calculated gradients to model parameters
     * @param gradient
     * @param batchSize
     */
    private void applyGradient(Gradient gradient, int batchSize, int iteration) {
        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        model.getUpdater().update(gradient, iteration, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        //Get a row vector gradient array, and apply it to the parameters to update the model
        model.update(gradient);

        //Notify training listeners
        Collection<TrainingListener> iterationListeners = model.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            iterationListeners.forEach((listener) -> {
                listener.iterationDone(model, iteration, j);
            });
        }
        cgConf.setIterationCount(iterationCount + 1);
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
