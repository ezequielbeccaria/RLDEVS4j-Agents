
package rldevs4j.rldevs4j.agents.a3c;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Collection;
import java.util.logging.Level;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.primitives.Triple;
import rldevs4j.base.agent.PersistModel;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.Container;
import rldevs4j.base.env.Environment;
import rldevs4j.base.env.factory.EnvironmentFactory;
import rldevs4j.experiment.ExperimentResult;
import rldevs4j.rldevs4j.agents.policy.factory.PolicyFactory;

/**
 * Asynchronous advantage actor-critic (A3C) global class (http://arxiv.org/abs/1602.01783).
 * This class' instance is in charge of create, starts and admin all the workers'
 * container enviroments.
 * @author Ezequiel Beccaría
 */
public class A3C implements PersistModel{
    private final EnvironmentFactory envFactory;
    private final INDArray initialState;
    private final int doNothingActionId;
    private ComputationGraph cg;
    private final boolean recurrent;
    private final PolicyFactory policyFactory;
    private final ConcurrentLinkedQueue<Triple<Gradient[],Integer, ComputationGraph>> queue; //gradients-batchsize
    private final List<A3CThread> workersThreads;
    private final double discountFactor;
    private final int stepCounterMax;
    private final Preprocessing preprocessing;
    
    private final int episodesPerWorker;
    private final double episodeMaxSimTime;
    
    private final Logger logger;
    private final boolean debug;
    
    private ExperimentResult results = null;
    

    public A3C(
            String name, 
            ComputationGraph cg,
            boolean recurrent,
            PolicyFactory policyFactory, 
            double discountFactor,
            int stepCounterMax,
            int doNothingActionId, 
            EnvironmentFactory envFactory, 
            INDArray initialState,    
            int episodesPerWorker, 
            double episodeMaxSimTime, 
            Preprocessing preprocessing,                
            Logger logger,            
            boolean debug) {     
        this.workersThreads = new ArrayList<>();
        this.envFactory = envFactory;
        this.initialState = initialState;
        this.preprocessing = preprocessing;
        this.discountFactor = discountFactor;
        this.stepCounterMax = stepCounterMax;
        this.doNothingActionId = doNothingActionId;
        this.cg = cg;
        this.recurrent = recurrent;
        this.policyFactory = policyFactory;
        this.queue = new ConcurrentLinkedQueue();
        this.episodesPerWorker = episodesPerWorker;
        this.episodeMaxSimTime = episodeMaxSimTime;
        this.logger = logger;
        this.debug = debug;
    }
    
    public synchronized void enqueueGradient(Gradient[] gradient, int steps, ComputationGraph c){
        queue.add(new Triple<>(gradient, steps, c));
    }
    
    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize 
     * @param c 
     */
    public void applyGradient(Gradient[] gradient, int batchSize, ComputationGraph c) {
        if (recurrent) {
            // assume batch sizes of 1 for recurrent networks,
            // since we are learning each episode as a time serie
            batchSize = 1;
        }
        ComputationGraphConfiguration cgConf = cg.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        cg.getUpdater().update(gradient[0], iterationCount, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        Collection<TrainingListener> iterationListeners = cg.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            iterationListeners.forEach((listener) -> {                
                listener.iterationDone(c, iterationCount, epochCount);
            });
        }
        cgConf.setIterationCount(iterationCount + 1);
    }
    
    /**
     * Save episode statistics
     * @param thread
     * @param episode
     * @param episodeReward 
     */
    public synchronized void saveStatistics(String thread, int episode, double episodeReward, long episodeTime){
        results.addResult(episodeReward, episodeTime);        
        if(episode%1==0 && this.debug)
            logger.log(Level.INFO, "{0} episode {1} terminated. Reward: {2}. Avg-Reward: {3}", new Object[]{thread, episode, episodeReward, results.getLastAverageReward()});   
    }
    
    /**
     * Iterate over ThreadWorkers checking if are still running.
     * @return 
     */
    private boolean isTrainingComplete(){
        boolean flag = true;
        for(A3CThread tw : workersThreads)
            if(tw.isRunning()){
                flag = false;
                break;
            }
        return flag;
    }
    
    /**
     * Start the threads of each workers.
     * @param workers
     * @throws InterruptedException 
     */
    public void startTraining(int workers) throws InterruptedException{
        this.results = new ExperimentResult();
        this.workersThreads.clear();
        // create workers
        for(int i=0;i<workers;i++){
            Environment env = envFactory.createInstance();
            ComputationGraph net = cg.clone();
            A3CWorker worker = new A3CWorker(
                    i, 
                    net, 
                    doNothingActionId, 
                    initialState,                     
                    policyFactory.create(net),                                    
                    this,
                    discountFactor,
                    stepCounterMax,
                    preprocessing.clone(),
                    false,
                    logger);
            Container container = new Container(worker, env); 
            A3CThread thread = new A3CThread("worker_thread_"+i, this, episodesPerWorker, episodeMaxSimTime, container, logger);
            this.workersThreads.add(thread);
        }
        // Start workers
        ExecutorService es = Executors.newCachedThreadPool();
        for(int i=0;i<workers;i++)
            es.execute(workersThreads.get(i));
        es.execute(new Runnable() {
                @Override
                public void run() {
                    // While loop for queued gradient updates
                    while (!isTrainingComplete()) {
                        if (!queue.isEmpty()) {
                            Triple<Gradient[], Integer, ComputationGraph> triple = queue.poll();
                            Gradient[] gradient = triple.getFirst();
                            applyGradient(gradient, triple.getSecond(), triple.getThird());                                            
                        }
                    }
                }
            });
        es.shutdown();
        es.awaitTermination(10, TimeUnit.DAYS);
    }
    
    public synchronized INDArray[] getNetsParams(){
        return new INDArray[]{this.cg.params()};
    }

    public ExperimentResult getResults() {
        return results;
    }

    @Override
    public boolean saveNetwork(String path) {
        try {
            ModelSerializer.writeModel(cg, path, true);
        } catch (IOException ex) {
            Logger.getLogger(A3C.class.getName()).log(Level.SEVERE, null, ex);
            return false;
        }
        return true;
    }

    @Override
    public void loadNetwork(String path) {
        try {
            //Load the model
            cg = ModelSerializer.restoreComputationGraph(path);            
        } catch (IOException ex) {
            Logger.getLogger(A3C.class.getName()).log(Level.SEVERE, null, ex);
        }
    }  
   
    public boolean isRecurrent() {
        return recurrent;
    }
}
