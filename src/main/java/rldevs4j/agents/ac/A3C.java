
package rldevs4j.agents.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Triple;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.Environment;
import rldevs4j.base.env.RLEnvironment;
import rldevs4j.base.env.factory.EnvironmentFactory;
import rldevs4j.experiment.ExperimentResult;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Asynchronous advantage actor-critic (A3C) global class (http://arxiv.org/abs/1602.01783).
 * This class' instance is in charge of create, starts and admin all the workers'
 * container enviroments.
 * @author Ezequiel Beccaría
 */
public class A3C {
    private final EnvironmentFactory envFactory;
    private DiscreteACActor actor;
    private ACCritic critic;
    private float[][] actionSpace;
    private final ConcurrentLinkedQueue<Triple<Gradient[],Integer, ComputationGraph>> queue; //gradients-batchsize
    private final List<A3CThread> workersThreads;
    private final double discountFactor;
    private final int horizon;
    private final Preprocessing preprocessing;
    
    private final int episodesPerWorker;
    private final double episodeMaxSimTime;

    private final Logger logger;
    private final boolean debug;
    
    private ExperimentResult results = null;
    

    public A3C(
            DiscreteACActor actor,
            ACCritic critic,
            Preprocessing preprocessing,
            EnvironmentFactory envFactory,
            Map<String,Object> params) {
        this.actor = actor;
        this.critic = critic;
        this.workersThreads = new ArrayList<>();
        this.envFactory = envFactory;
        this.preprocessing = preprocessing;
        this.discountFactor = (double) params.getOrDefault("DISCOUNT_RATE", 0.99D);
        this.horizon = (int) params.getOrDefault("HORIZON", 100);
        this.queue = new ConcurrentLinkedQueue();
        this.episodesPerWorker = (int) params.getOrDefault("EPISODES_WORKER", 10);
        this.episodeMaxSimTime = (double) params.getOrDefault("SIMULATION_TIME", 3000);
        this.actionSpace = (float[][]) params.get("ACTION_SPACE");
        this.debug = (boolean) params.getOrDefault("DEBUG", false);
        this.logger = Logger.getGlobal();
    }
    
//    public synchronized void enqueueGradient(Gradient[] gradient, int steps, ComputationGraph c){
    public synchronized void enqueueGradient(Gradient[] gradient, int steps){
        queue.add(new Triple<>(gradient, steps, null));
    }

    /**
     * Apply to global parameters gradients generated and queue by the workers
     * @param gradient
     * @param batchSize
     */
    private void applyGradient(Gradient[] gradient, int batchSize) {
        //Critic
        critic.applyGradient(gradient[0], batchSize);
        //Actor
        actor.applyGradient(gradient[1], batchSize);
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
            A3CWorker worker = new A3CWorker(
                    i,
                    (DiscreteACActor) actor.clone(),
                    critic.clone(),
                    this,
                    discountFactor,
                    horizon,
                    preprocessing.clone(),
                    actionSpace);
            RLEnvironment container = new RLEnvironment(worker, env);
            A3CThread thread = new A3CThread("worker_thread_"+i, this, episodesPerWorker, episodeMaxSimTime, container);
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
                            applyGradient(gradient, triple.getSecond());
                        }
                    }
                }
            });
        es.shutdown();
        es.awaitTermination(10, TimeUnit.DAYS);
    }
    
    public synchronized INDArray[] getNetsParams(){
        return new INDArray[]{this.actor.getParams(), this.critic.getParams()};
    }

    public ExperimentResult getResults() {
        return results;
    }

    public void saveModel(String path) throws IOException {
        this.actor.saveModel(path);
        this.critic.saveModel(path);
    }

    public void loadModel(String path) throws IOException {
        this.actor.loadModel(path);
        this.critic.loadModel(path);
    }  
}
