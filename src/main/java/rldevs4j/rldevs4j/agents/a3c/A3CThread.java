package rldevs4j.rldevs4j.agents.a3c;

import facade.DevsSuiteFacade;
import java.util.logging.Logger;
import rldevs4j.base.env.RLEnvironment;

/**
 * Thread class for run each worker training thread.
 * @author Ezequiel Beccaría
 */
public class A3CThread extends Thread {
    private final A3C a3c;
    private final DevsSuiteFacade facade;
    private final int workerEpisodes;
    private final double episodeTime;
    private final RLEnvironment container;
    private boolean running;
    private Logger logger;
    private final boolean DEBUG_MODE = false;

    public A3CThread(String name, A3C a3c, int workerEpisodes, double episodeTime, RLEnvironment container, Logger logger) {
        super(name);
        this.a3c = a3c;
        this.workerEpisodes = workerEpisodes;
        this.episodeTime = episodeTime;
        this.container = container;
        this.facade = new DevsSuiteFacade(this.container);
        this.logger = logger;
        this.running = false;
    }

    public boolean isRunning() {
        return running;
    }

    @Override
    public void run() {
        this.running = true;
        for (int i = 1; i <= workerEpisodes; i++) {
            if(i==workerEpisodes && this.getName().equals("worker_thread_"+i))
                container.getAgent().setDebugMode(true);
            //Inititalize environment and simulator
            facade.reset();
            //Episode time start
            long initTime = System.currentTimeMillis();
            //Simulate during "episodeTime" t (minuts of the day) 
            facade.simulateToTime(episodeTime);
            //Episode time stop
            long finishTime = System.currentTimeMillis();
            //Save results
            a3c.saveStatistics(this.getName(), i, this.container.getAgent().getTotalReward(), finishTime-initTime);
            //Calc gradients, enqueue gradients and reset worker agent
            this.container.getAgent().episodeFinished();   
        }
        this.running = false;        
        container.getAgent().clear();
    }

}