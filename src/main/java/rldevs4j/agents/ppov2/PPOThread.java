package rldevs4j.agents.ppov2;

import facade.DevsSuiteFacade;
import rldevs4j.base.env.RLEnvironment;

/**
 * Thread class for run each worker training thread.
 * @author Ezequiel Beccaría
 */
public class PPOThread extends Thread {
    private final PPO PPO;
    private DevsSuiteFacade facade;
    private final int workerEpisodes;
    private final double episodeTime;
    private final RLEnvironment container;
    private boolean running;
    private final boolean DEBUG_MODE = false;

    public PPOThread(String name, PPO ppo, int workerEpisodes, double episodeTime, RLEnvironment container) {
        super(name);
        this.PPO = ppo;
        this.workerEpisodes = workerEpisodes;
        this.episodeTime = episodeTime;
        this.container = container;
        this.running = false;
    }

    public boolean isRunning() {
        return running;
    }

    @Override
    public void run() {
        this.running = true;
        for (int i = 1; i <= workerEpisodes; i++) {
            //Inititalize environment and simulator
            facade = new DevsSuiteFacade(this.container);
            facade.reset();
            //Episode time start
            long initTime = System.currentTimeMillis();
            //Simulate during "episodeTime" t (minuts of the day) 
            facade.simulateToTime(episodeTime);
            //Episode time stop
            long finishTime = System.currentTimeMillis();
            //Save results
            PPO.saveStatistics(this.getName(), i, this.container.getAgent().getTotalReward(), finishTime-initTime);
            //Calc gradients, enqueue gradients and reset worker agent
            this.container.getAgent().episodeFinished();   
        }
        this.running = false;        
        container.getAgent().clear();
    }

}