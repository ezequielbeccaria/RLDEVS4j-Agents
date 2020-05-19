package rldevs4j.agents.utils.memory;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.rng.Random;

/**
 * Uniform sample experience replay memory with maxSize capacity.
 * When maxSize is reached, the older element added is replaced
 * @author Ezequiel Beccaria
 * @param <E>
 */
public class ExperienceReplayBuffer<E> {
    protected final int maxSize;
    protected int next_idx;
    protected final List<E> memory;
    protected final Random rnd;

    public ExperienceReplayBuffer(int maxSize, Random rnd) {
        this.maxSize = maxSize;
        this.next_idx = 0;
        this.memory = new ArrayList<>();  
        this.rnd = rnd;
    }
    
    /**
     * Add an element to the memory returning the index of the insertion.
     * @param e
     * @return idx
     */
    public synchronized int add(E e){
        if(next_idx>=memory.size()){
            memory.add(e);
        }else{
            memory.remove(next_idx);
            memory.add(next_idx, e);
        }    
        int insertIdx = next_idx;
        next_idx = (next_idx+1)%maxSize;
        return insertIdx;
    }
    
    public synchronized List<E> sample(int batchSize){
        //Select samples from memory with uniform distribution
        List<E> samples = new ArrayList<>();
        for(int i=0;i<Math.min(memory.size(), batchSize);i++){
            samples.add(memory.get(rnd.nextInt(memory.size())));
        }
        return samples;
    }
    
    /**
     * Get tuple in memory index idx.
     * @param idx
     * @return 
     */
    public E get(int idx){
        return memory.get(idx);
    }
    
    public int size(){
        return memory.size();
    }
    
    public void clear(){
        memory.clear();
    }
}
