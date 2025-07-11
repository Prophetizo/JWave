package jwave.transforms;

import jwave.transforms.wavelets.daubechies.Daubechies4;
import jwave.transforms.wavelets.PooledWavelet;
import org.junit.Test;
import org.junit.Before;
import jwave.utils.ArrayBufferPool;
import jwave.exceptions.JWaveException;

import java.util.Random;

/**
 * Comprehensive performance comparison of pooled vs standard implementations.
 * 
 * @author Stephen Romano
 */
public class PoolingPerformanceComparisonTest {
    
    private static final int ITERATIONS = 1000;
    private static final int SIGNAL_SIZE = 1024;
    private double[] testSignal;
    
    @Before
    public void setUp() {
        Random rand = new Random(42);
        testSignal = new double[SIGNAL_SIZE];
        for (int i = 0; i < SIGNAL_SIZE; i++) {
            testSignal[i] = rand.nextGaussian();
        }
    }
    
    @Test
    public void testWaveletPacketTransformPooling() throws Exception {
        System.out.println("\n=== WaveletPacketTransform Pooling Performance ===");
        
        Daubechies4 wavelet = new Daubechies4();
        WaveletPacketTransform standard = new WaveletPacketTransform(wavelet);
        PooledWaveletPacketTransform pooled = new PooledWaveletPacketTransform(wavelet);
        
        // Warmup
        for (int i = 0; i < 100; i++) {
            standard.forward(testSignal, 3);
            pooled.forward(testSignal, 3);
        }
        
        // Test standard
        long standardTime = 0;
        long standardGC = getGCCount();
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            standard.forward(testSignal, 5);
            standardTime += System.nanoTime() - start;
        }
        standardGC = getGCCount() - standardGC;
        
        // Test pooled
        long pooledTime = 0;
        long pooledGC = getGCCount();
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            pooled.forward(testSignal, 5);
            pooledTime += System.nanoTime() - start;
        }
        pooledGC = getGCCount() - pooledGC;
        
        System.out.printf("Standard: %.3f ms avg, %d GCs\n", 
            standardTime / 1_000_000.0 / ITERATIONS, standardGC);
        System.out.printf("Pooled:   %.3f ms avg, %d GCs\n", 
            pooledTime / 1_000_000.0 / ITERATIONS, pooledGC);
        System.out.printf("Speedup:  %.2fx\n", (double)standardTime / pooledTime);
        
        ArrayBufferPool.remove();
    }
    
    @Test
    public void testFFTPooling() throws Exception {
        System.out.println("\n=== FastFourierTransform Pooling Performance ===");
        
        FastFourierTransform standard = new FastFourierTransform();
        PooledFastFourierTransform pooled = new PooledFastFourierTransform();
        
        // Warmup
        for (int i = 0; i < 100; i++) {
            standard.forward(testSignal);
            pooled.forward(testSignal);
        }
        
        // Test standard
        long standardTime = 0;
        long standardGC = getGCCount();
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            double[] fft = standard.forward(testSignal);
            standard.reverse(fft);
            standardTime += System.nanoTime() - start;
        }
        standardGC = getGCCount() - standardGC;
        
        // Test pooled
        long pooledTime = 0;
        long pooledGC = getGCCount();
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            double[] fft = pooled.forward(testSignal);
            pooled.reverse(fft);
            pooledTime += System.nanoTime() - start;
        }
        pooledGC = getGCCount() - pooledGC;
        
        System.out.printf("Standard: %.3f ms avg, %d GCs\n", 
            standardTime / 1_000_000.0 / ITERATIONS, standardGC);
        System.out.printf("Pooled:   %.3f ms avg, %d GCs\n", 
            pooledTime / 1_000_000.0 / ITERATIONS, pooledGC);
        System.out.printf("Speedup:  %.2fx\n", (double)standardTime / pooledTime);
        
        ArrayBufferPool.remove();
    }
    
    @Test
    public void testWaveletPooling() throws Exception {
        System.out.println("\n=== Wavelet Base Class Pooling Performance ===");
        
        Daubechies4 standard = new Daubechies4();
        PooledWavelet pooled = new PooledWavelet(new Daubechies4());
        
        // Warmup
        for (int i = 0; i < 100; i++) {
            standard.forward(testSignal, testSignal.length);
            pooled.forward(testSignal, testSignal.length);
        }
        
        // Test standard
        long standardTime = 0;
        long standardGC = getGCCount();
        for (int i = 0; i < ITERATIONS * 10; i++) { // More iterations since it's faster
            long start = System.nanoTime();
            standard.forward(testSignal, testSignal.length);
            standardTime += System.nanoTime() - start;
        }
        standardGC = getGCCount() - standardGC;
        
        // Test pooled
        long pooledTime = 0;
        long pooledGC = getGCCount();
        for (int i = 0; i < ITERATIONS * 10; i++) {
            long start = System.nanoTime();
            pooled.forward(testSignal, testSignal.length);
            pooledTime += System.nanoTime() - start;
        }
        pooledGC = getGCCount() - pooledGC;
        
        System.out.printf("Standard: %.3f µs avg, %d GCs\n", 
            standardTime / 1_000.0 / (ITERATIONS * 10), standardGC);
        System.out.printf("Pooled:   %.3f µs avg, %d GCs\n", 
            pooledTime / 1_000.0 / (ITERATIONS * 10), pooledGC);
        System.out.printf("Speedup:  %.2fx\n", (double)standardTime / pooledTime);
        
        ArrayBufferPool.remove();
    }
    
    private long getGCCount() {
        long totalGCs = 0;
        for (java.lang.management.GarbageCollectorMXBean gc : 
             java.lang.management.ManagementFactory.getGarbageCollectorMXBeans()) {
            totalGCs += gc.getCollectionCount();
        }
        return totalGCs;
    }
}