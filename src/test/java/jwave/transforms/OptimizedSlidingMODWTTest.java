/**
 * JWave is distributed under the MIT License (MIT); this file is part of.
 *
 * Copyright (c) 2008-2024 Christian (graetz23@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package jwave.transforms;

import static org.junit.Assert.*;
import org.junit.Test;

import jwave.transforms.wavelets.Wavelet;
import jwave.transforms.wavelets.daubechies.Daubechies4;
import jwave.transforms.wavelets.haar.Haar1;

/**
 * Test class for OptimizedSlidingMODWT.
 */
public class OptimizedSlidingMODWTTest {
    
    private static final double EPSILON = 1e-10;
    
    @Test
    public void testCorrectnessWithHaar() {
        Wavelet wavelet = new Haar1();
        OptimizedSlidingMODWT optimized = new OptimizedSlidingMODWT(wavelet);
        MODWTTransform standard = new MODWTTransform(wavelet);
        
        // Test data
        double[] data = {1, 2, 3, 4, 5, 6, 7, 8};
        int maxLevel = 2;
        
        // Compute with both methods
        double[][] optimizedResult = optimized.compute(data, maxLevel);
        double[][] standardResult = standard.forwardMODWT(data, maxLevel);
        
        // Verify results match
        assertEquals("Same number of levels", standardResult.length, optimizedResult.length);
        for (int level = 0; level < standardResult.length; level++) {
            assertArrayEquals("Level " + level + " should match",
                            standardResult[level], optimizedResult[level], EPSILON);
        }
    }
    
    @Test
    public void testSlidingWindowOptimization() {
        Wavelet wavelet = new Daubechies4();
        OptimizedSlidingMODWT optimized = new OptimizedSlidingMODWT(wavelet);
        MODWTTransform standard = new MODWTTransform(wavelet);
        
        int N = 64;
        int maxLevel = 3;
        
        // Create initial data
        double[] data1 = new double[N];
        for (int i = 0; i < N; i++) {
            data1[i] = Math.sin(2 * Math.PI * i / 16.0);
        }
        
        // Create sliding window (left shift)
        double[] data2 = new double[N];
        System.arraycopy(data1, 1, data2, 0, N - 1);
        data2[N - 1] = Math.sin(2 * Math.PI * N / 16.0);
        
        // First computation (should cache)
        double[][] result1 = optimized.compute(data1, maxLevel);
        
        // Second computation (should use optimization)
        double[][] result2 = optimized.compute(data2, maxLevel);
        
        // Compare with standard MODWT
        double[][] expected = standard.forwardMODWT(data2, maxLevel);
        
        // Verify correctness
        for (int level = 0; level <= maxLevel; level++) {
            assertArrayEquals("Level " + level + " should match",
                            expected[level], result2[level], EPSILON);
        }
    }
    
    @Test
    public void testPerformanceImprovement() {
        Wavelet wavelet = new Daubechies4();
        OptimizedSlidingMODWT optimized = new OptimizedSlidingMODWT(wavelet);
        MODWTTransform standard = new MODWTTransform(wavelet);
        
        int N = 1024;
        int maxLevel = 5;
        int iterations = 100;
        
        // Create test data sequence
        double[][] dataSequence = new double[iterations][N];
        for (int iter = 0; iter < iterations; iter++) {
            if (iter == 0) {
                // Initial data
                for (int i = 0; i < N; i++) {
                    dataSequence[iter][i] = Math.sin(2 * Math.PI * i / 64.0) + 
                                           0.5 * Math.sin(2 * Math.PI * i / 16.0);
                }
            } else {
                // Sliding window
                System.arraycopy(dataSequence[iter-1], 1, dataSequence[iter], 0, N - 1);
                dataSequence[iter][N - 1] = Math.sin(2 * Math.PI * (N + iter - 1) / 64.0) + 
                                           0.5 * Math.sin(2 * Math.PI * (N + iter - 1) / 16.0);
            }
        }
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            optimized.compute(dataSequence[0], maxLevel);
            standard.forwardMODWT(dataSequence[0], maxLevel);
        }
        
        // Time standard MODWT
        long startStandard = System.nanoTime();
        for (int iter = 0; iter < iterations; iter++) {
            standard.forwardMODWT(dataSequence[iter], maxLevel);
        }
        long timeStandard = System.nanoTime() - startStandard;
        
        // Clear cache for fair comparison
        optimized.clearCache();
        
        // Time optimized MODWT
        long startOptimized = System.nanoTime();
        for (int iter = 0; iter < iterations; iter++) {
            optimized.compute(dataSequence[iter], maxLevel);
        }
        long timeOptimized = System.nanoTime() - startOptimized;
        
        // Calculate speedup
        double speedup = (double) timeStandard / timeOptimized;
        System.out.println("Standard time: " + timeStandard + " ns");
        System.out.println("Optimized time: " + timeOptimized + " ns");
        System.out.println("Speedup: " + speedup + "x");
        
        // Should be faster with optimization
        assertTrue("Optimized should be faster for sliding windows", speedup > 1.0);
    }
    
    @Test
    public void testRightShiftOptimization() {
        Wavelet wavelet = new Haar1();
        OptimizedSlidingMODWT optimized = new OptimizedSlidingMODWT(wavelet);
        MODWTTransform standard = new MODWTTransform(wavelet);
        
        int N = 32;
        int maxLevel = 2;
        
        // Create initial data
        double[] data1 = new double[N];
        for (int i = 0; i < N; i++) {
            data1[i] = i + 1.0;
        }
        
        // Create right shift
        double[] data2 = new double[N];
        data2[0] = 0.0;
        System.arraycopy(data1, 0, data2, 1, N - 1);
        
        // Compute with optimization
        optimized.compute(data1, maxLevel); // Cache
        double[][] result = optimized.compute(data2, maxLevel);
        
        // Compare with standard
        double[][] expected = standard.forwardMODWT(data2, maxLevel);
        
        for (int level = 0; level <= maxLevel; level++) {
            assertArrayEquals("Level " + level + " should match",
                            expected[level], result[level], EPSILON);
        }
    }
    
    @Test
    public void testCacheClearance() {
        Wavelet wavelet = new Daubechies4();
        OptimizedSlidingMODWT optimized = new OptimizedSlidingMODWT(wavelet);
        
        double[] data1 = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] data2 = {2, 3, 4, 5, 6, 7, 8, 9};
        int maxLevel = 2;
        
        // First computation
        double[][] result1 = optimized.compute(data1, maxLevel);
        
        // Clear cache
        optimized.clearCache();
        
        // Second computation should not use optimization
        double[][] result2 = optimized.compute(data2, maxLevel);
        
        assertNotNull("Results should not be null", result1);
        assertNotNull("Results should not be null", result2);
    }
    
    @Test
    public void testLargeScalePerformance() {
        Wavelet wavelet = new Daubechies4();
        OptimizedSlidingMODWT optimized = new OptimizedSlidingMODWT(wavelet);
        
        int N = 4096;
        int maxLevel = 8;
        
        // Create large dataset
        double[] data1 = new double[N];
        double[] data2 = new double[N];
        
        for (int i = 0; i < N; i++) {
            data1[i] = Math.random();
        }
        
        // Create sliding window
        System.arraycopy(data1, 1, data2, 0, N - 1);
        data2[N - 1] = Math.random();
        
        // Time the operations
        long start1 = System.nanoTime();
        double[][] result1 = optimized.compute(data1, maxLevel);
        long time1 = System.nanoTime() - start1;
        
        long start2 = System.nanoTime();
        double[][] result2 = optimized.compute(data2, maxLevel);
        long time2 = System.nanoTime() - start2;
        
        System.out.println("First computation: " + time1 + " ns");
        System.out.println("Second computation (optimized): " + time2 + " ns");
        System.out.println("Optimization ratio: " + (double) time1 / time2 + "x");
        
        // Second should be significantly faster
        assertTrue("Second computation should be faster", time2 < time1);
        
        // Verify some coefficients changed
        boolean someChanged = false;
        for (int level = 0; level <= maxLevel; level++) {
            for (int i = N - 100; i < N; i++) {
                if (Math.abs(result1[level][i] - result2[level][i]) > EPSILON) {
                    someChanged = true;
                    break;
                }
            }
        }
        assertTrue("Some coefficients should have changed", someChanged);
    }
}