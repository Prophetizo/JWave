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
import java.util.Arrays;

import jwave.transforms.wavelets.Wavelet;
import jwave.transforms.wavelets.daubechies.Daubechies4;
import jwave.transforms.wavelets.haar.Haar1;

/**
 * Test class for EnhancedSlidingMODWT.
 */
public class EnhancedSlidingMODWTTest {
    
    private static final double EPSILON = 1e-10;
    
    @Test
    public void testOptimizationStatsDetection() {
        EnhancedSlidingMODWT enhanced = new EnhancedSlidingMODWT(new Haar1());
        
        // Test sliding window detection
        double[] oldData = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] slidingLeft = {2, 3, 4, 5, 6, 7, 8, 9};
        double[] slidingRight = {0, 1, 2, 3, 4, 5, 6, 7};
        double[] notSliding = {1, 3, 5, 7, 9, 11, 13, 15};
        
        EnhancedSlidingMODWT.OptimizationStats stats;
        
        // Test left sliding
        stats = enhanced.getOptimizationStats(oldData, slidingLeft);
        assertTrue("Should detect sliding window", stats.isSlidingWindow);
        assertEquals("All coefficients are different", 8, stats.changedCoefficients);
        assertEquals("Change ratio should be 1.0", 1.0, stats.changeRatio, EPSILON);
        assertFalse("Should not be optimizable (100% change)", stats.isOptimizable());
        
        // Test right sliding
        stats = enhanced.getOptimizationStats(oldData, slidingRight);
        assertTrue("Should detect sliding window", stats.isSlidingWindow);
        
        // Test non-sliding
        stats = enhanced.getOptimizationStats(oldData, notSliding);
        assertFalse("Should not detect sliding window", stats.isSlidingWindow);
    }
    
    @Test
    public void testOptimizableScenario() {
        EnhancedSlidingMODWT enhanced = new EnhancedSlidingMODWT(new Daubechies4());
        
        // Create a scenario where only a small portion of data changes
        int size = 1000;
        double[] oldData = new double[size];
        double[] newData = new double[size];
        
        // Initialize with zeros
        Arrays.fill(oldData, 0.0);
        Arrays.fill(newData, 0.0);
        
        // Change only the last few values (simulating new samples arriving)
        oldData[size - 2] = 1.0;
        oldData[size - 1] = 2.0;
        
        newData[size - 2] = 3.0;
        newData[size - 1] = 4.0;
        
        EnhancedSlidingMODWT.OptimizationStats stats = enhanced.getOptimizationStats(oldData, newData);
        
        assertFalse("Should not be a sliding window", stats.isSlidingWindow);
        assertEquals("Only 2 coefficients changed", 2, stats.changedCoefficients);
        assertEquals("Change ratio should be 0.002", 0.002, stats.changeRatio, EPSILON);
        // Not optimizable because it's not a sliding window
        assertFalse("Should not be optimizable (not sliding)", stats.isOptimizable());
    }
    
    @Test
    public void testTrueSlidingWindowScenario() {
        EnhancedSlidingMODWT enhanced = new EnhancedSlidingMODWT(new Daubechies4());
        
        // Create a large array with minimal changes
        int size = 1000;
        double[] oldData = new double[size];
        double[] newData = new double[size];
        
        // Initialize with sine wave
        for (int i = 0; i < size; i++) {
            oldData[i] = Math.sin(2 * Math.PI * i / 64.0);
        }
        
        // Create sliding window - all values are different even though it's a shift
        System.arraycopy(oldData, 1, newData, 0, size - 1);
        newData[size - 1] = Math.sin(2 * Math.PI * size / 64.0);
        
        EnhancedSlidingMODWT.OptimizationStats stats = enhanced.getOptimizationStats(oldData, newData);
        
        assertTrue("Should detect sliding window", stats.isSlidingWindow);
        // Note: Even though it's a sliding window, ALL coefficients are different
        // because we're comparing values at different positions
        assertEquals("All coefficients changed", 1.0, stats.changeRatio, EPSILON);
        assertFalse("Should not be optimizable (100% change)", stats.isOptimizable());
    }
    
    @Test
    public void testCachingFunctionality() {
        EnhancedSlidingMODWT enhanced = new EnhancedSlidingMODWT(new Haar1());
        
        // Test data
        double[] data1 = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] data2 = {2, 3, 4, 5, 6, 7, 8, 9}; // Sliding window
        int maxLevel = 2;
        
        // Compute with caching enabled
        double[][] result1 = enhanced.compute(data1, maxLevel, true);
        assertNotNull("Result should not be null", result1);
        
        // Compute sliding window with caching
        double[][] result2 = enhanced.compute(data2, maxLevel, true);
        assertNotNull("Result should not be null", result2);
        
        // Clear cache and compute again
        enhanced.clearCache();
        double[][] result3 = enhanced.compute(data2, maxLevel, false);
        assertNotNull("Result should not be null", result3);
        
        // Results should be the same regardless of caching
        assertArrayEquals("Results should match", result2[0], result3[0], EPSILON);
    }
    
    @Test
    public void testPerformanceComparison() {
        Wavelet wavelet = new Daubechies4();
        EnhancedSlidingMODWT enhanced = new EnhancedSlidingMODWT(wavelet);
        MODWTTransform standard = new MODWTTransform(wavelet);
        
        // Large dataset
        int size = 1024;
        int maxLevel = 5;
        double[] data = new double[size];
        
        // Initialize with random data
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        
        // Warm up
        enhanced.compute(data, maxLevel, false);
        standard.forwardMODWT(data, maxLevel);
        
        // Time standard MODWT
        long startStandard = System.nanoTime();
        double[][] resultStandard = standard.forwardMODWT(data, maxLevel);
        long timeStandard = System.nanoTime() - startStandard;
        
        // Time enhanced MODWT (without caching, should be similar)
        long startEnhanced = System.nanoTime();
        double[][] resultEnhanced = enhanced.compute(data, maxLevel, false);
        long timeEnhanced = System.nanoTime() - startEnhanced;
        
        // Results should be identical
        for (int level = 0; level <= maxLevel; level++) {
            assertArrayEquals("Level " + level + " should match",
                            resultStandard[level], resultEnhanced[level], EPSILON);
        }
        
        // Performance should be comparable (within 20%)
        double ratio = (double) timeEnhanced / timeStandard;
        System.out.println("Enhanced/Standard time ratio: " + ratio);
        assertTrue("Enhanced should not be significantly slower", ratio < 1.2);
    }
}