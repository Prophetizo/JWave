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

import jwave.transforms.wavelets.Wavelet;
import java.util.HashMap;
import java.util.Map;

/**
 * Enhanced MODWT computation for sliding windows with caching support.
 * This class provides infrastructure for efficient MODWT updates when data
 * slides by one position, with support for caching intermediate results.
 */
public class EnhancedSlidingMODWT {
    private final Wavelet wavelet;
    private final MODWTTransform modwt;
    
    // Cache for storing intermediate approximation coefficients
    private Map<Integer, double[][]> approximationCache;
    private double[] lastProcessedData;
    private int lastMaxLevel;
    
    public EnhancedSlidingMODWT(Wavelet wavelet) {
        this.wavelet = wavelet;
        this.modwt = new MODWTTransform(wavelet);
        this.approximationCache = new HashMap<>();
        this.lastProcessedData = null;
        this.lastMaxLevel = -1;
    }
    
    /**
     * Performs MODWT with sliding window optimization when possible.
     * 
     * @param data The input data
     * @param maxLevel Maximum decomposition level
     * @param enableCaching Whether to cache intermediate results
     * @return The MODWT coefficients
     */
    public double[][] compute(double[] data, int maxLevel, boolean enableCaching) {
        // Check if this is a sliding window update
        if (enableCaching && lastProcessedData != null && 
            lastMaxLevel == maxLevel && isSlidingWindow(lastProcessedData, data)) {
            
            // Attempt optimized update
            double[][] optimized = attemptOptimizedUpdate(data, maxLevel);
            if (optimized != null) {
                updateCache(data, maxLevel, optimized);
                return optimized;
            }
        }
        
        // Full computation
        double[][] result = modwt.forwardMODWT(data, maxLevel);
        
        if (enableCaching) {
            updateCache(data, maxLevel, result);
        }
        
        return result;
    }
    
    /**
     * Checks if the new data is a sliding window of the previous data.
     */
    private boolean isSlidingWindow(double[] oldData, double[] newData) {
        if (oldData.length != newData.length) {
            return false;
        }
        
        // Check for left shift
        boolean isLeftShift = true;
        for (int i = 0; i < oldData.length - 1; i++) {
            if (Math.abs(oldData[i + 1] - newData[i]) > 1e-10) {
                isLeftShift = false;
                break;
            }
        }
        
        if (isLeftShift) {
            return true;
        }
        
        // Check for right shift
        boolean isRightShift = true;
        for (int i = 0; i < oldData.length - 1; i++) {
            if (Math.abs(oldData[i] - newData[i + 1]) > 1e-10) {
                isRightShift = false;
                break;
            }
        }
        
        return isRightShift;
    }
    
    /**
     * Attempts to perform an optimized update for sliding windows.
     * Returns null if optimization is not possible.
     */
    private double[][] attemptOptimizedUpdate(double[] data, int maxLevel) {
        // Future implementation would use cached approximation coefficients
        // to update only affected regions
        
        // For now, return null to indicate full computation is needed
        // This provides the infrastructure for future optimization
        return null;
    }
    
    /**
     * Updates the cache with new results.
     */
    private void updateCache(double[] data, int maxLevel, double[][] coefficients) {
        lastProcessedData = data.clone();
        lastMaxLevel = maxLevel;
        
        // In a full implementation, we would also cache intermediate
        // approximation coefficients V_j for each level
    }
    
    /**
     * Clears all cached data.
     */
    public void clearCache() {
        approximationCache.clear();
        lastProcessedData = null;
        lastMaxLevel = -1;
    }
    
    /**
     * Gets statistics about potential optimization opportunities.
     */
    public OptimizationStats getOptimizationStats(double[] oldData, double[] newData) {
        if (oldData.length != newData.length) {
            return new OptimizationStats(0, 0, false);
        }
        
        int changedCount = 0;
        for (int i = 0; i < oldData.length; i++) {
            if (Math.abs(oldData[i] - newData[i]) > 1e-10) {
                changedCount++;
            }
        }
        
        double changeRatio = (double) changedCount / oldData.length;
        boolean isSlidingWindow = isSlidingWindow(oldData, newData);
        
        return new OptimizationStats(changedCount, changeRatio, isSlidingWindow);
    }
    
    /**
     * Statistics about optimization opportunities.
     */
    public static class OptimizationStats {
        public final int changedCoefficients;
        public final double changeRatio;
        public final boolean isSlidingWindow;
        
        public OptimizationStats(int changedCoefficients, double changeRatio, boolean isSlidingWindow) {
            this.changedCoefficients = changedCoefficients;
            this.changeRatio = changeRatio;
            this.isSlidingWindow = isSlidingWindow;
        }
        
        public boolean isOptimizable() {
            return isSlidingWindow && changeRatio < 0.1;
        }
    }
}