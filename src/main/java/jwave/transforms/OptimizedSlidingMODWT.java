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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Optimized MODWT computation for sliding windows with full caching
 * and partial boundary updates.
 */
public class OptimizedSlidingMODWT {
    private final Wavelet wavelet;
    private final double[] g0; // Scaling filter (normalized) - low pass
    private final double[] h0; // Wavelet filter (normalized) - high pass
    private final int filterLength;
    
    // Caches for upsampled filters
    private final Map<Integer, double[]> gFilterCache;
    private final Map<Integer, double[]> hFilterCache;
    
    // Cache for intermediate approximation coefficients
    private CachedMODWTState cachedState;
    
    public OptimizedSlidingMODWT(Wavelet wavelet) {
        this.wavelet = wavelet;
        
        // Get filters from wavelet - use decomposition filters
        // Note: In MODWT/JWave convention, g is scaling and h is wavelet
        double[] g_dwt = wavelet.getScalingDeComposition().clone();
        double[] h_dwt = wavelet.getWaveletDeComposition().clone();
        
        // Normalize to unit energy
        normalize(g_dwt);
        normalize(h_dwt);
        
        // Scale by 1/sqrt(2) for MODWT
        double scaleFactor = Math.sqrt(2.0);
        this.g0 = new double[g_dwt.length];
        this.h0 = new double[h_dwt.length];
        
        for (int i = 0; i < g_dwt.length; i++) {
            this.g0[i] = g_dwt[i] / scaleFactor;
        }
        for (int i = 0; i < h_dwt.length; i++) {
            this.h0[i] = h_dwt[i] / scaleFactor;
        }
        
        this.filterLength = wavelet.getFilterLength();
        
        // Initialize filter caches
        this.gFilterCache = new HashMap<>();
        this.hFilterCache = new HashMap<>();
        
        this.cachedState = null;
    }
    
    /**
     * Performs MODWT with sliding window optimization.
     * 
     * @param data The input data
     * @param maxLevel Maximum decomposition level
     * @return The MODWT coefficients
     */
    public double[][] compute(double[] data, int maxLevel) {
        // Check if we can use cached state
        if (cachedState != null && cachedState.canOptimize(data, maxLevel)) {
            return optimizedUpdate(data, maxLevel);
        } else {
            return fullComputation(data, maxLevel);
        }
    }
    
    /**
     * Performs full MODWT computation and caches intermediate results.
     */
    private double[][] fullComputation(double[] data, int maxLevel) {
        int N = data.length;
        double[][] coeffs = new double[maxLevel + 1][N];
        double[][] vCoeffs = new double[maxLevel + 1][N]; // V_j approximations
        
        // V_0 is the input data
        System.arraycopy(data, 0, vCoeffs[0], 0, N);
        
        // Compute each level
        for (int j = 1; j <= maxLevel; j++) {
            double[] hj = getCachedHFilter(j);
            double[] gj = getCachedGFilter(j);
            
            // Compute W_j (detail) and V_j (approximation)
            // Note: Scaling is already included in the MODWT filters
            for (int t = 0; t < N; t++) {
                coeffs[j - 1][t] = circularConvolve(vCoeffs[j - 1], hj, t, N);
                vCoeffs[j][t] = circularConvolve(vCoeffs[j - 1], gj, t, N);
            }
        }
        
        // Copy final approximation coefficients
        System.arraycopy(vCoeffs[maxLevel], 0, coeffs[maxLevel], 0, N);
        
        // Cache the state
        cachedState = new CachedMODWTState(data.clone(), coeffs, vCoeffs, maxLevel);
        
        return coeffs;
    }
    
    /**
     * Performs optimized update for sliding windows.
     */
    private double[][] optimizedUpdate(double[] data, int maxLevel) {
        int N = data.length;
        SlidingWindowInfo slideInfo = cachedState.analyzeSlidingWindow(data);
        
        if (!slideInfo.isSliding) {
            // Not a sliding window, do full computation
            return fullComputation(data, maxLevel);
        }
        
        // Clone the cached coefficients
        double[][] coeffs = new double[maxLevel + 1][];
        double[][] vCoeffs = new double[maxLevel + 1][];
        for (int i = 0; i <= maxLevel; i++) {
            coeffs[i] = cachedState.coeffs[i].clone();
            vCoeffs[i] = cachedState.vCoeffs[i].clone();
        }
        
        // Update V_0 with new data
        System.arraycopy(data, 0, vCoeffs[0], 0, N);
        
        // Update each level, only computing affected coefficients
        for (int j = 1; j <= maxLevel; j++) {
            updateLevelOptimized(coeffs, vCoeffs, j, slideInfo, N);
        }
        
        // Update cached state
        cachedState = new CachedMODWTState(data.clone(), coeffs, vCoeffs, maxLevel);
        
        return coeffs;
    }
    
    /**
     * Updates only the affected coefficients at a given level.
     */
    private void updateLevelOptimized(double[][] coeffs, double[][] vCoeffs, 
                                     int level, SlidingWindowInfo slideInfo, int N) {
        double[] hj = getCachedHFilter(level);
        double[] gj = getCachedGFilter(level);
        
        // Calculate affected range based on filter support
        int affectedSize = getAffectedBoundarySize(level);
        
        if (slideInfo.isLeftShift) {
            // For left shift, update coefficients at the end
            int startIdx = Math.max(0, N - affectedSize);
            for (int t = startIdx; t < N; t++) {
                coeffs[level - 1][t] = circularConvolve(vCoeffs[level - 1], hj, t, N);
                vCoeffs[level][t] = circularConvolve(vCoeffs[level - 1], gj, t, N);
            }
        } else {
            // For right shift, update coefficients at the beginning
            int endIdx = Math.min(N, affectedSize);
            for (int t = 0; t < endIdx; t++) {
                coeffs[level - 1][t] = circularConvolve(vCoeffs[level - 1], hj, t, N);
                vCoeffs[level][t] = circularConvolve(vCoeffs[level - 1], gj, t, N);
            }
        }
        
        // Update final approximation if this is the last level
        if (level == coeffs.length - 1) {
            if (slideInfo.isLeftShift) {
                int startIdx = Math.max(0, N - affectedSize);
                System.arraycopy(vCoeffs[level], startIdx, coeffs[level], startIdx, N - startIdx);
            } else {
                int endIdx = Math.min(N, affectedSize);
                System.arraycopy(vCoeffs[level], 0, coeffs[level], 0, endIdx);
            }
        }
    }
    
    /**
     * Optimized circular convolution for a single point.
     */
    private double circularConvolve(double[] signal, double[] filter, int t, int N) {
        double sum = 0.0;
        int filterLen = filter.length;
        
        // Optimize for common case where filter is much smaller than signal
        if (filterLen < N / 4) {
            for (int m = 0; m < filterLen; m++) {
                if (filter[m] != 0.0) { // Skip zero coefficients
                    int idx = (t - m + N) % N;
                    sum += signal[idx] * filter[m];
                }
            }
        } else {
            // Standard implementation for larger filters
            for (int m = 0; m < filterLen; m++) {
                int idx = Math.floorMod(t - m, N);
                sum += signal[idx] * filter[m];
            }
        }
        
        return sum;
    }
    
    /**
     * Calculate affected boundary size at a given level.
     */
    private int getAffectedBoundarySize(int level) {
        // At level j, affected size is (2^(j-1)) * L
        // where L is the filter length
        return (1 << (level - 1)) * filterLength;
    }
    
    /**
     * Normalizes a filter to have unit energy (L2 norm = 1).
     * Modifies the filter in place.
     */
    private void normalize(double[] filter) {
        double energy = 0.0;
        for (double c : filter) {
            energy += c * c;
        }
        double norm = Math.sqrt(energy);
        if (norm > 1e-12) {
            for (int i = 0; i < filter.length; i++) {
                filter[i] /= norm;
            }
        }
    }
    
    /**
     * Get cached upsampled G filter for the specified level.
     */
    private double[] getCachedGFilter(int level) {
        return gFilterCache.computeIfAbsent(level, k -> upsample(g0, k));
    }
    
    /**
     * Get cached upsampled H filter for the specified level.
     */
    private double[] getCachedHFilter(int level) {
        return hFilterCache.computeIfAbsent(level, k -> upsample(h0, k));
    }
    
    /**
     * Upsamples a filter for a specific decomposition level.
     */
    private static double[] upsample(double[] filter, int level) {
        if (level <= 1) return filter;
        int gap = (1 << (level - 1)) - 1;
        int newLength = filter.length + (filter.length - 1) * gap;
        double[] upsampled = new double[newLength];
        for (int i = 0; i < filter.length; i++) {
            upsampled[i * (gap + 1)] = filter[i];
        }
        return upsampled;
    }
    
    /**
     * Clears all cached data.
     */
    public void clearCache() {
        cachedState = null;
        gFilterCache.clear();
        hFilterCache.clear();
    }
    
    /**
     * Inner class to store cached MODWT state.
     */
    private static class CachedMODWTState {
        final double[] data;
        final double[][] coeffs;
        final double[][] vCoeffs;
        final int maxLevel;
        
        CachedMODWTState(double[] data, double[][] coeffs, double[][] vCoeffs, int maxLevel) {
            this.data = data;
            this.coeffs = coeffs;
            this.vCoeffs = vCoeffs;
            this.maxLevel = maxLevel;
        }
        
        boolean canOptimize(double[] newData, int newMaxLevel) {
            if (data == null || newData.length != data.length || newMaxLevel != maxLevel) {
                return false;
            }
            
            // Check if it's a sliding window
            return isLeftShift(newData) || isRightShift(newData);
        }
        
        SlidingWindowInfo analyzeSlidingWindow(double[] newData) {
            boolean isLeft = isLeftShift(newData);
            boolean isRight = isRightShift(newData);
            return new SlidingWindowInfo(isLeft || isRight, isLeft);
        }
        
        private boolean isLeftShift(double[] newData) {
            for (int i = 0; i < data.length - 1; i++) {
                if (Math.abs(data[i + 1] - newData[i]) > 1e-10) {
                    return false;
                }
            }
            return true;
        }
        
        private boolean isRightShift(double[] newData) {
            for (int i = 0; i < data.length - 1; i++) {
                if (Math.abs(data[i] - newData[i + 1]) > 1e-10) {
                    return false;
                }
            }
            return true;
        }
    }
    
    /**
     * Information about sliding window.
     */
    private static class SlidingWindowInfo {
        final boolean isSliding;
        final boolean isLeftShift;
        
        SlidingWindowInfo(boolean isSliding, boolean isLeftShift) {
            this.isSliding = isSliding;
            this.isLeftShift = isLeftShift;
        }
    }
}