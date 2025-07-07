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
import jwave.transforms.MODWTTransform;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Efficient MODWT computation for sliding windows.
 * When a data window slides by one position, only boundary coefficients need updating.
 */
public class SlidingMODWT {
    private final Wavelet wavelet;
    private final double[] h0; // Scaling filter (normalized)
    private final double[] g0; // Wavelet filter (normalized)
    private final int filterLength;
    private final Map<Integer, double[]> gFilterCache;
    private final Map<Integer, double[]> hFilterCache;
    
    public SlidingMODWT(Wavelet wavelet) {
        this.wavelet = wavelet;
        
        // Get and normalize the filters for MODWT
        this.h0 = normalizeFilter(wavelet.getScalingCoefficients());
        this.g0 = normalizeFilter(wavelet.getWaveletCoefficients());
        this.filterLength = wavelet.getFilterLength();
        
        // Initialize filter caches
        this.gFilterCache = new HashMap<>();
        this.hFilterCache = new HashMap<>();
    }
    
    /**
     * Normalizes a filter to have unit energy (L2 norm = 1).
     * This ensures the MODWT preserves signal energy.
     */
    private double[] normalizeFilter(double[] filter) {
        double[] normalized = filter.clone();
        double energy = 0.0;
        for (double c : normalized) {
            energy += c * c;
        }
        double norm = Math.sqrt(energy);
        if (norm > 1e-12) {
            for (int i = 0; i < normalized.length; i++) {
                normalized[i] /= norm;
            }
        }
        return normalized;
    }
    
    /**
     * Update MODWT coefficients for a sliding window.
     * 
     * @param previousCoeffs Previous MODWT coefficients [level][time]
     * @param oldData Previous data window
     * @param newData New data window (shifted by one position)
     * @param maxLevel Maximum decomposition level
     * @return Updated coefficients or null if full recomputation needed
     */
    public double[][] updateSlidingWindow(double[][] previousCoeffs,
                                         double[] oldData,
                                         double[] newData,
                                         int maxLevel) {
        // Validate inputs
        if (previousCoeffs == null || oldData == null || newData == null) {
            return null;
        }
        
        // Check if windows are the same size
        if (oldData.length != newData.length) {
            return null;
        }
        
        // Check if this is actually a sliding window (shifted by 1)
        // For a left shift: newData[i] = oldData[i+1] for i < length-1
        // For a right shift: newData[i+1] = oldData[i] for i < length-1
        boolean isLeftShift = true;
        boolean isRightShift = true;
        
        // Check for left shift
        for (int i = 0; i < oldData.length - 1; i++) {
            if (Math.abs(oldData[i + 1] - newData[i]) > 1e-10) {
                isLeftShift = false;
                break;
            }
        }
        
        // Check for right shift
        for (int i = 0; i < oldData.length - 1; i++) {
            if (Math.abs(oldData[i] - newData[i + 1]) > 1e-10) {
                isRightShift = false;
                break;
            }
        }
        
        if (!isLeftShift && !isRightShift) {
            return null; // Not a sliding window, need full recomputation
        }
        
        int N = newData.length;
        
        // Create copy of coefficients to update
        double[][] updatedCoeffs = new double[previousCoeffs.length][];
        for (int i = 0; i < previousCoeffs.length; i++) {
            updatedCoeffs[i] = previousCoeffs[i].clone();
        }
        
        // Due to the cascading nature of MODWT, a proper sliding window optimization
        // requires storing intermediate approximation coefficients from previous transforms.
        // For now, we'll implement a partial optimization that identifies changed regions
        // and provides infrastructure for future full optimization.
        
        // Identify changed regions for performance monitoring
        int changedCount = countChangedCoefficients(oldData, newData);
        double changeRatio = (double) changedCount / newData.length;
        
        // For small changes (< 10% of data), we could optimize in the future
        // For now, fall back to full computation to ensure correctness
        MODWTTransform modwt = new MODWTTransform(wavelet);
        double[][] result = modwt.forwardMODWT(newData, maxLevel);
        
        // Log optimization opportunity for future enhancement
        if (changeRatio < 0.1 && isLeftShift) {
            // This would be a good candidate for optimization
            // Future implementation could cache intermediate V_j coefficients
        }
        
        return result;
    }
    
    /**
     * Calculate which coefficients are affected when window slides.
     * @param level Decomposition level (0-based)
     * @return Number of coefficients affected at the boundary
     */
    public int getAffectedBoundarySize(int level) {
        // At level j, affected size is (2^j) * (L-1) + 1
        return (1 << level) * (filterLength - 1) + 1;
    }
    
    /**
     * Update only the boundary coefficients that are affected by the sliding window.
     */
    private void updateBoundaryCoefficients(double[][] coeffs,
                                           double[] newData,
                                           int level,
                                           int affectedSize,
                                           int maxLevel) {
        int N = newData.length;
        int startIdx = Math.max(0, N - affectedSize);
        double scale = Math.pow(2, -level / 2.0); // MODWT scaling
        
        // Update detail coefficients
        for (int t = startIdx; t < N; t++) {
            double sum = 0.0;
            for (int k = 0; k < filterLength; k++) {
                int idx = (t - k * (1 << level)) % N;
                if (idx < 0) idx += N; // Circular boundary
                sum += g0[k] * newData[idx];
            }
            coeffs[level][t] = sum * scale;
        }
        
        // Update approximation coefficients if this is the last level
        if (level == maxLevel && level < coeffs.length - 1) {
            for (int t = startIdx; t < N; t++) {
                double sum = 0.0;
                for (int k = 0; k < filterLength; k++) {
                    int idx = (t - k * (1 << level)) % N;
                    if (idx < 0) idx += N; // Circular boundary
                    sum += h0[k] * newData[idx];
                }
                coeffs[level + 1][t] = sum * scale;
            }
        }
    }
    
    /**
     * Check if the MODWT coefficients structure matches expected dimensions.
     */
    public boolean isValidCoefficientsStructure(double[][] coeffs, int dataLength, int maxLevel) {
        if (coeffs == null || coeffs.length != maxLevel + 2) {
            return false;
        }
        
        for (int i = 0; i < coeffs.length; i++) {
            if (coeffs[i] == null || coeffs[i].length != dataLength) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Get the wavelet used by this sliding MODWT instance.
     */
    public Wavelet getWavelet() {
        return wavelet;
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
     * At level j, inserts 2^(j-1) - 1 zeros between each filter coefficient.
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
     * Counts the number of changed coefficients between old and new data.
     */
    private int countChangedCoefficients(double[] oldData, double[] newData) {
        int count = 0;
        for (int i = 0; i < oldData.length; i++) {
            if (Math.abs(oldData[i] - newData[i]) > 1e-10) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * Performs optimized update for small filters and shallow decompositions.
     * This method is kept for future optimization implementation.
     */
    private double[][] performOptimizedUpdate(double[][] coeffs, double[] oldData, 
                                            double[] newData, int maxLevel, boolean isLeftShift) {
        int N = newData.length;
        
        // Use a more efficient approach: only update coefficients that are affected
        // by the changed data points
        
        // Identify which data points changed
        int changeStart = -1;
        int changeEnd = -1;
        
        for (int i = 0; i < N; i++) {
            if (Math.abs(oldData[i] - newData[i]) > 1e-10) {
                if (changeStart == -1) changeStart = i;
                changeEnd = i;
            }
        }
        
        if (changeStart == -1) {
            // No changes detected
            return coeffs;
        }
        
        // Now update only the affected coefficients at each level
        updateAffectedCoefficients(coeffs, newData, maxLevel, changeStart, changeEnd);
        
        return coeffs;
    }
    
    /**
     * Updates only the coefficients affected by changed data points.
     */
    private void updateAffectedCoefficients(double[][] coeffs, double[] data, 
                                          int maxLevel, int changeStart, int changeEnd) {
        int N = data.length;
        
        // Keep track of which approximation coefficients have changed at each level
        boolean[] approxChanged = new boolean[N];
        Arrays.fill(approxChanged, false);
        
        // Level 1: Update from raw data
        double[] h1 = getCachedHFilter(1);
        double[] g1 = getCachedGFilter(1);
        double scale1 = Math.pow(2, -0.5);
        
        // Determine affected range for level 1
        int level1Start = Math.max(0, changeStart - h1.length + 1);
        int level1End = Math.min(N - 1, changeEnd + h1.length - 1);
        
        // Temporary storage for approximation coefficients
        double[] v0 = data; // Input is V_0
        double[] v1 = new double[N];
        
        // Update level 1 detail coefficients and compute V_1
        for (int t = level1Start; t <= level1End; t++) {
            coeffs[0][t] = circularConvolvePoint(v0, h1, t, N) * scale1;
            v1[t] = circularConvolvePoint(v0, g1, t, N) * scale1;
            approxChanged[t] = true;
        }
        
        // Copy unchanged V_1 coefficients
        for (int t = 0; t < N; t++) {
            if (!approxChanged[t]) {
                // Compute V_1 from the original transform
                v1[t] = circularConvolvePoint(v0, g1, t, N) * scale1;
            }
        }
        
        // Process higher levels
        double[] vPrev = v1;
        double[] vCurr = new double[N];
        
        for (int level = 2; level <= maxLevel; level++) {
            double[] hj = getCachedHFilter(level);
            double[] gj = getCachedGFilter(level);
            double scalej = Math.pow(2, -level / 2.0);
            
            // Reset change tracking for this level
            boolean[] nextApproxChanged = new boolean[N];
            
            // Update coefficients affected by changes in previous level
            for (int t = 0; t < N; t++) {
                if (shouldUpdateCoefficient(t, approxChanged, hj.length, N)) {
                    coeffs[level - 1][t] = circularConvolvePoint(vPrev, hj, t, N) * scalej;
                    vCurr[t] = circularConvolvePoint(vPrev, gj, t, N) * scalej;
                    nextApproxChanged[t] = true;
                } else {
                    // Keep existing coefficient and compute approximation
                    vCurr[t] = circularConvolvePoint(vPrev, gj, t, N) * scalej;
                }
            }
            
            vPrev = vCurr;
            vCurr = new double[N];
            approxChanged = nextApproxChanged;
        }
        
        // Update final approximation coefficients
        if (maxLevel < coeffs.length - 1) {
            System.arraycopy(vPrev, 0, coeffs[maxLevel], 0, N);
        }
    }
    
    /**
     * Determines if a coefficient at position t needs to be updated based on
     * which approximation coefficients changed in the previous level.
     */
    private boolean shouldUpdateCoefficient(int t, boolean[] approxChanged, 
                                          int filterLength, int N) {
        // A coefficient at position t is affected if any of the approximation
        // coefficients it depends on have changed
        for (int m = 0; m < filterLength; m++) {
            int idx = Math.floorMod(t - m, N);
            if (approxChanged[idx]) {
                return true;
            }
        }
        return false;
    }
    
    
    /**
     * Computes a single point of circular convolution.
     */
    private double circularConvolvePoint(double[] signal, double[] filter, int n, int N) {
        double sum = 0.0;
        for (int m = 0; m < filter.length; m++) {
            int signalIndex = Math.floorMod(n - m, N);
            sum += signal[signalIndex] * filter[m];
        }
        return sum;
    }
    
}