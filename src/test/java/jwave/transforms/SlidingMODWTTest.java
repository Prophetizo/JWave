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
 * Test class for SlidingMODWT sliding window optimization.
 */
public class SlidingMODWTTest {
    
    private static final double EPSILON = 1e-10;
    
    @Test
    public void testWaveletCoefficientAccess() {
        Wavelet wavelet = new Daubechies4();
        
        // Test that we can access coefficients
        double[] waveletCoeffs = wavelet.getWaveletCoefficients();
        double[] scalingCoeffs = wavelet.getScalingCoefficients();
        int filterLength = wavelet.getFilterLength();
        
        assertNotNull("Wavelet coefficients should not be null", waveletCoeffs);
        assertNotNull("Scaling coefficients should not be null", scalingCoeffs);
        assertEquals("Filter length should be 8 for Daubechies4", 8, filterLength);
        assertEquals("Wavelet coefficients length should match filter length", filterLength, waveletCoeffs.length);
        assertEquals("Scaling coefficients length should match filter length", filterLength, scalingCoeffs.length);
    }
    
    @Test
    public void testSlidingWindowDetection() {
        Wavelet wavelet = new Haar1();
        SlidingMODWT slidingMODWT = new SlidingMODWT(wavelet);
        
        // Create test data
        double[] oldData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] newData = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // Shifted by 1
        double[] nonSliding = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0}; // Not sliding
        
        // Create dummy previous coefficients
        int maxLevel = 2;
        double[][] previousCoeffs = new double[maxLevel + 2][oldData.length];
        
        // Test sliding window detection
        double[][] result = slidingMODWT.updateSlidingWindow(previousCoeffs, oldData, newData, maxLevel);
        assertNotNull("Should detect valid sliding window", result);
        
        // Test non-sliding window
        result = slidingMODWT.updateSlidingWindow(previousCoeffs, oldData, nonSliding, maxLevel);
        assertNull("Should return null for non-sliding window", result);
    }
    
    @Test
    public void testAffectedBoundarySize() {
        Wavelet wavelet = new Daubechies4();
        SlidingMODWT slidingMODWT = new SlidingMODWT(wavelet);
        
        // For Daubechies4, filter length is 8
        // At level j, affected size is (2^j) * (L-1) + 1
        assertEquals("Level 0 affected size", 8, slidingMODWT.getAffectedBoundarySize(0));
        assertEquals("Level 1 affected size", 15, slidingMODWT.getAffectedBoundarySize(1));
        assertEquals("Level 2 affected size", 29, slidingMODWT.getAffectedBoundarySize(2));
        assertEquals("Level 3 affected size", 57, slidingMODWT.getAffectedBoundarySize(3));
    }
    
    @Test
    public void testSlidingWindowUpdateWithMODWT() {
        Wavelet wavelet = new Haar1();
        SlidingMODWT slidingMODWT = new SlidingMODWT(wavelet);
        MODWTTransform modwt = new MODWTTransform(wavelet);
        
        // Create test data
        int dataLength = 64;
        double[] oldData = new double[dataLength];
        double[] newData = new double[dataLength];
        
        // Initialize with sliding window data
        for (int i = 0; i < dataLength; i++) {
            oldData[i] = Math.sin(2 * Math.PI * i / 16.0);
        }
        
        // Create sliding window by shifting left
        for (int i = 0; i < dataLength - 1; i++) {
            newData[i] = oldData[i + 1]; // Shift left by 1
        }
        newData[dataLength - 1] = Math.sin(2 * Math.PI * dataLength / 16.0); // New value
        
        // Compute full MODWT for both windows
        int maxLevel = 3;
        double[][] oldCoeffs = modwt.forwardMODWT(oldData, maxLevel);
        double[][] newCoeffsExpected = modwt.forwardMODWT(newData, maxLevel);
        
        // Debug: verify coefficient structure
        assertEquals("Old coeffs should have correct levels", maxLevel + 1, oldCoeffs.length);
        assertEquals("New coeffs should have correct levels", maxLevel + 1, newCoeffsExpected.length);
        
        // Update using sliding window
        double[][] newCoeffsSliding = slidingMODWT.updateSlidingWindow(oldCoeffs, oldData, newData, maxLevel);
        
        if (newCoeffsSliding == null) {
            // Debug why it's null
            System.out.println("Debug: Sliding window returned null");
            System.out.println("OldData length: " + oldData.length);
            System.out.println("NewData length: " + newData.length);
            System.out.println("First 5 old values: " + oldData[0] + ", " + oldData[1] + ", " + oldData[2] + ", " + oldData[3] + ", " + oldData[4]);
            System.out.println("First 5 new values: " + newData[0] + ", " + newData[1] + ", " + newData[2] + ", " + newData[3] + ", " + newData[4]);
            System.out.println("Check sliding: oldData[1]=" + oldData[1] + " vs newData[0]=" + newData[0]);
        }
        
        assertNotNull("Sliding window update should not return null", newCoeffsSliding);
        
        // Compare only the affected boundary coefficients
        for (int level = 0; level <= maxLevel; level++) {
            int affectedSize = slidingMODWT.getAffectedBoundarySize(level);
            int startIdx = Math.max(0, dataLength - affectedSize);
            
            for (int i = startIdx; i < dataLength; i++) {
                assertEquals("Level " + level + " coefficient at index " + i,
                            newCoeffsExpected[level][i], newCoeffsSliding[level][i], EPSILON);
            }
        }
    }
    
    @Test
    public void testCoefficientsStructureValidation() {
        Wavelet wavelet = new Haar1();
        SlidingMODWT slidingMODWT = new SlidingMODWT(wavelet);
        
        int dataLength = 32;
        int maxLevel = 3;
        
        // Valid structure
        double[][] validCoeffs = new double[maxLevel + 2][dataLength];
        assertTrue("Should validate correct structure", 
                  slidingMODWT.isValidCoefficientsStructure(validCoeffs, dataLength, maxLevel));
        
        // Invalid structures
        assertFalse("Should reject null coefficients", 
                   slidingMODWT.isValidCoefficientsStructure(null, dataLength, maxLevel));
        
        double[][] wrongLevels = new double[maxLevel + 1][dataLength];
        assertFalse("Should reject wrong number of levels", 
                   slidingMODWT.isValidCoefficientsStructure(wrongLevels, dataLength, maxLevel));
        
        double[][] wrongLength = new double[maxLevel + 2][dataLength - 1];
        assertFalse("Should reject wrong data length", 
                   slidingMODWT.isValidCoefficientsStructure(wrongLength, dataLength, maxLevel));
    }
    
    @Test
    public void testPerformanceImprovement() {
        Wavelet wavelet = new Daubechies4();
        SlidingMODWT slidingMODWT = new SlidingMODWT(wavelet);
        MODWTTransform modwt = new MODWTTransform(wavelet);
        
        // Create larger test data for performance comparison
        int dataLength = 1024;
        int maxLevel = 5;
        double[] oldData = new double[dataLength];
        double[] newData = new double[dataLength];
        
        // Initialize with random data
        for (int i = 0; i < dataLength; i++) {
            oldData[i] = Math.random();
        }
        
        // Create sliding window by shifting left
        for (int i = 0; i < dataLength - 1; i++) {
            newData[i] = oldData[i + 1];
        }
        newData[dataLength - 1] = Math.random();
        
        // Compute initial coefficients
        double[][] oldCoeffs = modwt.forwardMODWT(oldData, maxLevel);
        
        // Measure sliding window update time
        long startSliding = System.nanoTime();
        double[][] slidingResult = slidingMODWT.updateSlidingWindow(oldCoeffs, oldData, newData, maxLevel);
        long slidingTime = System.nanoTime() - startSliding;
        
        // Measure full MODWT computation time
        long startFull = System.nanoTime();
        double[][] fullResult = modwt.forwardMODWT(newData, maxLevel);
        long fullTime = System.nanoTime() - startFull;
        
        assertNotNull("Sliding window result should not be null", slidingResult);
        
        // The sliding window update should be significantly faster
        // Note: This is a soft assertion as timing can vary
        System.out.println("Sliding window time: " + slidingTime + " ns");
        System.out.println("Full MODWT time: " + fullTime + " ns");
        System.out.println("Speedup: " + (double)fullTime / slidingTime + "x");
    }
}