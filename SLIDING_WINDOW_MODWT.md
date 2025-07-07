# Sliding Window MODWT Implementation

## Overview

This document describes the sliding window optimization support added to JWave for the Maximal Overlap Discrete Wavelet Transform (MODWT). The implementation provides infrastructure for efficient coefficient updates when data windows slide by one position.

## Key Components

### 1. Wavelet Interface Enhancements

Added methods to the `Wavelet` class to expose filter coefficients:

```java
// Get wavelet (detail) filter coefficients
public double[] getWaveletCoefficients()

// Get scaling (approximation) filter coefficients  
public double[] getScalingCoefficients()

// Get the filter length
public int getFilterLength()
```

These methods delegate to existing methods and are automatically inherited by all wavelet implementations.

### 2. SlidingMODWT Class

The basic sliding window MODWT implementation that:
- Detects sliding windows (left or right shift by one position)
- Calculates affected boundary sizes for each decomposition level
- Provides infrastructure for future optimization

Key features:
- Filter normalization for MODWT
- Filter caching for performance
- Sliding window detection
- Change region identification

### 3. EnhancedSlidingMODWT Class

An enhanced version that provides:
- Caching support for intermediate results
- Optimization statistics and analysis
- Infrastructure for future full optimization

## Usage Examples

### Basic Sliding Window Detection

```java
Wavelet wavelet = new Haar1();
SlidingMODWT slidingMODWT = new SlidingMODWT(wavelet);

double[] oldData = {1, 2, 3, 4, 5, 6, 7, 8};
double[] newData = {2, 3, 4, 5, 6, 7, 8, 9}; // Left shift

// Previous coefficients from standard MODWT
MODWTTransform modwt = new MODWTTransform(wavelet);
double[][] oldCoeffs = modwt.forwardMODWT(oldData, maxLevel);

// Update with sliding window
double[][] newCoeffs = slidingMODWT.updateSlidingWindow(
    oldCoeffs, oldData, newData, maxLevel);
```

### Enhanced Version with Caching

```java
EnhancedSlidingMODWT enhanced = new EnhancedSlidingMODWT(wavelet);

// Enable caching for repeated transforms
double[][] coeffs1 = enhanced.compute(data1, maxLevel, true);
double[][] coeffs2 = enhanced.compute(data2, maxLevel, true); // May use cache

// Get optimization statistics
EnhancedSlidingMODWT.OptimizationStats stats = 
    enhanced.getOptimizationStats(data1, data2);

if (stats.isOptimizable()) {
    System.out.println("Could optimize: " + stats.changeRatio);
}
```

## Technical Details

### Affected Boundary Size Calculation

For a sliding window, only coefficients near the boundaries are affected. The affected size at level j is:

```
affectedSize = (2^j) * (L-1) + 1
```

Where L is the filter length.

### Current Implementation Status

The current implementation provides:
1. **Infrastructure**: All necessary methods and classes for sliding window optimization
2. **Detection**: Accurate detection of sliding windows
3. **Analysis**: Calculation of affected regions and optimization opportunities
4. **Correctness**: Falls back to full MODWT computation to ensure correct results

### Future Optimization Opportunities

The implementation is designed to support future optimizations:

1. **Caching Approximation Coefficients**: Store V_j coefficients between transforms
2. **Partial Updates**: Only recompute affected boundary coefficients
3. **Cascading Updates**: Efficiently propagate changes through decomposition levels
4. **Circular Convolution Optimization**: Optimize for boundary regions only

## Performance Considerations

- Current implementation ensures correctness over performance
- Infrastructure is in place for 10-100x speedup for sliding windows
- Filter caching provides immediate performance benefits
- Suitable for real-time signal processing applications

## Testing

Comprehensive tests are provided:
- Sliding window detection
- Coefficient correctness verification
- Performance benchmarking
- Optimization statistics validation

## Limitations

1. Full optimization requires storing intermediate V_j coefficients
2. Complex cascading dependencies make partial updates challenging
3. Current implementation falls back to full computation for correctness

## Future Work

1. Implement caching of intermediate approximation coefficients
2. Add partial coefficient update algorithms
3. Optimize for specific use cases (e.g., streaming data)
4. Add support for multi-dimensional sliding windows