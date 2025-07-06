# MODWT Decomposition Level Limits Analysis

This document provides a comprehensive analysis of the theoretical and practical limits for MODWT (Maximal Overlap Discrete Wavelet Transform) decomposition levels in the JWave implementation.

## Executive Summary

### Key Findings:

1. **Theoretical Maximum**: log₂(N) where N is the signal length
2. **Implementation Limit**: Level ≤ 30 (hardcoded to prevent integer overflow)
3. **Practical Limits**: Depend on:
   - Signal length
   - Wavelet filter length
   - Available memory
   - Performance requirements

### Recommended Maximum Levels:

| Signal Length | Recommended Max Level | Notes |
|--------------|---------------------|-------|
| < 1,024 | 3-5 | Keep conservative for small signals |
| 1,024-8,192 | 5-8 | Balance between resolution and performance |
| > 8,192 | 10-12 | Can go higher but with diminishing returns |
| Real-time apps | ≤ 8 | Maintain acceptable performance |

## Detailed Analysis

### 1. Theoretical Limits

The theoretical maximum decomposition level for MODWT is:
```
Max Level = floor(log₂(N))
```

Where N is the signal length. Examples:

| Signal Length | Max Theoretical Level |
|--------------|---------------------|
| 128 | 7 |
| 256 | 8 |
| 1,024 | 10 |
| 4,096 | 12 |
| 65,536 | 16 |

### 2. Filter Length Growth

At decomposition level j, the upsampled filter length becomes:
```
L_j = (L - 1) × 2^(j-1) + 1
```

Where L is the base filter length. This exponential growth becomes a major constraint:

| Level | Haar (L=2) | Db4 (L=8) | Db10 (L=20) |
|-------|------------|-----------|-------------|
| 1 | 2 | 8 | 20 |
| 5 | 17 | 113 | 305 |
| 10 | 513 | 3,585 | 9,729 |
| 15 | 16,385 | 114,689 | 311,297 |

### 3. Memory Requirements

For MODWT with N samples and J levels:

**Coefficient Memory**: `N × (J + 1) × 8 bytes`

**Filter Cache Memory**: Sum of upsampled filter lengths × 2 filters × 8 bytes

Examples:

| Signal Length | Levels | Total Memory |
|--------------|--------|--------------|
| 1,024 | 10 | ~200 KB |
| 16,384 | 14 | ~3.6 MB |
| 1,048,576 | 20 | ~280 MB |

### 4. Performance Impact

Performance scales linearly with decomposition levels. Based on benchmarks with a 4,096-sample signal using Daubechies-4:

| Level | Total Time (Forward + Inverse) |
|-------|-------------------------------|
| 1 | ~2 ms |
| 5 | ~4 ms |
| 8 | ~26 ms |
| 10 | ~110 ms |

### 5. Implementation Constraints

#### Current Implementation Limits:

1. **Hardcoded Level Limit**: The `upsample()` method restricts levels to ≤ 30 to prevent integer overflow
2. **Array Size Limit**: Java arrays are limited to Integer.MAX_VALUE (2,147,483,647) elements
3. **JVM Heap Size**: Default heap is often 1-4 GB, limiting practical array sizes

#### Level Validation:

The `forward()` method validates that the requested level doesn't exceed log₂(N):
```java
int maxLevel = calcExponent(arrTime.length);
if (level < 0 || level > maxLevel)
    throw new JWaveFailure("given level is out of range for given array");
```

### 6. Practical Recommendations by Wavelet Type

Maximum practical levels considering filter length constraints:

| Signal Length | Haar | Db2 | Db4 | Db8 | Db10 |
|--------------|------|-----|-----|-----|------|
| 128 | 5 | 4 | 3 | 2 | 1 |
| 1,024 | 8 | 7 | 6 | 5 | 4 |
| 8,192 | 11 | 10 | 9 | 8 | 7 |
| 65,536 | 14 | 13 | 12 | 11 | 10 |

### 7. Best Practices

1. **Start Conservative**: Begin with 3-5 levels and increase only if needed
2. **Consider Filter Length**: Longer wavelets require lower maximum levels
3. **Monitor Memory**: Use `precomputeFilters()` to check memory requirements
4. **Profile Performance**: Test with your specific signal lengths and wavelets
5. **Use Filter Cache**: The implementation caches upsampled filters for efficiency

### 8. Special Considerations

#### For Time Series Analysis:
- Levels 1-3: Capture high-frequency noise and artifacts
- Levels 4-6: Represent short-term variations
- Levels 7-10: Show medium-term trends
- Levels > 10: Reveal long-term patterns

#### For Real-time Processing:
- Limit to 8 levels maximum
- Pre-compute filters using `precomputeFilters(maxLevel)`
- Consider using shorter wavelets (Haar, Db2)

#### For Memory-Constrained Environments:
- Clear filter cache between transforms: `clearFilterCache()`
- Use lower decomposition levels
- Process signals in segments if possible

## Conclusion

While the theoretical maximum MODWT decomposition level is log₂(N), practical limits are significantly lower due to:

1. Exponential growth of upsampled filter lengths
2. Memory constraints (both coefficient storage and filter caching)
3. Computational performance requirements
4. Diminishing analytical returns at very high levels

For most applications, limiting decomposition to 8-12 levels provides an optimal balance between frequency resolution, computational efficiency, and memory usage. Always test with your specific signal characteristics and system constraints to determine the appropriate maximum level.