# TODO: MODWT Filter Cache Optimization

## Quick Summary
The MODWTTransform currently creates new upsampled filter arrays for each decomposition level on every transform call. This can be optimized by caching the upsampled filters, providing 15-20% performance improvement for repeated transforms.

## Branch Strategy
1. Create new branch: `feature/modwt-filter-cache`
2. Base it on: `feature/MODWTTransform` (after merge to main)
3. Keep changes isolated to MODWTTransform class

## Key Code Locations
- **File**: `src/main/java/jwave/transforms/MODWTTransform.java`
- **Methods to modify**:
  - `forwardMODWT()` - Lines 166-167
  - `inverseMODWT()` - Lines 211-212
- **Bottleneck**: `upsample()` method called repeatedly

## Quick Implementation
```java
// Add to class fields
private Map<Integer, double[]> gFilterCache = new HashMap<>();
private Map<Integer, double[]> hFilterCache = new HashMap<>();

// Replace in forwardMODWT:
// OLD:
double[] gUpsampled = upsample(g_modwt, j);
double[] hUpsampled = upsample(h_modwt, j);

// NEW:
double[] gUpsampled = gFilterCache.computeIfAbsent(j, 
    k -> upsample(g_modwt, k));
double[] hUpsampled = hFilterCache.computeIfAbsent(j, 
    k -> upsample(h_modwt, k));
```

## Expected Benefits
- **Performance**: 15-20% faster for repeated transforms
- **Memory**: ~50KB overhead (negligible)
- **Use cases**: Batch processing, real-time analysis, interactive applications

## Full Implementation Guide
See: `MODWT_FILTER_CACHE_IMPLEMENTATION_GUIDE.md`

## Time Estimate
3-5 hours including testing and documentation