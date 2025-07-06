package jwave.transforms;

import jwave.transforms.wavelets.Wavelet;
import jwave.exceptions.JWaveException;
import jwave.exceptions.JWaveFailure;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An implementation of the Maximal Overlap Discrete Wavelet Transform (MODWT)
 * and its inverse, designed to integrate with the JWave library structure.
 * 
 * <h2>Overview</h2>
 * The MODWT is a shift-invariant, redundant wavelet transform that addresses
 * limitations of the standard Discrete Wavelet Transform (DWT):
 * <ul>
 *   <li>Translation invariance: shifting the input signal results in a shifted output</li>
 *   <li>No downsampling: all levels have the same length as the input signal</li>
 *   <li>Works with any signal length (not restricted to powers of 2)</li>
 *   <li>Increased redundancy provides better analysis capabilities</li>
 * </ul>
 * 
 * <h2>Mathematical Foundation</h2>
 * The MODWT modifies the DWT by:
 * <ol>
 *   <li>Rescaling the wavelet and scaling filters by 1/√2</li>
 *   <li>Using circular convolution instead of downsampling</li>
 *   <li>Upsampling filters at each level j by inserting 2^(j-1)-1 zeros</li>
 * </ol>
 * 
 * For level j, the MODWT filters are:
 * <pre>
 *   h̃_j,l = h_l / 2^(j/2)  (wavelet filter)
 *   g̃_j,l = g_l / 2^(j/2)  (scaling filter)
 * </pre>
 * 
 * <h2>Algorithm</h2>
 * Forward MODWT for J levels:
 * <pre>
 *   V_0 = X (input signal)
 *   For j = 1 to J:
 *     W_j = h̃_j ⊛ V_{j-1}  (detail coefficients)
 *     V_j = g̃_j ⊛ V_{j-1}  (approximation coefficients)
 * </pre>
 * 
 * Inverse MODWT:
 * <pre>
 *   Ṽ_J = V_J
 *   For j = J down to 1:
 *     Ṽ_{j-1} = g̃_j* ⊛ Ṽ_j + h̃_j* ⊛ W_j
 * </pre>
 * Where ⊛ denotes circular convolution and * denotes the adjoint operation.
 * 
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create MODWT instance with Daubechies-4 wavelet
 * MODWTTransform modwt = new MODWTTransform(new Daubechies4());
 * 
 * // Example 1: Decompose a signal to 3 levels
 * double[] signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
 * double[][] coeffs = modwt.forwardMODWT(signal, 3);
 * // coeffs[0] = W_1 (level 1 details)
 * // coeffs[1] = W_2 (level 2 details)
 * // coeffs[2] = W_3 (level 3 details)
 * // coeffs[3] = V_3 (level 3 approximation)
 * 
 * // Example 2: Using the 1D interface
 * double[] flatCoeffs = modwt.forward(signal, 2);
 * // Returns flattened array: [W_1[0..7], W_2[0..7], V_2[0..7]]
 * double[] reconstructed = modwt.reverse(flatCoeffs, 2);
 * 
 * // Example 3: Multi-resolution analysis
 * int maxLevel = 4;
 * double[][] mra = modwt.forwardMODWT(signal, maxLevel);
 * 
 * // Denoise by thresholding level 1 details
 * for (int i = 0; i < mra[0].length; i++) {
 *     if (Math.abs(mra[0][i]) < threshold) {
 *         mra[0][i] = 0.0;
 *     }
 * }
 * double[] denoised = modwt.inverseMODWT(mra);
 * }</pre>
 * 
 * <h2>References</h2>
 * <ul>
 *   <li>Percival, D. B., & Walden, A. T. (2000). Wavelet Methods for Time Series Analysis. 
 *       Cambridge University Press. ISBN: 0-521-68508-7</li>
 *   <li>Cornish, C. R., Bretherton, C. S., & Percival, D. B. (2006). Maximal overlap 
 *       wavelet statistical analysis with application to atmospheric turbulence. 
 *       Boundary-Layer Meteorology, 119(2), 339-374.</li>
 *   <li>Quilty, J., & Adamowski, J. (2018). Addressing the incorrect usage of wavelet-based 
 *       hydrological and water resources forecasting models for real-world applications with 
 *       best practices and a new forecasting framework. Journal of Hydrology, 563, 336-353.</li>
 * </ul>
 * 
 * @author Stephen Romano
 * @see jwave.transforms.wavelets.Wavelet
 */
public class MODWTTransform extends WaveletTransform {

    // Cache for upsampled filters, keyed by level
    private transient Map<Integer, double[]> gFilterCache;
    private transient Map<Integer, double[]> hFilterCache;
    
    // Base MODWT filters (computed once from wavelet)
    private transient double[] g_modwt_base;
    private transient double[] h_modwt_base;
    
    // Flag to track if cache is valid (volatile for thread visibility)
    private transient volatile boolean cacheInitialized = false;

    /**
     * Constructor for the MODWTTransform.
     * 
     * @param wavelet The mother wavelet to use for the transform. Common choices include:
     *                - Haar1: Simple, good for piecewise constant signals
     *                - Daubechies4: Smooth, good for general purpose analysis
     *                - Symlet8: Nearly symmetric, good for signal processing
     */
    public MODWTTransform(Wavelet wavelet) {
        super(wavelet);
    }

    /**
     * Performs a full forward Maximal Overlap Discrete Wavelet Transform.
     * 
     * <p>This is the primary method for MODWT decomposition. Unlike the DWT, the MODWT
     * preserves the length of the signal at each decomposition level, making it ideal
     * for time series analysis where temporal alignment is important.</p>
     * 
     * <p><b>Mathematical Description:</b><br>
     * At each level j, the transform computes:
     * <ul>
     *   <li>W_j = h̃_j ⊛ V_{j-1} (wavelet/detail coefficients)</li>
     *   <li>V_j = g̃_j ⊛ V_{j-1} (scaling/approximation coefficients)</li>
     * </ul>
     * where h̃_j and g̃_j are the upsampled and rescaled filters.</p>
     * 
     * @param data      The input time series data of any length (not restricted to 2^n)
     * @param maxLevel  The maximum level of decomposition (1 ≤ maxLevel ≤ log2(N))
     * @return A 2D array where:
     *         - coeffs[0] through coeffs[maxLevel-1] contain detail coefficients W_j
     *         - coeffs[maxLevel] contains approximation coefficients V_maxLevel
     *         - Each row has the same length as the input signal
     * 
     * @example
     * <pre>{@code
     * double[] ecgSignal = loadECGData();
     * MODWTTransform modwt = new MODWTTransform(new Daubechies4());
     * double[][] coeffs = modwt.forwardMODWT(ecgSignal, 5);
     * // Analyze heart rate variability at different scales
     * double[] scale1Details = coeffs[0]; // 2-4 sample periods
     * double[] scale2Details = coeffs[1]; // 4-8 sample periods
     * }</pre>
     */
    public double[][] forwardMODWT(double[] data, int maxLevel) {
        int N = data.length;
        
        // Initialize cache if needed
        initializeFilterCache();

        double[][] modwtCoeffs = new double[maxLevel + 1][N];
        double[] vCurrent = Arrays.copyOf(data, N);

        for (int j = 1; j <= maxLevel; j++) {
            // Use cached filters instead of creating new ones
            double[] gUpsampled = getCachedGFilter(j);
            double[] hUpsampled = getCachedHFilter(j);

            double[] wNext = circularConvolve(vCurrent, hUpsampled);
            double[] vNext = circularConvolve(vCurrent, gUpsampled);

            modwtCoeffs[j - 1] = wNext;
            vCurrent = vNext;

            if (j == maxLevel) {
                modwtCoeffs[j] = vNext;
            }
        }
        return modwtCoeffs;
    }

    /**
     * Performs the inverse Maximal Overlap Discrete Wavelet Transform (iMODWT).
     * 
     * <p>Reconstructs the original signal from MODWT coefficients using the
     * reconstruction formula:</p>
     * <pre>
     * X = Σ(j=1 to J) D_j + A_J
     * </pre>
     * where D_j are the detail components and A_J is the approximation at level J.
     * 
     * <p><b>Perfect Reconstruction Property:</b><br>
     * The MODWT satisfies the perfect reconstruction property, meaning
     * inverseMODWT(forwardMODWT(X)) = X (within numerical precision).</p>
     * 
     * @param coefficients A 2D array of MODWT coefficients as returned by forwardMODWT
     * @return The reconstructed time-domain signal with the same length as the original
     * 
     * @example
     * <pre>{@code
     * // Perform multi-resolution analysis
     * double[][] coeffs = modwt.forwardMODWT(signal, 4);
     * 
     * // Remove high-frequency noise (zero out level 1 details)
     * coeffs[0] = new double[coeffs[0].length]; // zeros
     * 
     * // Reconstruct denoised signal
     * double[] denoised = modwt.inverseMODWT(coeffs);
     * }</pre>
     */
    public double[] inverseMODWT(double[][] coefficients) {
        if (coefficients == null || coefficients.length == 0) {
            return new double[0];
        }

        int maxLevel = coefficients.length - 1;
        if (maxLevel < 0) return new double[0];

        int N = coefficients[0].length;
        
        // Initialize cache if needed
        initializeFilterCache();

        double[] vCurrent = Arrays.copyOf(coefficients[maxLevel], N);

        for (int j = maxLevel; j >= 1; j--) {
            // Use cached filters instead of creating new ones
            double[] gUpsampled = getCachedGFilter(j);
            double[] hUpsampled = getCachedHFilter(j);

            double[] wCurrent = coefficients[j - 1];

            // Use the adjoint convolution for the inverse transform.
            double[] vFromApprox = circularConvolveAdjoint(vCurrent, gUpsampled);
            double[] vFromDetail = circularConvolveAdjoint(wCurrent, hUpsampled);

            double[] vNext = new double[N];
            for (int i = 0; i < N; i++) {
                vNext[i] = vFromApprox[i] + vFromDetail[i];
            }

            vCurrent = vNext;
        }

        return vCurrent;
    }

    /**
     * Performs a forward MODWT to a specified decomposition level.
     * 
     * <p>This method allows control over the decomposition depth, useful when
     * analyzing specific frequency bands or limiting computational cost.</p>
     * 
     * @param arrTime The input signal (must be power of 2 length)
     * @param level The desired decomposition level (1 ≤ level ≤ log2(N))
     * @return Flattened array containing MODWT coefficients up to the specified level
     * @throws JWaveException if input is invalid or level is out of range
     */
    @Override
    public double[] forward(double[] arrTime, int level) throws JWaveException {
        if (arrTime == null || arrTime.length == 0) return new double[0];
        
        if (!isBinary(arrTime.length))
            throw new JWaveFailure("MODWTTransform#forward - " +
                "given array length is not 2^p | p E N ... = 1, 2, 4, 8, 16, 32, .. ");
        
        int maxLevel = calcExponent(arrTime.length);
        if (level < 0 || level > maxLevel)
            throw new JWaveFailure("MODWTTransform#forward - " +
                "given level is out of range for given array");
        
        // Perform MODWT decomposition to specified level
        double[][] coeffs2D = forwardMODWT(arrTime, level);
        
        // Flatten the 2D coefficient array into 1D
        int N = arrTime.length;
        double[] flatCoeffs = new double[N * (level + 1)];
        
        for (int lev = 0; lev <= level; lev++) {
            System.arraycopy(coeffs2D[lev], 0, flatCoeffs, lev * N, N);
        }
        
        return flatCoeffs;
    }
    
    @Override
    public double[] reverse(double[] arrHilb, int level) throws JWaveException {
        if (arrHilb == null || arrHilb.length == 0) return new double[0];
        
        // For MODWT, we need the full coefficient set to reconstruct
        // The level parameter indicates how many levels were used in decomposition
        int N = arrHilb.length / (level + 1);
        
        if (!isBinary(N))
            throw new JWaveFailure("MODWTTransform#reverse - " +
                "Invalid coefficient array for given level");
        
        if (arrHilb.length != N * (level + 1))
            throw new JWaveFailure("MODWTTransform#reverse - " +
                "Coefficient array length does not match expected size for given level");
        
        // Unflatten the 1D array back to 2D structure
        double[][] coeffs2D = new double[level + 1][N];
        for (int lev = 0; lev <= level; lev++) {
            System.arraycopy(arrHilb, lev * N, coeffs2D[lev], 0, N);
        }
        
        // Perform inverse MODWT
        return inverseMODWT(coeffs2D);
    }

    // --- Helper and Overridden Methods ---
    
    /**
     * Initializes the filter cache if not already initialized.
     * Computes the base MODWT filters from the wavelet coefficients.
     * Thread-safe through double-checked locking pattern.
     */
    private void initializeFilterCache() {
        if (!cacheInitialized || g_modwt_base == null) {
            synchronized (this) {
                // Double-check inside synchronized block
                if (!cacheInitialized || g_modwt_base == null) {
                    // Compute base MODWT filters
                    double[] g_dwt = Arrays.copyOf(_wavelet.getScalingDeComposition(), 
                                                   _wavelet.getScalingDeComposition().length);
                    double[] h_dwt = Arrays.copyOf(_wavelet.getWaveletDeComposition(), 
                                                   _wavelet.getWaveletDeComposition().length);
                    normalize(g_dwt);
                    normalize(h_dwt);
                    
                    double scaleFactor = Math.sqrt(2.0);
                    g_modwt_base = new double[g_dwt.length];
                    h_modwt_base = new double[h_dwt.length];
                    for (int i = 0; i < g_dwt.length; i++) {
                        g_modwt_base[i] = g_dwt[i] / scaleFactor;
                        h_modwt_base[i] = h_dwt[i] / scaleFactor;
                    }
                    
                    // Initialize cache maps with ConcurrentHashMap for thread safety
                    gFilterCache = new ConcurrentHashMap<>();
                    hFilterCache = new ConcurrentHashMap<>();
                    cacheInitialized = true;
                }
            }
        }
    }
    
    /**
     * Gets the cached upsampled G filter for the specified level.
     * Creates and caches it if not already present.
     */
    private double[] getCachedGFilter(int level) {
        initializeFilterCache();
        return gFilterCache.computeIfAbsent(level, k -> upsample(g_modwt_base, k));
    }
    
    /**
     * Gets the cached upsampled H filter for the specified level.
     * Creates and caches it if not already present.
     */
    private double[] getCachedHFilter(int level) {
        initializeFilterCache();
        return hFilterCache.computeIfAbsent(level, k -> upsample(h_modwt_base, k));
    }
    
    /**
     * Clears the filter cache. Call this if memory is a concern
     * or before changing wavelets. Thread-safe.
     */
    public void clearFilterCache() {
        synchronized (this) {
            if (gFilterCache != null) gFilterCache.clear();
            if (hFilterCache != null) hFilterCache.clear();
            // Don't set base filters to null - they can be reused
            // Just mark as uninitialized so they'll be recomputed if needed
            cacheInitialized = false;
        }
    }
    
    /**
     * Pre-computes filters for specified levels to avoid
     * computation during time-critical operations.
     */
    public void precomputeFilters(int maxLevel) {
        initializeFilterCache();
        for (int j = 1; j <= maxLevel; j++) {
            getCachedGFilter(j);
            getCachedHFilter(j);
        }
    }

    /**
     * Normalizes a filter to have unit energy (L2 norm = 1).
     * This ensures the transform preserves signal energy.
     */
    private void normalize(double[] filter) {
        double energy = 0.0;
        for (double c : filter) { energy += c * c; }
        double norm = Math.sqrt(energy);
        if (norm > 1e-12) {
            for (int i = 0; i < filter.length; i++) { filter[i] /= norm; }
        }
    }

    /**
     * Upsamples a filter for a specific decomposition level.
     * 
     * <p>At level j, inserts 2^(j-1) - 1 zeros between each filter coefficient.
     * This is a key operation that makes the MODWT shift-invariant.</p>
     * 
     * @param filter The base filter coefficients
     * @param level The decomposition level (1, 2, 3, ...)
     * @return The upsampled filter
     */
    private static double[] upsample(double[] filter, int level) {
        if (level <= 1) return filter;
        if (level > 30) throw new IllegalArgumentException("Level too large for upsampling: " + level);
        int gap = (1 << (level - 1)) - 1;
        int newLength = filter.length + (filter.length - 1) * gap;
        if (newLength < 0 || newLength < filter.length) throw new IllegalArgumentException("Upsampling would result in array too large");

        double[] upsampled = new double[newLength];
        for (int i = 0; i < filter.length; i++) {
            upsampled[i * (gap + 1)] = filter[i];
        }
        return upsampled;
    }

    /**
     * Performs circular convolution between a signal and filter.
     * 
     * <p>Uses periodic boundary conditions, treating the signal as if it
     * wraps around at the boundaries. This preserves the signal length
     * and is essential for the MODWT's shift-invariance property.</p>
     * 
     * @param signal The input signal
     * @param filter The filter to convolve with
     * @return The convolution result with the same length as the signal
     */
    private static double[] circularConvolve(double[] signal, double[] filter) {
        int N = signal.length;
        int M = filter.length;
        double[] output = new double[N];
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int m = 0; m < M; m++) {
                int signalIndex = Math.floorMod(n - m, N);
                sum += signal[signalIndex] * filter[m];
            }
            output[n] = sum;
        }
        return output;
    }

    /**
     * Performs the adjoint (transpose) of circular convolution.
     * 
     * <p>This operation is crucial for the inverse MODWT. If H is the
     * convolution matrix, this computes H^T * signal. The adjoint
     * operation reverses the time-reversal in standard convolution.</p>
     * 
     * @param signal The input signal
     * @param filter The filter for adjoint convolution
     * @return The adjoint convolution result
     */
    private static double[] circularConvolveAdjoint(double[] signal, double[] filter) {
        int N = signal.length;
        int M = filter.length;
        double[] output = new double[N];
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int m = 0; m < M; m++) {
                int signalIndex = Math.floorMod(n + m, N);
                sum += signal[signalIndex] * filter[m];
            }
            output[n] = sum;
        }
        return output;
    }

    /**
     * Performs a forward MODWT with automatic level selection.
     * 
     * <p>Computes the maximum possible decomposition level based on the
     * signal length and performs a full decomposition. The output is
     * flattened into a 1D array for compatibility with JWave's interface.</p>
     * 
     * <p><b>Output Structure:</b><br>
     * [W_1[0..N-1], W_2[0..N-1], ..., W_J[0..N-1], V_J[0..N-1]]</p>
     * 
     * @param arrTime The input signal (must be power of 2 length)
     * @return Flattened array containing all MODWT coefficients
     * @throws JWaveException if input is null, empty, or not power of 2
     */
    @Override
    public double[] forward(double[] arrTime) throws JWaveException {
        if (arrTime == null || arrTime.length == 0) return new double[0];
        
        // Calculate maximum decomposition level
        int maxLevel = calcExponent(arrTime.length);
        
        // Perform full MODWT decomposition
        double[][] coeffs2D = forwardMODWT(arrTime, maxLevel);
        
        // Flatten the 2D coefficient array into 1D
        // Structure: [D1, D2, ..., D_maxLevel, A_maxLevel]
        // Each has the same length as the input signal
        int N = arrTime.length;
        double[] flatCoeffs = new double[N * (maxLevel + 1)];
        
        for (int level = 0; level <= maxLevel; level++) {
            System.arraycopy(coeffs2D[level], 0, flatCoeffs, level * N, N);
        }
        
        return flatCoeffs;
    }

    @Override
    public double[] reverse(double[] arrHilb) throws JWaveException {
        if (arrHilb == null || arrHilb.length == 0) return new double[0];
        
        // Determine the signal length and number of levels from the flattened array
        // The flattened array contains (maxLevel + 1) segments of equal length
        // We need to find N such that arrHilb.length = N * (levels + 1)
        int totalLength = arrHilb.length;
        int N = 0;
        int levels = 0;
        
        // Find the signal length by trying different possibilities
        for (int testN = 1; testN <= totalLength; testN++) {
            if (totalLength % testN == 0) {
                int testLevels = (totalLength / testN) - 1;
                if (testLevels >= 0 && isBinary(testN) && testLevels <= calcExponent(testN)) {
                    N = testN;
                    levels = testLevels;
                    break;
                }
            }
        }
        
        if (N == 0) {
            throw new JWaveFailure("MODWTTransform#reverse - " +
                "Invalid flattened coefficient array length. Cannot determine original signal dimensions.");
        }
        
        // Unflatten the 1D array back to 2D structure
        double[][] coeffs2D = new double[levels + 1][N];
        for (int level = 0; level <= levels; level++) {
            System.arraycopy(arrHilb, level * N, coeffs2D[level], 0, N);
        }
        
        // Perform inverse MODWT
        return inverseMODWT(coeffs2D);
    }
}
