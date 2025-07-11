package jwave.transforms;

import jwave.transforms.wavelets.Wavelet;
import jwave.utils.ArrayBufferPool;
import jwave.datatypes.natives.Complex;
import java.util.Arrays;

/**
 * A pooled version of MODWTTransform that uses buffer pooling to reduce GC pressure.
 * 
 * <p>This implementation extends MODWTTransform and overrides the convolution methods
 * to use pooled arrays instead of allocating new arrays for each operation. This can
 * significantly reduce garbage collection overhead in applications that perform many
 * transforms.</p>
 * 
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * // Use pooled version for better performance
 * MODWTTransform modwt = new PooledMODWTTransform(new Daubechies4());
 * 
 * // Perform many transforms without GC pressure
 * for (int i = 0; i < 10000; i++) {
 *     double[][] coeffs = modwt.forwardMODWT(signal, 5);
 *     // Process coefficients...
 * }
 * 
 * // Clean up thread-local pools when done
 * ArrayBufferPool.remove();
 * }</pre>
 * 
 * @author Stephen Romano
 * @since 2.1.0
 */
public class PooledMODWTTransform extends MODWTTransform {
    
    /**
     * Constructor for pooled MODWT transform.
     * 
     * @param wavelet The wavelet to use for the transform
     */
    public PooledMODWTTransform(Wavelet wavelet) {
        super(wavelet);
        // Replace the standard FFT with a pooled version
        this.fft = new PooledFastFourierTransform();
    }
    
    /**
     * Constructor for pooled MODWT transform with custom FFT threshold.
     * 
     * @param wavelet The wavelet to use for the transform
     * @param fftThreshold Custom threshold for FFT-based convolution
     */
    public PooledMODWTTransform(Wavelet wavelet, int fftThreshold) {
        super(wavelet, fftThreshold);
        // Replace the standard FFT with a pooled version
        this.fft = new PooledFastFourierTransform();
    }
    
    /**
     * Override performConvolution to use pooled arrays based on the method selection.
     */
    @Override
    protected double[] performConvolution(double[] signal, double[] filter, boolean adjoint) {
        boolean useFFT = false;
        
        switch (getConvolutionMethod()) {
            case FFT:
                useFFT = true;
                break;
            case DIRECT:
                useFFT = false;
                break;
            case AUTO:
                // Use parent's threshold logic
                useFFT = (signal.length * filter.length) > super.fftConvolutionThreshold;
                break;
        }
        
        if (useFFT) {
            return adjoint ? circularConvolveFFTAdjointPooled(signal, filter) 
                          : circularConvolveFFTPooled(signal, filter);
        } else {
            return adjoint ? this.circularConvolveAdjointPooled(signal, filter) 
                          : this.circularConvolvePooled(signal, filter);
        }
    }
    
    /**
     * Performs circular convolution using pooled arrays.
     */
    private double[] circularConvolvePooled(double[] signal, double[] filter) {
        int N = signal.length;
        int M = filter.length;
        
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        double[] output = pool.borrowDoubleArray(N);
        
        try {
            // Clear the array in case it has old data
            Arrays.fill(output, 0, N, 0.0);
            
            for (int n = 0; n < N; n++) {
                double sum = 0.0;
                for (int m = 0; m < M; m++) {
                    int signalIndex = Math.floorMod(n - m, N);
                    sum += signal[signalIndex] * filter[m];
                }
                output[n] = sum;
            }
            
            // Return a copy and return the buffer to the pool
            double[] result = Arrays.copyOf(output, N);
            return result;
        } finally {
            pool.returnDoubleArray(output);
        }
    }
    
    /**
     * Performs adjoint circular convolution using pooled arrays.
     */
    private double[] circularConvolveAdjointPooled(double[] signal, double[] filter) {
        int N = signal.length;
        int M = filter.length;
        
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        double[] output = pool.borrowDoubleArray(N);
        
        try {
            // Clear the array in case it has old data
            Arrays.fill(output, 0, N, 0.0);
            
            for (int n = 0; n < N; n++) {
                double sum = 0.0;
                for (int m = 0; m < M; m++) {
                    int signalIndex = Math.floorMod(n + m, N);
                    sum += signal[signalIndex] * filter[m];
                }
                output[n] = sum;
            }
            
            // Return a copy and return the buffer to the pool
            double[] result = Arrays.copyOf(output, N);
            return result;
        } finally {
            pool.returnDoubleArray(output);
        }
    }
    
    /**
     * Performs circular convolution using FFT with pooled arrays.
     */
    private double[] circularConvolveFFTPooled(double[] signal, double[] filter) {
        int N = signal.length;
        
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        // Borrow arrays from pool
        double[] paddedFilter = pool.borrowDoubleArray(N);
        Complex[] signalComplex = pool.borrowComplexArray(N);
        Complex[] filterComplex = pool.borrowComplexArray(N);
        Complex[] productFFT = pool.borrowComplexArray(N);
        double[] output = pool.borrowDoubleArray(N);
        
        try {
            // Clear arrays
            Arrays.fill(paddedFilter, 0, N, 0.0);
            
            // Wrap filter to signal length
            for (int i = 0; i < filter.length; i++) {
                paddedFilter[i % N] += filter[i];
            }
            
            // Convert to complex arrays
            for (int i = 0; i < N; i++) {
                signalComplex[i].setReal(signal[i]);
                signalComplex[i].setImag(0);
                filterComplex[i].setReal(paddedFilter[i]);
                filterComplex[i].setImag(0);
            }
            
            // Compute FFTs
            Complex[] signalFFT = fft.forward(signalComplex);
            Complex[] filterFFT = fft.forward(filterComplex);
            
            // Pointwise multiplication in frequency domain
            for (int i = 0; i < N; i++) {
                productFFT[i] = signalFFT[i].mul(filterFFT[i]);
            }
            
            // Inverse FFT - reuse productFFT array for result
            // Note: FFT.reverse() returns a new array internally. By using PooledFastFourierTransform
            // in the constructor, we minimize allocations in the conversion between double[] and Complex[].
            // The core FFT algorithm allocations are harder to eliminate without modifying the FFT itself.
            Complex[] inversed = fft.reverse(productFFT);
            
            // Extract real part directly
            for (int i = 0; i < N; i++) {
                output[i] = inversed[i].getReal();
            }
            
            // Return a copy
            return Arrays.copyOf(output, N);
            
        } finally {
            // Return all borrowed arrays to the pool
            pool.returnDoubleArray(paddedFilter);
            pool.returnComplexArray(signalComplex);
            pool.returnComplexArray(filterComplex);
            pool.returnComplexArray(productFFT);
            pool.returnDoubleArray(output);
        }
    }
    
    /**
     * Performs the adjoint of circular convolution using FFT with pooled arrays.
     */
    private double[] circularConvolveFFTAdjointPooled(double[] signal, double[] filter) {
        int N = signal.length;
        
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        // Borrow arrays from pool
        double[] paddedFilter = pool.borrowDoubleArray(N);
        Complex[] signalComplex = pool.borrowComplexArray(N);
        Complex[] filterComplex = pool.borrowComplexArray(N);
        Complex[] productFFT = pool.borrowComplexArray(N);
        double[] output = pool.borrowDoubleArray(N);
        
        try {
            // Clear arrays
            Arrays.fill(paddedFilter, 0, N, 0.0);
            
            // Wrap filter to signal length
            for (int i = 0; i < filter.length; i++) {
                paddedFilter[i % N] += filter[i];
            }
            
            // Convert to complex arrays
            for (int i = 0; i < N; i++) {
                signalComplex[i].setReal(signal[i]);
                signalComplex[i].setImag(0);
                filterComplex[i].setReal(paddedFilter[i]);
                filterComplex[i].setImag(0);
            }
            
            // Compute FFTs
            Complex[] signalFFT = fft.forward(signalComplex);
            Complex[] filterFFT = fft.forward(filterComplex);
            
            // For the adjoint operation, conjugate the filter FFT
            for (int i = 0; i < N; i++) {
                productFFT[i] = signalFFT[i].mul(filterFFT[i].conjugate());
            }
            
            // Inverse FFT - reuse productFFT array for result
            // Note: FFT.reverse() returns a new array internally. By using PooledFastFourierTransform
            // in the constructor, we minimize allocations in the conversion between double[] and Complex[].
            // The core FFT algorithm allocations are harder to eliminate without modifying the FFT itself.
            Complex[] inversed = fft.reverse(productFFT);
            
            // Extract real part directly
            for (int i = 0; i < N; i++) {
                output[i] = inversed[i].getReal();
            }
            
            // Return a copy
            return Arrays.copyOf(output, N);
            
        } finally {
            // Return all borrowed arrays to the pool
            pool.returnDoubleArray(paddedFilter);
            pool.returnComplexArray(signalComplex);
            pool.returnComplexArray(filterComplex);
            pool.returnComplexArray(productFFT);
            pool.returnDoubleArray(output);
        }
    }
    
    /**
     * Performs inverse MODWT with pooled arrays for vNext allocation.
     */
    @Override
    public double[] inverseMODWT(double[][] coefficients) {
        if (coefficients == null || coefficients.length == 0) {
            return new double[0];
        }

        int maxLevel = coefficients.length - 1;
        if (maxLevel <= 0) {
            return new double[0];
        }

        int N = coefficients[0].length;
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        // Initialize cache if needed
        initializeFilterCache();

        double[] vCurrent = Arrays.copyOf(coefficients[maxLevel], N);
        double[] vNext = pool.borrowDoubleArray(N);

        try {
            for (int j = maxLevel; j >= 1; j--) {
                // Clear vNext
                Arrays.fill(vNext, 0, N, 0.0);
                
                // Use cached filters
                double[] gUpsampled = getCachedGFilter(j);
                double[] hUpsampled = getCachedHFilter(j);

                double[] wCurrent = coefficients[j - 1];

                // Use the adjoint convolution for the inverse transform
                double[] vFromApprox = performConvolution(vCurrent, gUpsampled, true);
                double[] vFromDetail = performConvolution(wCurrent, hUpsampled, true);

                for (int i = 0; i < N; i++) {
                    vNext[i] = vFromApprox[i] + vFromDetail[i];
                }

                // Swap arrays to avoid allocation
                double[] temp = vCurrent;
                vCurrent = vNext;
                vNext = temp;
            }

            // Return a copy of the final result
            return Arrays.copyOf(vCurrent, N);
            
        } finally {
            pool.returnDoubleArray(vNext);
            if (vCurrent != coefficients[maxLevel]) {
                pool.returnDoubleArray(vCurrent);
            }
        }
    }
}