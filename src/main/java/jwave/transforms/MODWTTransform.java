package jwave.transforms;

import jwave.transforms.wavelets.Wavelet;

import java.util.Arrays;

/**
 * An implementation of the Maximal Overlap Discrete Wavelet Transform (MODWT)
 * and its inverse, designed to integrate with the JWave library structure.
 */
public class MODWTTransform extends WaveletTransform {

    /**
     * Constructor for the MODWTTransform.
     * @param wavelet The mother wavelet to use for the transform.
     */
    public MODWTTransform(Wavelet wavelet) {
        super(wavelet);
    }

    /**
     * Performs a full forward Maximal Overlap Discrete Wavelet Transform.
     * This is the recommended method to use for this class.
     * @param data      The input time series data.
     * @param maxLevel  The maximum level of decomposition to perform.
     * @return A 2D array where rows represent coefficients for each level.
     * Structure: [D1, D2, ..., D_maxLevel, A_maxLevel]
     */
    public double[][] forwardMODWT(double[] data, int maxLevel) {
        int N = data.length;

        double[] g_dwt = Arrays.copyOf(_wavelet.getScalingDeComposition(), _wavelet.getScalingDeComposition().length);
        double[] h_dwt = Arrays.copyOf(_wavelet.getWaveletDeComposition(), _wavelet.getWaveletDeComposition().length);
        normalize(g_dwt);
        normalize(h_dwt);

        double scaleFactor = Math.sqrt(2.0);
        double[] g_modwt = new double[g_dwt.length];
        double[] h_modwt = new double[h_dwt.length];
        for (int i = 0; i < g_dwt.length; i++) {
            g_modwt[i] = g_dwt[i] / scaleFactor;
            h_modwt[i] = h_dwt[i] / scaleFactor;
        }

        double[][] modwtCoeffs = new double[maxLevel + 1][N];
        double[] vCurrent = Arrays.copyOf(data, N);

        for (int j = 1; j <= maxLevel; j++) {
            double[] gUpsampled = upsample(g_modwt, j);
            double[] hUpsampled = upsample(h_modwt, j);

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
     * @param coefficients A 2D array of MODWT coefficients with structure:
     * [D1, D2, ..., D_maxLevel, A_maxLevel]
     * @return The reconstructed time-domain signal.
     */
    public double[] inverseMODWT(double[][] coefficients) {
        if (coefficients == null || coefficients.length == 0) {
            return new double[0];
        }

        int maxLevel = coefficients.length - 1;
        if (maxLevel < 0) return new double[0];

        int N = coefficients[0].length;

        // *** FINAL REFINEMENT: The inverse transform uses the DECOMPOSITION filters with the adjoint operator. ***
        double[] g_dwt = _wavelet.getScalingDeComposition();
        double[] h_dwt = _wavelet.getWaveletDeComposition();
        normalize(g_dwt);
        normalize(h_dwt);

        double scaleFactor = Math.sqrt(2.0);
        double[] g_modwt = new double[g_dwt.length];
        double[] h_modwt = new double[h_dwt.length];
        for (int i = 0; i < g_dwt.length; i++) {
            g_modwt[i] = g_dwt[i] / scaleFactor;
            h_modwt[i] = h_dwt[i] / scaleFactor;
        }

        double[] vCurrent = Arrays.copyOf(coefficients[maxLevel], N);

        for (int j = maxLevel; j >= 1; j--) {
            double[] gUpsampled = upsample(g_modwt, j);
            double[] hUpsampled = upsample(h_modwt, j);

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

    // --- Helper and Overridden Methods ---

    private void normalize(double[] filter) {
        double energy = 0.0;
        for (double c : filter) { energy += c * c; }
        double norm = Math.sqrt(energy);
        if (norm > 1e-12) {
            for (int i = 0; i < filter.length; i++) { filter[i] /= norm; }
        }
    }

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

    private static double[] circularConvolveAdjoint(double[] signal, double[] filter) {
        // The adjoint (transpose) of the circular convolution operator.
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

    @Override
    public double[] forward(double[] arrTime) {
        if (arrTime == null || arrTime.length == 0) return new double[0];
        double[][] allCoeffs = forwardMODWT(arrTime, 1);
        return allCoeffs.length > 0 ? allCoeffs[0] : new double[0];
    }

    @Override
    public double[] reverse(double[] arrHilb) {
        System.err.println("Reverse MODWT (iMODWT) is not supported for 1D input. Use inverseMODWT(double[][]).");
        return new double[0];
    }
}
