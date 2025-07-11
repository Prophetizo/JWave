package jwave.transforms;

import jwave.transforms.wavelets.Wavelet;
import jwave.utils.ArrayBufferPool;
import jwave.exceptions.JWaveException;
import jwave.exceptions.JWaveFailure;
import java.util.Arrays;

/**
 * Pooled version of WaveletPacketTransform that eliminates loop allocations.
 * This is critical for performance as the standard version allocates arrays
 * inside tight nested loops.
 * 
 * @author Stephen Romano
 * @since 2.1.0
 */
public class PooledWaveletPacketTransform extends WaveletPacketTransform {
    
    public PooledWaveletPacketTransform(Wavelet wavelet) {
        super(wavelet);
    }
    
    @Override
    public double[] forward(double[] arrTime, int level) throws JWaveException {
        if (!isBinary(arrTime.length))
            throw new JWaveFailure("PooledWaveletPacketTransform#forward - array length is not 2^p");
            
        int maxLevel = calcExponent(arrTime.length);
        if (level <= 0 || level > maxLevel)
            throw new JWaveFailure("PooledWaveletPacketTransform#forward - invalid level");
            
        int length = arrTime.length;
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        // Allocate result array
        double[] arrHilb = Arrays.copyOf(arrTime, length);
        
        int transformWavelength = _wavelet.getTransformWavelength();
        int h = length;
        int l = 0;
        
        while (h >= transformWavelength && l < level) {
            int g = length / h; // number of packets at this level
            
            // Get buffer for this level's packet size
            double[] iBuf = pool.borrowDoubleArray(h);
            
            try {
                for (int p = 0; p < g; p++) {
                    // Clear the buffer
                    Arrays.fill(iBuf, 0.0);
                    
                    int offset = p * h;
                    for (int i = 0; i < h; i++)
                        iBuf[i] = arrHilb[offset + i];
                        
                    double[] oBuf = _wavelet.forward(iBuf, h);
                    
                    for (int i = 0; i < h; i++)
                        arrHilb[offset + i] = oBuf[i];
                }
            } finally {
                pool.returnDoubleArray(iBuf);
            }
            
            h = h >> 1;
            l++;
        }
        
        return arrHilb;
    }
    
    @Override
    public double[] reverse(double[] arrHilb, int level) throws JWaveException {
        if (!isBinary(arrHilb.length))
            throw new JWaveFailure("PooledWaveletPacketTransform#reverse - array length is not 2^p");
            
        int maxLevel = calcExponent(arrHilb.length);
        if (level < 0 || level > maxLevel)
            throw new JWaveFailure("PooledWaveletPacketTransform#reverse - invalid level");
            
        int length = arrHilb.length;
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        double[] arrTime = Arrays.copyOf(arrHilb, length);
        
        int transformWavelength = _wavelet.getTransformWavelength();
        int h = transformWavelength;
        
        int steps = calcExponent(length);
        for (int l = level; l < steps; l++)
            h = h << 1; // begin reverse transform at certain level
        
        // No pre-allocation of buffer here, will allocate as needed
        
        while (h <= arrTime.length && h >= transformWavelength) {
            int g = length / h; // number of packets at this level
            
            // Get buffer for this level's packet size
            double[] iBuf = pool.borrowDoubleArray(h);
            
            try {
                for (int p = 0; p < g; p++) {
                    // Clear the buffer
                    Arrays.fill(iBuf, 0.0);
                    
                    int offset = p * h;
                    for (int i = 0; i < h; i++)
                        iBuf[i] = arrTime[offset + i];
                        
                    double[] oBuf = _wavelet.reverse(iBuf, h);
                    
                    for (int i = 0; i < h; i++)
                        arrTime[offset + i] = oBuf[i];
                }
            } finally {
                pool.returnDoubleArray(iBuf);
            }
            
            h = h << 1;
        }
        
        return arrTime;
    }
}