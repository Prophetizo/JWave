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
            throw new JWaveFailure("WaveletPacketTransform#forward - array length is not 2^p");
            
        int maxLevel = calcExponent(arrTime.length);
        if (level <= 0 || level > maxLevel)
            throw new JWaveFailure("WaveletPacketTransform#forward - invalid level");
            
        int length = arrTime.length;
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        // Allocate result array
        double[] arrHilb = Arrays.copyOf(arrTime, length);
        
        // Pre-allocate the largest buffer we'll need for the loops
        int maxH = length;
        double[] iBufPool = pool.borrowDoubleArray(maxH);
        
        try {
            int k = 0;
            int h = length;
            int steps = 1;
            
            for (int l = 0; l < level; l++) {
                h = h >> 1;
                
                for (int s = 0; s < steps; s++) {
                    // Reuse the pooled buffer, just the portion we need
                    double[] iBuf = iBufPool; // Use first h elements
                    Arrays.fill(iBuf, 0, h, 0.0);
                    
                    for (int i = 0; i < h; i++)
                        iBuf[i] = arrHilb[k + i];
                        
                    double[] oBuf = _wavelet.forward(iBuf, h);
                    
                    for (int i = 0; i < h; i++)
                        arrHilb[k + i] = oBuf[i];
                        
                    k += h;
                }
                
                steps = steps << 1;
            }
            
            return arrHilb;
            
        } finally {
            pool.returnDoubleArray(iBufPool);
        }
    }
    
    @Override
    public double[] reverse(double[] arrHilb, int level) throws JWaveException {
        if (!isBinary(arrHilb.length))
            throw new JWaveFailure("WaveletPacketTransform#reverse - array length is not 2^p");
            
        int maxLevel = calcExponent(arrHilb.length);
        if (level < 0 || level > maxLevel)
            throw new JWaveFailure("WaveletPacketTransform#reverse - invalid level");
            
        int length = arrHilb.length;
        ArrayBufferPool pool = ArrayBufferPool.getInstance();
        
        double[] arrTime = Arrays.copyOf(arrHilb, length);
        
        // Pre-allocate the largest buffer we'll need
        int maxH = length >> (level - 1);
        double[] iBufPool = pool.borrowDoubleArray(maxH);
        
        try {
            int steps = calcExponent(length) - level;
            steps = (int)(Math.pow(2.0, (double)steps));
            
            int h = length;
            for (int l = 0; l < level; l++)
                h = h >> 1;
                
            int k = 0;
            
            for (int l = level; l > 0; l--) {
                for (int s = 0; s < steps; s++) {
                    // Reuse the pooled buffer
                    double[] iBuf = iBufPool;
                    Arrays.fill(iBuf, 0, h, 0.0);
                    
                    for (int i = 0; i < h; i++)
                        iBuf[i] = arrTime[k + i];
                        
                    double[] oBuf = _wavelet.reverse(iBuf, h);
                    
                    for (int i = 0; i < h; i++)
                        arrTime[k + i] = oBuf[i];
                        
                    k += h;
                }
                
                h = h << 1;
                steps = steps >> 1;
            }
            
            return arrTime;
            
        } finally {
            pool.returnDoubleArray(iBufPool);
        }
    }
}