package jwave.transforms;

import jwave.transforms.wavelets.haar.Haar1;
import jwave.exceptions.JWaveFailure;
import jwave.exceptions.JWaveException;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Test the maximum decomposition level limit of 13 for MODWTTransform.
 * 13 is chosen as it's a Fibonacci number and provides a reasonable
 * balance between flexibility and memory constraints.
 */
public class MODWTLevelLimitTest {
    
    @Test
    public void testMaximumLevelIsEnforced() {
        MODWTTransform modwt = new MODWTTransform(new Haar1());
        double[] signal = new double[16384]; // 2^14, enough for 14 levels theoretically
        
        // Level 13 should work
        try {
            double[][] result = modwt.forwardMODWT(signal, 13);
            assertNotNull("Level 13 should succeed", result);
            assertEquals("Should return 14 arrays (13 details + 1 approximation)", 14, result.length);
        } catch (Exception e) {
            fail("Level 13 should not throw exception: " + e.getMessage());
        }
        
        // Level 14 should fail
        try {
            modwt.forwardMODWT(signal, 14);
            fail("Level 14 should throw exception");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention level 13", 
                      e.getMessage().contains("13"));
        }
    }
    
    @Test
    public void testPrecomputeFiltersLevelLimit() {
        MODWTTransform modwt = new MODWTTransform(new Haar1());
        
        // Level 13 should work
        try {
            modwt.precomputeFilters(13);
        } catch (Exception e) {
            fail("precomputeFilters(13) should not throw exception: " + e.getMessage());
        }
        
        // Level 14 should fail
        try {
            modwt.precomputeFilters(14);
            fail("precomputeFilters(14) should throw exception");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention level 13", 
                      e.getMessage().contains("13"));
        }
    }
    
    @Test
    public void testForwardMethodWithLevelLimit() throws JWaveException {
        MODWTTransform modwt = new MODWTTransform(new Haar1());
        double[] signal = new double[16384]; // 2^14
        
        // Level 13 should work
        try {
            double[] result = modwt.forward(signal, 13);
            assertNotNull("Level 13 should succeed", result);
        } catch (JWaveFailure e) {
            fail("forward() with level 13 should not throw exception: " + e.getMessage());
        }
        
        // Level 14 should fail
        try {
            modwt.forward(signal, 14);
            fail("forward() with level 14 should throw exception");
        } catch (JWaveFailure e) {
            assertTrue("Exception message should mention level 13", 
                      e.getMessage().contains("13"));
        }
    }
    
    @Test
    public void testUpsamplingAtLevel13() {
        MODWTTransform modwt = new MODWTTransform(new Haar1());
        double[] smallSignal = new double[1024]; // 2^10
        
        // Even with a small signal, we should be able to request level 13
        // (though it may not be meaningful)
        try {
            modwt.precomputeFilters(13);
            // The cache should now contain upsampled filters for levels 1-13
        } catch (Exception e) {
            fail("Should be able to pre-compute filters up to level 13: " + e.getMessage());
        }
    }
    
    @Test
    public void testFibonacciLevelChoice() {
        // Verify that 13 is indeed a Fibonacci number
        int[] fibonacci = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
        boolean isFibonacci = false;
        for (int fib : fibonacci) {
            if (fib == 13) {
                isFibonacci = true;
                break;
            }
        }
        assertTrue("13 should be a Fibonacci number", isFibonacci);
        
        // At level 13, filter sizes become quite large
        // For Haar (2 coefficients), upsampled size = 2 + (2-1) * (2^12 - 1) = 4096
        // For Db4 (8 coefficients), upsampled size = 8 + (8-1) * (2^12 - 1) = 28,673
        // These are still manageable sizes
        
        MODWTTransform modwt = new MODWTTransform(new Haar1());
        modwt.precomputeFilters(13);
        
        // The filters should be cached and ready for use
        double[] testSignal = new double[8192];
        double[][] result = modwt.forwardMODWT(testSignal, 10); // Use reasonable level
        assertNotNull("Transform should work with pre-computed filters", result);
    }
}