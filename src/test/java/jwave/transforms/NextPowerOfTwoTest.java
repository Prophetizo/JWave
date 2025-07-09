/**
 * JWave is distributed under the MIT License (MIT); this file is part of.
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
import jwave.utils.MathUtils;

/**
 * Test to verify nextPowerOfTwo implementations.
 *
 * @author Stephen Romano
 * @date 07.01.2025
 */
public class NextPowerOfTwoTest {
    
    @Test
    public void testNextPowerOfTwo() {
        // Test the bit-twiddling approach from MathUtils
        assertEquals(1, MathUtils.nextPowerOfTwo(0));
        assertEquals(1, MathUtils.nextPowerOfTwo(1));
        assertEquals(2, MathUtils.nextPowerOfTwo(2));
        assertEquals(4, MathUtils.nextPowerOfTwo(3));
        assertEquals(4, MathUtils.nextPowerOfTwo(4));
        assertEquals(8, MathUtils.nextPowerOfTwo(5));
        assertEquals(8, MathUtils.nextPowerOfTwo(6));
        assertEquals(8, MathUtils.nextPowerOfTwo(7));
        assertEquals(8, MathUtils.nextPowerOfTwo(8));
        assertEquals(16, MathUtils.nextPowerOfTwo(9));
        assertEquals(16, MathUtils.nextPowerOfTwo(15));
        assertEquals(16, MathUtils.nextPowerOfTwo(16));
        assertEquals(32, MathUtils.nextPowerOfTwo(17));
        assertEquals(64, MathUtils.nextPowerOfTwo(63));
        assertEquals(64, MathUtils.nextPowerOfTwo(64));
        assertEquals(128, MathUtils.nextPowerOfTwo(65));
        assertEquals(256, MathUtils.nextPowerOfTwo(255));
        assertEquals(256, MathUtils.nextPowerOfTwo(256));
        assertEquals(512, MathUtils.nextPowerOfTwo(257));
        assertEquals(1024, MathUtils.nextPowerOfTwo(1000));
        assertEquals(2048, MathUtils.nextPowerOfTwo(2000));
        assertEquals(4096, MathUtils.nextPowerOfTwo(3000));
        assertEquals(4096, MathUtils.nextPowerOfTwo(4096));
        assertEquals(8192, MathUtils.nextPowerOfTwo(4097));
        assertEquals(65536, MathUtils.nextPowerOfTwo(50000));
        assertEquals(1048576, MathUtils.nextPowerOfTwo(1000000));
    }
    
    @Test
    public void testIsPowerOfTwo() {
        // Test isPowerOfTwo from MathUtils
        assertFalse(MathUtils.isPowerOfTwo(0));
        assertTrue(MathUtils.isPowerOfTwo(1));
        assertTrue(MathUtils.isPowerOfTwo(2));
        assertFalse(MathUtils.isPowerOfTwo(3));
        assertTrue(MathUtils.isPowerOfTwo(4));
        assertFalse(MathUtils.isPowerOfTwo(5));
        assertFalse(MathUtils.isPowerOfTwo(6));
        assertFalse(MathUtils.isPowerOfTwo(7));
        assertTrue(MathUtils.isPowerOfTwo(8));
        assertFalse(MathUtils.isPowerOfTwo(9));
        assertTrue(MathUtils.isPowerOfTwo(16));
        assertTrue(MathUtils.isPowerOfTwo(32));
        assertTrue(MathUtils.isPowerOfTwo(64));
        assertTrue(MathUtils.isPowerOfTwo(128));
        assertTrue(MathUtils.isPowerOfTwo(256));
        assertTrue(MathUtils.isPowerOfTwo(512));
        assertTrue(MathUtils.isPowerOfTwo(1024));
        assertFalse(MathUtils.isPowerOfTwo(1000));
        assertFalse(MathUtils.isPowerOfTwo(-1));
        assertFalse(MathUtils.isPowerOfTwo(-8));
    }
    
    @Test
    public void compareBitTwiddlingVsFloatingPoint() {
        System.out.println("Comparing bit-twiddling vs floating-point approaches:");
        
        // Test a range of values
        int[] testValues = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                           63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
                           1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
                           8191, 8192, 8193, 16383, 16384, 16385, 32767, 32768, 32769,
                           65535, 65536, 65537, 131071, 131072, 131073,
                           1000000, 10000000};
        
        boolean allMatch = true;
        for (int n : testValues) {
            int bitResult = MathUtils.nextPowerOfTwo(n);
            int floatResult = nextPowerOfTwoFloat(n);
            
            if (bitResult != floatResult) {
                System.out.printf("Mismatch at n=%d: bit=%d, float=%d%n", 
                                n, bitResult, floatResult);
                allMatch = false;
            }
        }
        
        assertTrue("Mismatch detected between bit-twiddling and floating-point implementations!", allMatch);
        
        if (allMatch) {
            System.out.println("All values match between bit-twiddling and floating-point!");
        }
        
        // Performance comparison
        long startBit = System.nanoTime();
        for (int i = 0; i < 1000000; i++) {
            MathUtils.nextPowerOfTwo(i % 100000);
        }
        long timeBit = System.nanoTime() - startBit;
        
        long startFloat = System.nanoTime();
        for (int i = 0; i < 1000000; i++) {
            nextPowerOfTwoFloat(i % 100000);
        }
        long timeFloat = System.nanoTime() - startFloat;
        
        System.out.printf("%nPerformance comparison (1M operations):%n");
        System.out.printf("Bit-twiddling: %.2f ms%n", timeBit / 1_000_000.0);
        System.out.printf("Floating-point: %.2f ms%n", timeFloat / 1_000_000.0);
        System.out.printf("Speedup: %.2fx%n", (double)timeFloat / timeBit);
    }
    
    private int nextPowerOfTwoFloat(int n) {
        if (n <= 1) return 1;
        return (int) Math.pow(2, Math.ceil(Math.log(n) / Math.log(2)));
    }
}