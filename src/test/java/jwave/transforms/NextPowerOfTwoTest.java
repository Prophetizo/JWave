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

/**
 * Test to verify nextPowerOfTwo implementations.
 *
 * @author Stephen Romano
 * @date 07.01.2025
 */
public class NextPowerOfTwoTest {
    
    @Test
    public void testNextPowerOfTwo() {
        // Test the bit-twiddling approach
        assertEquals(1, nextPowerOfTwo(0));
        assertEquals(1, nextPowerOfTwo(1));
        assertEquals(2, nextPowerOfTwo(2));
        assertEquals(4, nextPowerOfTwo(3));
        assertEquals(4, nextPowerOfTwo(4));
        assertEquals(8, nextPowerOfTwo(5));
        assertEquals(8, nextPowerOfTwo(6));
        assertEquals(8, nextPowerOfTwo(7));
        assertEquals(8, nextPowerOfTwo(8));
        assertEquals(16, nextPowerOfTwo(9));
        assertEquals(16, nextPowerOfTwo(15));
        assertEquals(16, nextPowerOfTwo(16));
        assertEquals(32, nextPowerOfTwo(17));
        assertEquals(64, nextPowerOfTwo(63));
        assertEquals(64, nextPowerOfTwo(64));
        assertEquals(128, nextPowerOfTwo(65));
        assertEquals(256, nextPowerOfTwo(255));
        assertEquals(256, nextPowerOfTwo(256));
        assertEquals(512, nextPowerOfTwo(257));
        assertEquals(1024, nextPowerOfTwo(1000));
        assertEquals(2048, nextPowerOfTwo(2000));
        assertEquals(4096, nextPowerOfTwo(3000));
        assertEquals(4096, nextPowerOfTwo(4096));
        assertEquals(8192, nextPowerOfTwo(4097));
        assertEquals(65536, nextPowerOfTwo(50000));
        assertEquals(1048576, nextPowerOfTwo(1000000));
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
            int bitResult = nextPowerOfTwo(n);
            int floatResult = nextPowerOfTwoFloat(n);
            
            if (bitResult != floatResult) {
                System.out.printf("Mismatch at n=%d: bit=%d, float=%d%n", 
                                n, bitResult, floatResult);
                allMatch = false;
            }
        }
        
        if (allMatch) {
            System.out.println("All values match between bit-twiddling and floating-point!");
        }
        
        // Performance comparison
        long startBit = System.nanoTime();
        for (int i = 0; i < 1000000; i++) {
            nextPowerOfTwo(i % 100000);
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
    
    private int nextPowerOfTwo(int n) {
        if (n <= 1) return 1;
        return 1 << (32 - Integer.numberOfLeadingZeros(n - 1));
    }
    
    private int nextPowerOfTwoFloat(int n) {
        if (n <= 1) return 1;
        return (int) Math.pow(2, Math.ceil(Math.log(n) / Math.log(2)));
    }
}