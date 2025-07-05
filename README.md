# JWave
## Java library keeping orthogonal / orthonormal and bi-orthogonal wavelets

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/graetz23/JWave)
![GitHub Repo stars](https://img.shields.io/github/stars/graetz23/JWave?style=social)
![GitHub forks](https://img.shields.io/github/forks/graetz23/JWave?style=social)

## Introduction

Java implementation of a **Discrete Fourier Transform (DFT)**, a **Fast Wavelet Transform (FWT)**, a **Wavelet Packet Transform (WPT)**, and a **Maximal Overlap Discrete Wavelet Transform (MODWT)** algorithm. All algorithms are available **in 1-D, 2-D, and 3-D** (except MODWT which is currently 1-D). The wavelet transform algorithms are **using** normalized orthogonal or if available **orthonormal** wavelets. The comon **wavelets like Haar, Coiflet, Daubechies, Symlets, and Legendre** are available. Additionally there are also some Bi-Orthogonal and unusal wavelets implemented - in total around 50 wavelets.

The implementation of JWave is based on several software design patterns and - hopefully - appears therefore user-friendly.

## GETTING STARTED

### First steps

Have a look at the little **how to down the page** or try looking at  [the unit tests](https://github.com/graetz23/JWave/tree/master/test/jwave).

### Can I perform odd samplings like, e.g. 127 data points?

The standard FWT and WPT algorithms in JWave only support data sampled by 2^p | p E N; e.g. 2, 4, 8, 16, .. 128, 256, 512, 1024, .. and so on. However, you have two options for handling arbitrary length signals:

1. **AncientEgyptianDecomposition class** - wraps FWT/WPT to handle odd samplings (see examples below)
2. **MODWT (Maximal Overlap Discrete Wavelet Transform)** - natively supports signals of any length without padding or truncation

### Why are the results totally different then expected?!

**Why do the results (hilbert spaces) look totally different to those from my matlab or some other implementation I found on net?!**

In most cases, other libraries construct the orthogonal / orthonormal bases of the wavelet and scaling function in a different way. Especially for those bases of multiple dimension or wavelets of higher dimension, repectively. But does this hurt?

**Totally not!**

The *why* can be found in mathematics. Due to using some *orthogonal* transform (or better, an orthogonal base), it is up to oneself how to *construct* this base (as long it stays orthogonal over all dimensions). Next it is also up to oneself how to apply the sequence of the *transform steps*. Both does not influence any performance of the wavelet transforms! But again why?

The base stays orthogonal, and **one's data is *unit* rotated and mirrowed differently**, which *makes a long story short*.

Additionally the application of the transform - independent of using some different rotating and mirrowing base - is like *dancing some 90's techno*: As long as you do the *same amount* of steps *independently* of the performed sequence, even in 2-D and 3-D dimensions, the *expected magic* wavelets can bring in, will be there, and stays the same! For example, the result in values, and e.g the compression rates will stay exactely the same. Only _all_ intermediate performed results (or intermediate hilbert spaces) will be different, if someone else *dances* differently to otherones.

### HowTo

For a quick test, pull the repository and then: *ant && ant test*. This builds a JWave.jar and the corresponding unit tests. Afterwards *all* units test are executed.

### MODWT - Maximal Overlap Discrete Wavelet Transform

The MODWT is a shift-invariant wavelet transform that offers several advantages over the standard DWT:

- **Handles any signal length** - no need for padding or power-of-2 restrictions
- **Shift-invariant** - translated signals produce translated coefficients
- **Energy preserving** - total variance equals sum of coefficient variances
- **Perfect reconstruction** - exact signal recovery (within numerical precision)
- **Multi-resolution analysis** - analyze signals at different frequency scales

Key applications include:
- **Signal denoising** - threshold detail coefficients to remove noise
- **Feature extraction** - analyze energy distribution across scales
- **Time-frequency analysis** - maintains temporal alignment of features
- **Selective reconstruction** - reconstruct using specific frequency bands

### Doing own stuff e.g. data compression

For a lot of own stuff with JWave, have a look at the main junit test file / method: [a lot of examples](https://github.com/graetz23/JWave/tree/master/test/jwave)!

For example, how to perform a (losless) data compression with over 98 % compression rate using _all available_ wavelets is shown by the following [junit test](https://github.com/graetz23/JWave/blob/master/test/jwave/CompressorTest.java)!

### Some easy code ..

**example for 1-D DFT:**
```Java
Transform t = new Transform( new DiscreteFourierTransform( ) );

// arrTime = { r1, c1, r2, c2, ... }`
double[ ] arrTime = { 1., 1., 1., 1., 1., 1., 1., 1. };

double[ ] arrFreq = t.forward( arrTime ); // 1-D DFT forward

double[ ] arrReco = t.reverse( arrFreq ); // 1-D DFT reverse
```

**example for 1-D, 2-D FWT:**
```Java
Transform t = new Transform( new FastWaveletTransform( new Haar1( ) ) );

double[ ] arrTime = { 1., 1., 1., 1., 1., 1., 1., 1. };

double[ ] arrHilb = t.forward( arrTime ); // 1-D FWT Haar forward

double[ ] arrReco = t.reverse( arrHilb ); // 1-D FWT Haar reverse

double[ ][ ] matTime = { { 1., 1., 1., 1. },
                         { 1., 1., 1., 1. },
                         { 1., 1., 1., 1. },
                         { 1., 1., 1., 1. } };

double[ ][ ] matHilb = t.forward( matTime ); // 2-D FWT Haar forward

double[ ][ ] matReco = t.reverse( matHilb ); // 2-D FWT Haar reverse
// example in 3-D in common to 2-D using a N^3 double[ ][ ][ ] space.
```

**example for 1-D, 2-D WPT:**
```Java
Transform t = new Transform( new WaveletPacketTransform( new Haar1( ) ) );

double[ ] arrTime = { 1., 1., 1., 1., 1., 1., 1., 1. };

double[ ] arrHilb = t.forward( arrTime ); // 1-D WPT Haar forward

double[ ] arrReco = t.reverse( arrHilb ); // 1-D WPT Haar reverse

double[ ][ ] matTime = { { 1., 1., 1., 1. },
                         { 1., 1., 1., 1. },
                         { 1., 1., 1., 1. },
                         { 1., 1., 1., 1. } };

double[ ][ ] matHilb = t.forward( matTime ); // 2-D WPT Haar forward

double[ ][ ] matReco = t.reverse( matHilb ); // 2-D WPT Haar reverse

// example in 3-D in common to 2-D using a N^3 double[ ][ ][ ] space.
```

**example for 1-D FWT of arbitrary length:**
```Java
Transform t = new Transform(
               new AncientEgyptianDecomposition(
                new FastWaveletTransform(
                 new Haar1( ) ) ) );

double[ ] arrTime = { 1., 1., 1., 1., 1., 1., 1. }; // length = 7

double[ ] arrHilb = t.forward( arrTime ); // 1-D AED FWT Haar forward

//           |    2 steps    |   1 step   | 0  |
// arrHilb = { 2., 0., 0., 0., 1.41421, 0., 1. };
double[ ] arrReco = t.reverse( arrHilb ); // 1-D AED FWT Haar reverse
```

**example for 1-D WPT (WPD) of arbitrary length:**
```Java
Transform t = new Transform(
               new AncientEgyptianDecomposition(
                new WaveletPacketTransform(
                 new Haar1( ) ) ) );

double[ ] arrTime = { 1., 1., 1., 1., 1., 1., 1. }; // length = 7

double[ ] arrHilb = t.forward( arrTime ); // 1-D AED WPT Haar forward

double[ ] arrReco = t.reverse( arrHilb ); // 1-D AED WPT Haar reverse
```

**example for 1-D MODWT (shift-invariant transform):**
```Java
// MODWT works with any signal length and is shift-invariant
MODWTTransform modwt = new MODWTTransform( new Daubechies4( ) );

double[ ] signal = { 1.5, 0.8, -0.3, 2.1, -1.2, 0.7, 1.9 }; // arbitrary length

// Forward MODWT to level 3
double[ ][ ] coeffs = modwt.forwardMODWT( signal, 3 );
// coeffs[0] = detail coefficients level 1
// coeffs[1] = detail coefficients level 2  
// coeffs[2] = detail coefficients level 3
// coeffs[3] = approximation coefficients level 3

// Perfect reconstruction
double[ ] reconstructed = modwt.inverseMODWT( coeffs );

// Denoising example - threshold detail coefficients
double threshold = 0.5;
for( int level = 0; level < 3; level++ ) {
    for( int i = 0; i < coeffs[level].length; i++ ) {
        if( Math.abs( coeffs[level][i] ) < threshold )
            coeffs[level][i] = 0.0; // soft thresholding
    }
}
double[ ] denoised = modwt.inverseMODWT( coeffs );
```

**Have fun! :-)**

## CONTACT

If there are doubts, try mailing me, otherwise have fun with JWave.

## LICENSE

**JWave is distributed under the MIT License (MIT); this file is part of.**

**Copyright (c) 2008-2024 Christian (graetz23@gmail.com)**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## VERSION

**JWave is in version 250105.**

## CHANGE LOG

version **250105**:
- added Maximal Overlap Discrete Wavelet Transform (MODWT) implementation
 - shift-invariant wavelet transform for signals of arbitrary length
 - includes forward MODWT and inverse MODWT (iMODWT) with perfect reconstruction
 - supports multi-level decomposition with all JWave wavelets
 - ideal for denoising, feature extraction, and time-frequency analysis
 - comprehensive unit tests and example applications included

version **200303**:
- updating copyright and contact information to graetz23

version **180222**:
- updating project description, especially using markdown for the README.md,
- updated JUnit imports and travis-ci.org configuration for continuous integration,

version **160218**:
- added a new type of wavelet transform algorithm: Shifting Wavelet Transform.
 - the algorithm shifts a wavelet by smallest length of 2 over the input array,
 - then by the double of 4, 8, 16, .., p-1, p.
 - the reverse transform takes the largest wavelength of p and shifts,
 - then by half of p-1, .., 16, 8, 4, 2.

version **160109**:
- moved the JUnit tests to an own source directory
 - updated the build.xml file
 - automatically run all files with regex \*\*/\*Test\* in build directory
- updated the build.xml - allowing for OS specific builds now
 - set JUnit path OS dependent; MAC, WIN, or GNU/Linux and Unix
 - additionally set for WIN OS org.hamcrest.core JAR

version **160107**:
- added junit test for compressing a sine signal:
 - sine signal of 1024 * 1024 samples by 1024 oscillations
 - calculating the compression rate in percent; e.g. 99.70703125 % by Daubechies 20 wavelet
 - calculating the absolute maximal differences; e.g. 3.086649707542688E-4 by Daubechies 20 wavelet
- added method for calculating compression rate to class Compressor

version **160106**:
- added build.xml for using ant
 - set in build.xml your path JUnit4, e.g. /usr/share/java/junit.jar
 - ! to have the build done:
  - ant
  - should work easy peasy
 - ! for a quick console example:
  - ant run
 - ! for running the junit tests
  - ant test
 - ! to have a run or test on your system:
  - java -cp ./dist/JWave.jar jwave.JWave Fast Wavelet Transform Daubechies 20
  - java -cp /usr/share/java/junit4.jar:./dist/JWave.jar org.junit.runner.JUnitCore jwave.TransformTest
- added automatic build using travis-ci.com
 - ! https://travis-ci.org/graetz23/JWave
 - have a look at ./.travis.yml
- added README.md
- added LICENSE.md
- updated Copyright to the years 2008-2018
- fixed bug in JWave.java for calling console example with Haar wavelet
