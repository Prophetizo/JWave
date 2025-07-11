# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JWave is a comprehensive Java library for wavelet transforms and signal processing. It implements various discrete and continuous wavelet transforms, Fourier transforms, and provides tools for time-frequency analysis.

## Build and Test Commands

```bash
# Build the project
mvn clean compile

# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=FastWaveletTransformTest
mvn test -Dtest=MODWTTransformTest

# Package as JAR
mvn package

# Package without tests
mvn package -DskipTests

# Update dependencies
mvn dependency:resolve -U

# Run JWave from command line
java -cp target/jwave-2.0.0.jar jwave.JWave "Fast Wavelet Transform" "Daubechies 20"
```

## Architecture

### Design Patterns
- **Strategy Pattern**: Transform algorithms are interchangeable through `BasicTransform`
- **Builder Pattern**: `TransformBuilder` and `WaveletBuilder` for object creation
- **Facade Pattern**: `Transform` class provides unified interface to all transforms
- **Template Method**: Abstract classes define algorithm structure

### Core Components

1. **Entry Points**:
   - `Transform.java` - Main API entry point for library usage
   - `JWave.java` - Console application for testing

2. **Transform Types** (in `jwave.transforms/`):
   - `FastWaveletTransform` (FWT) - Requires power-of-2 length
   - `WaveletPacketTransform` (WPT) - Requires power-of-2 length
   - `MODWTTransform` - Handles arbitrary length, 1D only, shift-invariant
   - `ContinuousWaveletTransform` (CWT) - Time-frequency analysis, 1D only
   - `DiscreteFourierTransform` (DFT)
   - `FastFourierTransform` (FFT)
   - `AncientEgyptianDecomposition` - Wrapper for odd-length signals
   - `ShiftingWaveletTransform` - Shifted wavelet analysis

3. **Wavelet Families** (in `jwave.transforms.wavelets/`):
   - **Orthogonal**: Haar, Daubechies (2-20), Symlets (2-20), Coiflets (1-5)
   - **Biorthogonal**: Various BiOrthogonal wavelets
   - **Continuous**: Morlet, Mexican Hat, Paul, DOG, Meyer
   - **Other**: Legendre, Battle, CDF, Discrete Meyer

4. **Exception Hierarchy**:
   - `JWaveException` - Base exception
   - `JWaveError` - Recoverable errors
   - `JWaveFailure` - Non-recoverable failures

### Key Implementation Notes

- All transforms support 1D, 2D, and 3D operations (except MODWT and CWT which are 1D only)
- FWT and WPT require power-of-2 signal lengths
- MODWT handles arbitrary length signals natively
- Thread-safe implementations
- No external runtime dependencies (only JUnit for testing)
- Java 21 required

### Adding New Components

**New Wavelet**:
1. Create class in appropriate package under `jwave.transforms.wavelets/`
2. Extend `Wavelet` abstract class
3. Implement `matDecompose()` and `matReconstruct()` methods
4. Add to `WaveletBuilder` for factory access
5. Create unit test

**New Transform**:
1. Create class in `jwave.transforms/`
2. Extend `BasicTransform` abstract class
3. Implement forward/reverse methods for required dimensions
4. Add to `TransformBuilder` if needed
5. Create comprehensive unit tests

### Testing Strategy

- Base test class: `jwave.Base` provides utility methods
- Test all transforms with various wavelets
- Test edge cases (odd lengths, boundary conditions)
- Performance tests for optimization validation
- Integration tests for multi-dimensional transforms