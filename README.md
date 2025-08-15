# Phasepack

A toolkit for phase-based image feature detection.

This toolkit consists of a set of functions which use information contained within the phase of a Fourier-transformed image to detect localised features such as edges, blobs and corners. These methods have the key advantage that the properties they measure are invariant with respect to image brightness and contrast.

## Functions

- **`phasecong`** - Phase congruency using oriented filters
- **`phasecongmono`** - Fast phase congruency using monogenic filters  
- **`phasesym`** - Phase symmetry using oriented filters
- **`phasesymmono`** - Fast phase symmetry using monogenic filters

For more information on a particular function, see the associated docstring and the references therein.

## Installation

```bash
pip install phasepack
```

## Fast Fourier Transforms

All functions in this module use the Fast Fourier Transform (FFT) via [pyFFTW](http://hgomersall.github.io/pyFFTW/), which provides Python bindings to the FFTW C library. This is substantially faster than scipy's default `fftpack` module and is now included as a required dependency.

## Authorship

These functions were originally written for MATLAB by Peter Kovesi, and were ported to Python by Alistair Muldal. The original MATLAB code, as well as further explanatory information and references are available from [Peter Kovesi's website](http://www.csse.uwa.edu.au/~pk/Research/MatlabFns/index.html#phasecong).

## License

MIT License - see [LICENSE](LICENSE) file for details.