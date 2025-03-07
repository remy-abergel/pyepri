# Changelog

<!--next-version-placeholder-->

## v1.0.4 (March 7th, 2025)

### Documentation

- improved installation instructions and added videos

### Code 

- Removed escape character `\` from class docstrings in
  [backends.py](src/pyepri/backends.py) and
  [displayers.py](src/pyepri/displayers.py) to avoid SyntaxWarning
  ([#8](../../pull/8))

- PyEPRI now require numpy >= 2.0.0, unified FFT functions for `numpy`
  and `cupy` backends and fixed numpy deprecated warning
  ([#7](../../pull/7))

- fixed ruff errors (all checks passed): remove unused variables and
  fixed minor bugs ([#6](../../pull/6))

### Repository 

- shorten installation instructions and refer to the online
  documentation ([#9](../../pull/9))

- added welcome message in the [Discussions](../../discussions)
  section ([#5](../../pull/5))
  
- added Github continuous integration workflow ([#2](../../pull/2))
  and issue templates ([#3](../../pull/3), [#4](../../pull/4))

### Python Packaging

- Fix datasets packaging for pip+git installation
  ([#11](../../pull/11))

## v1.0.3 (February 5th, 2025)

### Packaging

- split [torch] optional dependencies into [torch-cpu] (with finufft)
  and [torch-cuda] (cufinufft) in order to avoid pip install error on
  windows systems

### Documentation

- updated installation instructions

## v1.0.2 (January 3rd, 2025)

### Code

- minor fix for `processing.eprfbp2d` and `processing.eprfbp3d`
  functions (`displayer=None` was not working as expected)

- changed normalization in `processing.eprfbp2d` and
  `processing.eprfbp3d` to get a consistent quadrature scheme

### Documentation

- added the mathematical description of the filtered backrojection
  scheme implemented in the package

- fixed Gaussian derivative normalization factor for demo with
  simulated reference spectra

## v1.0.1 (December 3rd, 2024)

### Code

- temporary fix related to a multithreading issue with FINUFFT (see
  [FINUFFT issue
  #596](https://github.com/flatironinstitute/finufft/issues/596)):
  introduced a decorator in [backends.py](src/pyepri/backends.py) to
  change the default value of the `nthreads` keyword argument of the
  finufft functions according to the number of physical cores (or the
  `OMP_NUM_THREADS` environment variable if set)

- increased to `1E6` the default maximal number of iterations
  (parameter `nitermax`) for optimization schemes and related functions

- fixed type inference for `backend.from_numpy()` (torch backend)

- fixed typo in function name (read_bruker_best3_dataset instead of
  read_bruker_bes3t_dataset), old name was kept available for backward
  compatibility

- fixed sphinx rendering issues in various function headers in
  [multisrc.py](src/pyepri/multisrc.py)

### Documentation

- updated installation instructions to allow cupy installation using
  pip

- fixed bibtex reference [Bar21]

- fixed minor issues in demonstration examples

- fixed pip installation instructions in [README.md](README.md)

## v1.0.0 (October 16, 2024)

- First public release of `pyepri` (imported from private dev repository, tag = v1.0.0).
