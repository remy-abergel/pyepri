# Changelog

<!--next-version-placeholder-->

## v1.0.1 (under development)

### code

- increased to `1E6` the default maximal number of iterations
  (parameter `nitermax`) for optimization schemes and related functions

- fixed type inference for `backend.from_numpy()` for torch backend

- fixed typo in function name (read_bruker_best3_dataset instead of
  read_bruker_bes3t_dataset), old name was kept available for backward
  compatibility

- fixed sphinx rendering issues in various function headers in
  [multisrc.py](src/pyepri/multisrc.py)

### documentation

- fixed bibtex reference [Bar21]

- fixed minor issues in demonstration examples

- fixed pip installation instructions in [README.md](README.md)

## v1.0.0 (October 16, 2024)

- First public release of `pyepri` (imported from private dev repository, tag = v1.0.0).
