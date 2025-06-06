# PyEPRI -- A CPU & GPU compatible Python package for Electron Paramagnetic Resonance Imaging

This Python package implements low-level operators involved in
Electron Paramagnetic Resonance (EPR) and also high-level advanced
algorithms for end-users. This package is fully implemented in Python
and provides both CPU and GPU computation capabilities, through the
libraries [Numpy](https://numpy.org/), [PyTorch](https://pytorch.org/)
and [Cupy](https://cupy.dev/). 

The PyEPRI package is the fruit of a long-term interdisciplinary
collaboration between two French laboratories hosted at [Université
Paris Cité](https://u-paris.fr/): the [MAP5
Laboratory](https://map5.mi.parisdescartes.fr/) (Laboratory of Applied
Mathematics) and the
[LCBPT](https://lcbpt.biomedicale.parisdescartes.fr/) (Laboratory of
Chemistry and Biochemistry). If you use PyEPRI, please cite the
following paper:

```{container} PyEPRIcitation
Rémy Abergel, Sylvain Durand, and Yves-Michel Frapart. PyEPRI: a CPU & GPU compatible python package for electron paramagnetic resonance imaging. Journal of Magnetic Resonance, p. 107891, 2025. DOI: [10.1016/j.jmr.2025.107891](https://doi.org/10.1016/j.jmr.2025.107891).
```

The PyEPRI package is intended for the entire EPR community. It is
hosted on [Github](https://github.com/remy-abergel/pyepri) and
licensed under the terms of the [MIT
License](https://github.com/remy-abergel/pyepri/blob/main/LICENSE).

```{toctree}
:maxdepth: 2
:caption: The PyEPRI package
:hidden:

installation
definitions
processing
API documentation <autoapi/pyepri/index>
acknowledgments
references
```

```{toctree}
:maxdepth: 2
:caption: Tutorials & Examples
:hidden:
	
_gallery/getting_started/index
_gallery/filtered_back_projection/index
_gallery/tv_regularization/index
_gallery/sources_separation/index
_gallery/core/index
```
