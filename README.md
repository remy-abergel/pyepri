[![PyPI version](https://img.shields.io/pypi/v/pyepri?color=YELLOW)](https://pypi.org/project/pyepri/)
[![PyPI Downloads](https://static.pepy.tech/badge/pyepri)](https://pypi.org/project/pyepri)

# PyEPRI

A CPU & GPU compatible Python package for Electron Paramagnetic Resonance Imaging.

**Package documentation and tutorials are available
[here](https://pyepri.math.cnrs.fr/)**.

If you have any comments, questions, or suggestions regarding this
code, don't hesitate to open a
[discussion](https://github.com/remy-abergel/pyepri/discussions) or a
[bug issue](https://github.com/remy-abergel/pyepri/issues). 

## Installation

PyEPRI can be installed on all plateforms (Linux, MacOs or
Windows). However, GPU support is currently only available for systems
equipped with an NVIDIA graphics card and a working installation of
the CUDA drivers (which excludes MAC systems).

Installation commands provided below are valid for Linux and Mac
systems. More complete installation guidelines (including video
tutorials) for Linux, Mac and Windows are available in the [online
documentation](https://pyepri.math.cnrs.fr/installation.html).

### Install latest stable version using pip (recommended)

Open a terminal and execute the following steps in order to create a
virtual environment, and install the latest stable version of `pyepri`
from the [PyPi repository](https://pypi.org/project/pyepri/).

```bash
###################################################
# Create and activate a fresh virtual environment #
###################################################
python3 -m venv ~/.venv/pyepri
source ~/.venv/pyepri/bin/activate

#########################################################
# Install the `pyepri` package from the PyPi repository #
#########################################################
pip install pyepri

###########################################################
# Optional: enable {torch-cpu, torch-cuda, cupy} backends #
###########################################################

# enable `torch-cpu` backend
pip install pyepri[torch-cpu]

# enable `torch-cuda` backend (requires a NVIDIA graphics card with CUDA installed)
pip install pyepri[torch-cuda]

# enable `cupy` backend (requires a NVIDIA graphics card with CUDA installed)
# (please uncomment the appropriate line depending on your CUDA installation)
# pip install pyepri[cupy-cuda12x] # For CUDA 12.x
# pip install pyepri[cupy-cuda11x] # For CUDA 11.x
```

### Install latest version from Github

Open a terminal and execute the following steps in order to checkout
the current code release, create a virtual environment, and install
`pyepri` from the [github
repository](https://github.com/remy-abergel/pyepri/).

```bash
##################
# Clone the code #
##################
git clone https://github.com/remy-abergel/pyepri.git
cd pyepri

###################################################
# Create and activate a fresh virtual environment #
###################################################
python3 -m venv ~/.venv/pyepri
source ~/.venv/pyepri/bin/activate
	 
##########################################################
# Install the `pyepri` package from the checked out code #
# (do not forget the . at the end of the command line)   #
##########################################################
pip install -e .

###########################################################
# Optional: enable {torch-cpu, torch-cuda, cupy} backends #
###########################################################

# enable `torch-cpu` backend
pip install -e ".[torch-cpu]"

# enable `torch-cuda` backend (requires a NVIDIA graphics card with CUDA installed)
pip install -e ".[torch-cuda]"

# enable `cupy` backend (requires a NVIDIA graphics card with CUDA installed)
# (please uncomment the appropriate line depending on your CUDA installation)
# pip install -e ".[cupy-cuda12x]" # For CUDA 12.x
# pip install -e ".[cupy-cuda11x]" # For CUDA 11.x

################################################################
# If you want to compile the documentation by yourself, you    #
# must install the [doc] optional dependencies of the package, #
# compilation instructions are provided next                   #
################################################################
pip install -e ".[doc]" # install some optional dependencies
make -C docs html # build the documentation in html format
firefox docs/_build/html/index.html # open the built documentation (you can replace firefox by any other browser)
```

Because this installation was done in *editable* mode (thanks to the
``-e`` option of ``pip``), any further update of the repository (e.g.,
using the syncing commang ``git pull``) will also update the current
installation of the package.

### Troubleshooting

+ Mac users are strongly recommended to use ``bash`` shell instead of
  ``zsh`` to avoid slow copy-paste issues (type ``chsh -s /bin/bash``
  in a terminal).

+ Display issues related to matplotlib interactive mode were reported
  on Linux systems and were solved by installing ``python3-tk`` (type
  ``sudo apt-get install python3-tk`` in a terminal).
  
+ If the installation of the package or one of its optional dependency
  fails, you may have more chance with
  [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) (or
  [conda](https://anaconda.org/anaconda/conda)).

+ If you still encounter difficulties, feel free to open a [bug
  issue](https://github.com/remy-abergel/pyepri/issues).

## License

PyEPRI was created by Rémy Abergel ([Centre National de la Recherche
Scientifique](https://www.cnrs.fr/fr), [Université Paris
Cité](https://u-paris.fr/), [Laboratoire
MAP5](https://map5.mi.parisdescartes.fr/)). It is licensed under the
terms of the [MIT license](LICENSE).
