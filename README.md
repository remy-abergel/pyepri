# PyEPRI

A CPU & GPU compatible Python package for Electron Paramagnetic Resonance Imaging.

**Package documentation and tutorials are available
[here](https://pyepri.math.cnrs.fr/)**.

If you have any comments, questions, or suggestions regarding this
code, don't hesitate to open a
[discussion](https://github.com/remy-abergel/pyepri/discussions) or a
[bug issue](https://github.com/remy-abergel/pyepri/issues). 

## Installation

### System requirements 

PyEPRI can be installed on all plateforms (Linux, MacOs or
Windows). However, GPU support is currently only available for systems
equipped with an NVIDIA graphics card and a working installation of
the CUDA drivers (which excludes MAC systems).

The following installation guidelines assume that you have the
following libraries installed on your system: 

- **[mandatory]** python3 (the Python 3 programming language)
- **[mandatory]** python3-pip (to install Python packages using the ``pip`` command)
- **[mandatory]** python3-venv (for the creation of virtual environment)
- **[recommended]** python3-tk (to avoid display issues on Linux systems)
- **[optional]** git (if you want to install the PyEPRI package via the
  [github repository](https://github.com/remy-abergel/pyepri/))
  
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

################################################################
# Optionally (and advised if you have a GPU device with CUDA   #
# installed), you can enable `torch` and/or `cupy` backends by #
# executing the following commands                             #
################################################################

# enable `torch` backend support
pip install pyepri[torch] # for enabling `torch` backend support

# enable `cupy` backend support: you need to select the 
# appropriate line depending on your system 
#
# PLEASE BE CAREFUL NOT TO INSTALL MULTIPLE CUPY PACKAGES AT
# THE SAME TIME TO AVOID INTERNAL CONFLICTS
#
pip install pyepri[cupy-cuda12x] # For CUDA 12.x
pip install pyepri[cupy-cuda11x] # For CUDA 11.x
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

################################################################
# Optionally (and advised if you have a GPU device with CUDA   #
# installed), you can enable `torch` and/or `cupy` backends by #
# executing the following commands                             #
################################################################

# enable `torch` backend support
pip install ".[torch]" # for enabling `torch` backend support

# enable `cupy` backend support: you need to select the 
# appropriate line depending on your system 
#
# PLEASE BE CAREFUL NOT TO INSTALL MULTIPLE CUPY PACKAGES AT
# THE SAME TIME TO AVOID INTERNAL CONFLICTS
#
pip install ".[cupy-cuda12x]" # For CUDA 12.x
pip install ".[cupy-cuda11x]" # For CUDA 11.x

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
