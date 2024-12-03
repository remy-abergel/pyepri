(heading-installation)=
# Installation

## Install latest stable version using pip (recommended)

Assuming you have a compatible system with `python3` and `pip`
installed, the following steps will create a virtual environment, and
install the latest stable version of `pyepri`.

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

## Install latest version from sources

Assuming you have a compatible system with `python3`, `pip` and `git`
installed, the following steps will checkout current code release,
create a virtual environment, and install `pyepri`.

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

## Troubleshooting

+ If the installation of the package or one of its optional dependency
  fails, you may have more chance with
  [conda](https://anaconda.org/anaconda/conda) (or
  [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)).

+ If you still encounter difficulties, feel free to open a [bug
  issue](https://github.com/remy-abergel/pyepri/issues).

