# PyEPRI

Advanced Algorithms and Low-Level Operators for Electron Paramagnetic
Resonance Imaging.

**Package documentation and tutorials are available
[here](https://pyepri.pages.math.cnrs.fr/doc/)**.

If you have any comments, questions, or suggestions regarding this
code, don't hesitate to open a
[discussion](https://github.com/remy-abergel/pyepri/discussions) or a
[bug issue](https://github.com/remy-abergel/pyepri/issues). 

## Installation

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
pip install -e ".[torch]" # for enabling `torch` backend support
pip install -e ".[cupy]" # for enabling `cupy` backend support

################################################################
# If you want to compile the documentation by yourself, you    #
# must install the [doc] optional dependencies of the package, #
# compilation instructions are provided next                   #
################################################################
pip install -e ".[doc]"
make -C docs html
firefox docs/_build/html/index.html # open the built documentation 
                                    # (you can replace firefox by 
									# any other browser)
```

Because this installation was done in *editable* mode (thanks to the
``-e`` option of ``pip``), any further update of the repository (e.g.,
using the syncing commang ``git pull``) will also update the current
installation of the package.

## License

PyEPRI was created by Rémy Abergel ([Centre National de la Recherche
Scientifique](https://www.cnrs.fr/fr), [Université Paris
Cité](https://u-paris.fr/), [Laboratoire
MAP5](https://map5.mi.parisdescartes.fr/)). It is licensed under the
terms of the [MIT license](LICENSE).