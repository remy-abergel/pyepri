.. _heading-installation:

.. |ico-req| image:: _static/ico-req.png
   :height: 2ex
.. |ico-opt| image:: _static/ico-opt.png
   :height: 2ex
	    
Installation
------------

System requirements
~~~~~~~~~~~~~~~~~~~

PyEPRI can be installed on all plateforms (Linux, MacOs or
Windows). However, GPU support is currently only available for systems
equipped with an NVIDIA graphics card and a working installation of
the CUDA drivers (which excludes MAC systems).


.. tabs::

   .. group-tab:: Linux

      The installation guidelines assume that you have the following
      installed on your system:
      
      - |ico-req| ``python3`` (the Python 3 programming language)	
      - |ico-req| ``python3-pip`` (to install Python packages using the ``pip``
        command)
      - |ico-req| ``python3-venv`` (for the creation of virtual environment)      
      - |ico-opt| ``python3-tk`` (recommended, to avoid display issues on some Linux systems)
      - |ico-opt| an integrated development environment (IDE) suited
        to Python (for instance `Visual Studio Code
        <https://code.visualstudio.com/>`_)
      
      Under a Debian GNU/Linux distribution, one can easily get the
      required and recommended libraries by typing into a terminal the
      following apt-get command (requires superuser (root) privilege).
      
      .. code:: bash
	 
	 sudo apt update && sudo apt-get install python3 python3-pip python3-venv python3-tk
      
   .. group-tab:: Mac OSX

      The installation guidelines assume that you have the following
      installed on your system:
      
      - |ico-req| an official Python 3 obtained from from
        `<https://www.python.org/downloads/>`_.
      - |ico-opt| an integrated development environment (IDE) suited
        to Python (for instance `Visual Studio Code
        <https://code.visualstudio.com/>`_)
      
   .. group-tab:: Windows

      The installation guidelines assume that you have the following
      installed on your system:
      
      - |ico-req| an official Python 3 obtained from from
        `<https://www.python.org/downloads/>`_.
      - |ico-opt| an integrated development environment (IDE) suited
        to Python (for instance `Visual Studio Code
        <https://code.visualstudio.com/>`_)
      
If you encounter installation difficulties, feel free to reach us by
opening a `bug issue
<https://github.com/remy-abergel/pyepri/issues>`_.


Install latest stable version using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: Linux

      **Package installation and quick test in video**
      
      .. video:: _static/pyepri_installation_linux.webm
	 :width: 100%
	 :align: center

      |
      
      **Installation instructions in command lines**
      
      Open a terminal and execute the following steps in order to
      create a virtual environment, and install the latest stable
      version of `pyepri` from the `PyPi repository
      <https://pypi.org/project/pyepri>`_.

      .. code:: bash
   
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
	 
   .. group-tab:: Mac OSX

      **Package installation and quick test in video**
      
      .. video:: _static/pyepri_installation_macos.mp4
	 :width: 100%
	 :align: center
	 :caption: (many thanks to Camille Pouchol for sharing their
                   Macbook)

      **Installation instructions in command lines**
      
      Open a terminal and execute the following steps in order to
      create a virtual environment, and install the latest stable
      version of `pyepri` from the `PyPi repository
      <https://pypi.org/project/pyepri>`_.

      .. code:: bash
   
	 ###################################################
	 # Create and activate a fresh virtual environment #
	 ###################################################
	 python3 -m venv ~/.venv/pyepri
	 source ~/.venv/pyepri/bin/activate
	 
	 #########################################################
	 # Install the `pyepri` package from the PyPi repository #
	 #########################################################
	 pip install pyepri
	 
	 ############################################################
	 # Optional: enable torch-cpu backend (GPU backends are not #
	 # available yet on Mac systems)                            #
	 ############################################################
	 pip install pyepri[torch-cpu]
   
   .. group-tab:: Windows

      **Package installation (using VScode) and quick test in video**
      
      .. video:: _static/pyepri_installation_windows.webm
	 :width: 100%
	 :align: center
	 :caption: (this video was done using a very slow machine,
                   video editing tries to compensate for that)
      
      **Installation instructions in command lines (cmd)**
      
      For creating a virtual environment and installing the latest
      stable version of `pyepri` from the `PyPi repository
      <https://pypi.org/project/pyepri>`_ in command lines, open a
      MSDos terminal and execute the following commands.
      
      .. code:: bat
	 
	 :: ------------------------------------------------ 
	 :: Create and activate a fresh virtual environment 
	 :: ------------------------------------------------
	 py -m venv pyepri-venv
	 .\pyepri-venv\Scripts\activate
	 
	 :: --------------------------------------------------------
	 :: Optional: enable {torch-cpu, torch-cuda, cupy} backends
	 :: --------------------------------------------------------
	 
	 :: enable `torch-cpu` backend
	 pip install pyepri[torch-cpu]

	 :: enable `torch-cuda` backend (requires a NVIDIA graphics card with CUDA installed)
	 pip install pyepri[torch-cuda]
	 
	 :: enable `cupy` backend (requires a NVIDIA graphics card with CUDA installed)
	 :: (please uncomment the appropriate line depending on your CUDA installation)
	 :: pip install pyepri[cupy-cuda12x] # For CUDA 12.x
	 :: pip install pyepri[cupy-cuda11x] # For CUDA 11.x

Install latest version from Github
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a terminal and execute the following steps in order to
checkout the current code release, create a virtual environment,
and install `pyepri` from the `github repository
<https://github.com/remy-abergel/pyepri>`_. 

.. tabs::

   .. group-tab:: Linux
		  
      **Installation instructions in command lines**
      
      .. code:: bash
   
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
	 
      **Note**: the instructions above assume that you have ``git``
      and ``make`` installed on your system.
      
   .. group-tab:: Mac OSX
      
      **Installation instructions in command lines**
      
      .. code:: bash
	 
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
	 
	 ############################################################
	 # Optional: enable torch-cpu backend (GPU backends are not #
	 # available yet on Mac systems)                            #
	 ############################################################
	 pip install -e ".[torch-cpu]"
	 
	 ################################################################
	 # If you want to compile the documentation by yourself, you    #
	 # must install the [doc] optional dependencies of the package, #
	 # compilation instructions are provided next                   #
	 ################################################################
	 pip install -e ".[doc]" # install some optional dependencies
	 make -C docs html # build the documentation in html format
	 firefox docs/_build/html/index.html # open the built documentation (you can replace firefox by any other browser)
   
      **Note**: the instructions above assume that you have ``git``
      and ``make`` installed on your system.
      
   .. group-tab:: Windows
      
      **Installation instructions in command lines (cmd)**
      
      .. code:: bat
	 
	 :: ---------------
	 :: Clone the code 
	 :: ---------------
	 git clone https://github.com/remy-abergel/pyepri.git
	 cd pyepri
	 
	 :: ------------------------------------------------
	 :: Create and activate a fresh virtual environment
	 :: ------------------------------------------------
	 py -m venv pyepri-venv
	 .\pyepri-venv\Scripts\activate
	 	 
	 :: -------------------------------------------------------
	 :: Install the `pyepri` package from the checked out code 
	 :: (do not forget the . at the end of the command line)   
	 :: -------------------------------------------------------
	 pip install -e .
	 
	 :: --------------------------------------------------------
	 :: Optional: enable {torch-cpu, torch-cuda, cupy} backends
	 :: --------------------------------------------------------

	 :: enable `torch-cpu` backend
	 pip install -e ".[torch-cpu]"

	 :: enable `torch-cuda` backend (requires a NVIDIA graphics card with CUDA installed)
	 pip install -e ".[torch-cuda]"
	 
	 :: enable `cupy` backend (requires a NVIDIA graphics card with CUDA installed)
	 :: (please uncomment the appropriate line depending on your CUDA installation)
	 :: pip install -e ".[cupy-cuda12x]" # For CUDA 12.x
	 :: pip install -e ".[cupy-cuda11x]" # For CUDA 11.x	 
      
      **Note**: the instructions above assume that you have `git
      <https://git-scm.com/downloads/win>`_ installed on your
      system.
      
Because this installation was done in *editable* mode (thanks to the
``-e`` option of ``pip``), any further update of the repository (e.g.,
using the syncing commang ``git pull``) will also update the current
installation of the package.

Troubleshooting
~~~~~~~~~~~~~~~

+ Mac users are strongly recommended to use ``bash`` shell instead of
  ``zsh`` to avoid slow copy-paste issues (type ``chsh -s /bin/bash``
  in a terminal).

+ Display issues related to matplotlib interactive mode were reported
  on Linux systems and were solved by installing ``python3-tk`` (type
  ``sudo apt-get install python3-tk`` in a terminal).
  
+ If the installation of the package or one of its optional dependency
  fails, you may have more chance with `miniconda
  <https://docs.anaconda.com/miniconda/miniconda-install/>`_ (or
  `conda <https://anaconda.org/anaconda/conda>`_).

+ If you still encounter difficulties, feel free to open a `bug issue
  <https://github.com/remy-abergel/pyepri/issues>`_.

