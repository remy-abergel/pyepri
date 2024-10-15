"""Basic function for loading data in Bruker BES3T format.

The functions of this module were adapted from the open source MIT
licensed `DIVE package (v0.2.1)
<https://github.com/StollLab/dive/releases/tag/v0.2.1>`_, as detailed
below:

+ :py:func:`read_bruker_description_file` is a copy of the
  ``read_description_file()`` function coded into the file
  ``deerload.py`` of the DIVE package;

+ :py:func:`read_bruker_best3_dataset` was adapted from the
  ``deerload()`` function of coded ``deerload.py`` of the DIVE
  package.

We provide below the copy of the license file of the DIVE package
(v.0.2.1), in accordance with the rules of the MIT license.

.. code-block:: text

   MIT License
   
   Copyright (c) 2024 Sarah Sweger, Julian Cheung, Lukas Zha, Stephan Pribitzer, 
   Stefan Stoll
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

"""
import numpy as np
import os
import re
import pyepri.backends as backends

def read_bruker_description_file(name: str) -> dict:
    """Retrieves the parameters from a .DSC files as a dictionary.
    
    Parameters
    ----------
    name : str
        The filename to be read, including the .DSC extension.
    
    Returns
    -------
    Parameters : dict
        A dictionary of the parameters in the .DSC file.
    """
    with open(name,"r") as f:
        allLines = f.readlines()
    
    # Remove lines with comments
    allLines = [l for l in allLines if not l.startswith("*")]
    
    # Remove newlines
    allLines = [l.rstrip("\n") for l in allLines]
    
    # Remove empty lines
    allLines = [l for l in allLines if l]
    
    # Merge any line ending in \n\ with the subsequent one
    allLines2 = []
    val = ""
    for line in allLines:
        val = "".join([val, line])    
        if val.endswith("\\"):
            val = val.strip("\\")
        else:
            allLines2.append(val)
            val = ""
    allLines = allLines2
    
    Parameters = {}
    SectionName = None
    DeviceName = None
    
    # Regular expressions to match layer/section headers, device block headers, and key-value lines
    reSectionHeader = re.compile(r"#(\w+)\W+(\d+.\d+)")
    reDeviceHeader = re.compile(r"\.DVC\W+(\w+),\W+(\d+\.\d+)")
    reKeyValue = re.compile(r"(\w+)\W+(.*)")
    
    for line in allLines:
        
        # Layer/section header (possible values: #DESC, #SPL, #DSL, #MHL)
        mo1 = reSectionHeader.search(line) 
        if mo1:
            SectionName = mo1.group(1)
            SectionVersion = mo1.group(2)
            if SectionName not in {"DESC","SPL","DSL","MHL"}:
                raise ValueError("Found unrecognized section " + SectionName + ".")
            Parameters[SectionName] = {"_version": SectionVersion}
            DeviceName = None
            continue
        
        # Device block header (starts with .DVC)
        mo2 = reDeviceHeader.search(line)
        if mo2:
            DeviceName = mo2.group(1)
            DeviceVersion = mo2.group(2)
            Parameters[SectionName][DeviceName] = {"_version": DeviceVersion}
            continue
        
        # Key/value entry
        mo3 = reKeyValue.search(line)
        if not mo3:
            raise ValueError("Key/value pair expected.")        
        if not SectionName:
            raise ValueError("Found a line with key/value pair outside any layer.")
        if SectionName=="DSL" and not DeviceName:
            raise ValueError("Found a line with key-value pair outside .DVC in #DSL layer.")
        
        Key = mo3.group(1)
        Value = mo3.group(2)
        if DeviceName:
            Parameters[SectionName][DeviceName][Key] = Value
        else:
            Parameters[SectionName][Key] = Value
        
        # Assert DESC section is present
        if "DESC" not in Parameters:
            raise ValueError("Missing DESC section in .DSC file.")
    
    return Parameters

def read_bruker_best3_dataset(name, squeeze=True, stack=True, dtype=None, backend=None):
    """Read Bruker data from a .DSC or .DTA file.
    
    Read files in BES3T format (Bruker EPR Standard for Spectrum
    Storage and Transfer) which is used on Bruker ELEXSYS and EMX
    machines.
    
    The BES3T format consists of at least two files, a data file with
    extension ``.DTA`` and a descriptor file with extension
    ``.DSC``. This code assumes that the .DSC file and the .DTA file
    are both stored in the same directory.
    
    Code based on BES3T version 1.2 (Xepr >= 2.1), and adapted from
    the ``deerload()`` function of the open source MIT licensed DIVE
    package (v0.2.1).
    
    This code is restricted to real datasets with regular sampling
    grids.
    
    Parameters
    ----------
    
    name : str
        The full filename of the .DSC or .DTA file (including the .DSC
        or .DTA extension). Both the .DSC and the .DAT files must be
        stored in the same folder.
    
    squeeze : bool, optional
        When squeeze is True, axes with lenght one will be removed
        from the output ``data`` array.
    
    stack : bool, optional 
        When the input dataset contain more than one datapoint along
        the Z axis, set ``stack = True`` to keep the original data
        shape, or use ``stack = False`` to reshape the data (Z-axis
        signals are stacked vertically along the first axis).
    
    backends : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a numpy backend is instantiated.x

    Returns
    -------

    B : ndarray
        Monodimensional array containing the sampling nodes of the
        projections (homogeneous magnetic field), stored as a
        monodimensional array.

    data : ndarray
        Multidimensional array containing the measurements.

    parameters : dict
        Retrieved additional parameters.

    """
    # instantiate backend (if necessary)
    if backend is None:
        backend = backends.create_numpy_backend()

    # retrieve .DSC and .DTA filenames (assume both files are stored
    # in the same directory)
    filename = name[:-4]
    fileextension = name[-4:].upper() # case insensitive extension
    if fileextension in ['.DSC','.DTA']:
        filename_dsc = filename + '.DSC'
        filename_dta = filename + '.DTA'
    else:
        raise ValueError("Only Bruker BES3T files with extensions .DSC or .DTA are supported.")
    
    if not os.path.exists(filename_dta):
        filename_dta = filename_dta[:-4] + filename_dta[-4:].lower()
        filename_dsc = filename_dsc[:-4] + filename_dsc[-4:].lower()
    
    if not os.path.exists(filename_dta):
        RuntimeError("Could not find .DTA file")
    if not os.path.exists(filename_dsc):
        RuntimeError("Could not find .DSC file")
    
    # Read descriptor file (contains key-value pairs)
    parameters = read_bruker_description_file(filename_dsc)
    if 'DESC' not in parameters:
        raise ValueError('Failed to retrieve DESC parameters in the .DSC file.')
    parDESC = parameters["DESC"]
    
    # XPTS, YPTS, ZPTS specify the number of data points along x, y and z.
    if 'XPTS' in parDESC:
        nx = int(parDESC['XPTS'])
    else:
        raise ValueError('No XPTS in DSC file.')
    
    if 'YPTS' in parDESC:
        ny = int(parDESC['YPTS'])
    else:
        ny = 1
    
    if 'ZPTS' in parDESC:
        nz = int(parDESC['ZPTS'])
    else:
        nz = 1
    
    # BSEQ: Byte Sequence
    # BSEQ describes the byte order of the data, big-endian (BIG,
    # encoding = '>') or little-endian (LIT, encoding = '<').
    if 'BSEQ' in parDESC:
        if 'BIG' == parDESC['BSEQ']:
            byteorder = '>' 
        elif 'LIT' == parDESC['BSEQ']:
            byteorder = '<'
        else:
            raise ValueError('Unknown value for keyword BSEQ in .DSC file!')
    else:
        warn('Keyword BSEQ not found in .DSC file! Assuming BSEQ=BIG.')
        byteorder = '>'
    
    # IRFMT: Item Real Format
    # IIFMT: Item Imaginary Format
    # Data format tag of BES3T is IRFMT for the real part and IIFMT for the imaginary part.
    if 'IRFMT' in parDESC:
        IRFTM = parDESC["IRFMT"]
        if 'C' == IRFTM:
            dt_spc = np.dtype('int8')
        elif 'S' == IRFTM:
            dt_spc = np.dtype('int16')
        elif 'I' == IRFTM:
            dt_spc = np.dtype('int32')
        elif 'F' == IRFTM:
            dt_spc = np.dtype('float32')
        elif 'D' == IRFTM:
            dt_spc = np.dtype('float64')
        elif 'A' == IRFTM:
            raise TypeError('Cannot read BES3T data in ASCII format!')
        elif ('0' or 'N') == IRFTM:
            raise ValueError('No BES3T data!')
        else:
            raise ValueError('Unknown value for keyword IRFMT in .DSC file!')
    else:
        raise ValueError('Keyword IRFMT not found in .DSC file!')
    
    # IRFMT and IIFMT must be identical.
    if "IIFMT" in parDESC and parDESC["IIFMT"] != parDESC["IRFMT"]:
        raise ValueError("IRFMT and IIFMT in DSC file must be identical.")
    
    # Construct abscissa vectors (B)
    if ('XMIN' not in parDESC):
        raise ValueError('Could not retrieve X-axis (No XMIN in DSC file)!')
    if ('XWID' not in parDESC):
        raise ValueError('Could not retrieve X-axis (No XWID in DSC file)!')
    minimum = float(parDESC[str('XMIN')])
    width = float(parDESC[str('XWID')])
    npts = int(parDESC[str('XPTS')])
    B = backend.from_numpy(backend.linspace(minimum, minimum + width, npts, dtype=dtype))
    
    # Read data matrix (real format only)
    dt_data = dt_spc
    dt_spc = dt_spc.newbyteorder(byteorder)
    if 'IKKF' in parDESC: 
        if parDESC['IKKF'] != 'REAL':
            raise ValueError("Unsupported value for keyword IKKF in .DSC file!")
    else:
        warn("Keyword IKKF not found in .DSC file! Assuming IKKF is REAL.")
        
    with open(filename_dta, 'rb') as fp:
        data = backend.from_numpy(np.frombuffer(fp.read(), dtype=dt_spc).reshape(ny, nx, nz).astype(dt_data))

    # if the dataset contains projections, compute the coordinates of
    # the associated field gradient vectors    
    if ('DSL' in parameters) and 'grdUnit' in parameters['DSL']:
        grdUnit = parameters['DSL']['grdUnit']
        if ('SPL' in parameters) and ('GRAD' in parameters['SPL']):
            mu = float(parameters['SPL']['GRAD'])
        else:
            warn('Failed to retrieve field gradient magnitude (no SPL and/or GRAD in DSC file)!')
        if ('FirstAlpha' in grdUnit) and ('NrOfAlpha' in grdUnit):
            first_angle = float(grdUnit['FirstAlpha'][:-4])
            angle_step = 2. * first_angle
            N_angle = int(grdUnit['NrOfAlpha'])
            angle_rad = ((first_angle + (np.arange(N_angle) * angle_step)) * np.pi / 180.)
        if ('ImageType' in grdUnit) and ('2D' == grdUnit['ImageType'].upper()): # 2D imaging
            gx = (mu * np.cos(angle_rad))
            gy = (mu * np.sin(angle_rad))
            fgrad = backend.from_numpy(np.array((gx, gy)).astype(dt_data))
            parameters['FGRAD'] = fgrad
        elif ('ImageType' in grdUnit) and ('3D' == grdUnit['ImageType'].upper()): # 3D imaging
            [theta_rad, phi_rad] = np.meshgrid(angle_rad, angle_rad)
            phi_rad = phi_rad.reshape((-1,))
            theta_rad = theta_rad.reshape((-1,))
            gx = mu * np.cos(theta_rad) * np.sin(phi_rad)
            gy = mu * np.sin(theta_rad) * np.sin(phi_rad)
            gz = mu * np.cos(phi_rad)
            fgrad = backend.from_numpy(np.array((gx, gy, gz)).astype(dt_data))
            parameters['FGRAD'] = fgrad
        elif ('ImageType' in grdUnit): 
            warn("Failed to retrieve field gradient coordinates (Unsupported ImageType in DSC file, must be '2D' or '3D')!")
        else:
            warn("Failed to retrieve field gradient coordinates (no ImageType in DSC file)!")
    
    # deal with squeeze & stack options
    if not stack:
        data = backend.moveaxis(data, (0, 1, 2), (1, 2, 0)).reshape(ny * nz, nx)
    if squeeze:
        data = data.squeeze()

    # deal with dtype option
    if (dtype is not None):
        data = backend.cast(data, dtype)
        if 'FGRAD' in parameters:
            parameters['FGRAD'] = backend.cast(parameters['FGRAD'], dtype)
                
    return B, data, parameters
