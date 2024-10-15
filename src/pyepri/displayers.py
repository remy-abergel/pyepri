"""This module provides tools for displaying 2D & 3D, mono & multi
source images, with possibility to update the displayed image at any
moment (useful in an iterative framework). The display
functionnalities of this modules are compatible with console and
interactive Python (IPython notebook) environments.

"""
import numpy as np
import functools
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import time
import types
import pyepri.checks as checks

__EMPTY_ARRAY__ = np.empty(0)

def is_notebook() -> bool:
    """Infer whether code is executed using IPython notebook or not."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        else:
            return False # Other type (?)
    except NameError:
        return False     # Probably standard Python interpreter

def init_display_monosrc_2d(u, figsize=None, time_sleep=0.01,
                            units=None, display_labels=False,
                            displayFcn=None, cmap=None, grids=None,
                            origin='lower', aspect=None,
                            is_notebook=False):
    """Initialize display for a single 2D image.
    
    Parameters
    ----------
    
    u : ndarray
        Two-dimensional array
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        taht ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).

    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    units : str, optional 
        Units associated to the X and Y axes (handling of different
        axes units is not provided).
    
    display_labels : bool, optional
        Set ``display_labels = True`` to display axes labels (including
        units when given).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 2D image ``u`` into another 2D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
    grids : sequence, optional
        A sequence (tuple or list) of two monodimensional ndarrays,
        such that grids[0] and grids[1] contain the sampling nodes
        associated to axes 0 (Y-axis) and 1 (X-axis) of the input
        array ``u``. 
    
        When given, the input grids are used to set the extent of the
        displayed image (see `matplotlib.imshow` documentation).
    
    origin : str in {'upper', 'lower'}, optional 
        Place the [0, 0] index of the array in the upper left or lower
        left corner of the Axes. When not given, the default setting
        is that of `matplotlib` (see key 'image.origin' of the
        matplotlib configuration parameter ``rcParams``).
    
    aspect : str in {'equal', 'auto'} or float or None, optional
        The aspect ratio of the Axes. This parameter is particularly
        relevant for images since it determines whether data pixels
        are square (see `matplotlib.imshow` documentation).
    
        When not given, the default setting is that of `matplotlib`
        (see key 'image.aspect' of the matplotlib configuration
        parameter ``rcParams``).
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    
    Return
    ------
    
    fg : <class 'matplotlib.image.AxesImage'>
        Produced image instance.
    
    
    See also
    --------
    
    update_display_monosrc_2d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
    
    # compute imshow extent (if grids are given)
    if grids is not None:
        xgrid, ygrid = grids[1], grids[0]
        if origin == 'lower':
            extent = (xgrid[0], xgrid[-1], ygrid[0], ygrid[-1])
        else:
            extent = (xgrid[0], xgrid[-1], ygrid[-1], ygrid[0])
    else:
        extent = None
    
    # draw image & retrieve figure number
    fg = plt.imshow(im, cmap=cmap, extent=extent, origin=origin, aspect=aspect)
    
    # update figsize (if needed)
    if figsize is not None:
        _fg_ = plt.gcf()
        _fg_.set_figwidth(figsize[0])
        _fg_.set_figheight(figsize[1])
    
    # display axes labels (if needed)
    if display_labels:
        xlab = 'X' if units is None else ('X (%s)' % units)
        ylab = 'Y' if units is None else ('Y (%s)' % units)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
    
    # pause and return
    if is_notebook:
        time.sleep(time_sleep)
    else:
        plt.pause(time_sleep)
    
    return fg

def update_display_monosrc_2d(u, fg, is_notebook=False, displayFcn=None, adjust_dynamic=True, time_sleep=0.01):
    """Update single 2D image display.
    
    Parameters
    ----------
    
    u : ndarray
        Two-dimensional array
    
    fg : <class 'matplotlib.image.AxesImage'>
        Image instance to be updated.
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 2D image ``u`` into another 2D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    adjust_dynamic : bool, optional
        Set ``adjust_dynamic = True`` to maximize the dynamic of the
        displayed image during the updating process, otherwise, set
        ``adjust_dynamic = False`` to keep the displayed dynamic
        unchanged.
        
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    
    Return
    ------

    None

    
    See also
    --------
    
    init_display_monosrc_2d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
    
    # draw image
    fg.set_data(im)
    
    # if needed, adjust dynamic
    if(adjust_dynamic):
        fg.set_clim(im.min(), im.max())
    
    # deal with interactive notebook running environments
    if is_notebook:
        display.clear_output(wait=True)
        display.display(pl.gcf())    
        time.sleep(time_sleep)
    
    return
        

def init_display_monosrc_3d(u, figsize=None, time_sleep=0.01,
                            units=None, display_labels=False,
                            displayFcn=None, cmap=None, grids=None,
                            origin='lower', aspect=None,
                            boundaries='auto', is_notebook=False,
                            indexes=None):
    """Initialize display for a single 3D image (display the three central slices of a 3D volume).
    
    Parameters
    ----------
    
    u : ndarray
        Three-dimensional array
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        taht ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    units : str, optional 
        Units associated to the X, Y & Z axes (handling of different
        axes units is not provided).
    
    display_labels : bool, optional
        Set ``display_labels = True`` to display axes labels (including
        units when given).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 3D image ``u`` into another 3D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
    grids : sequence, optional
        A sequence (tuple or list) of three monodimensional ndarrays,
        such that grids[0], grids[1] and grids[2] contain the sampling
        nodes associated to axes 0 (Y-axis), axe 1 (X-axis), and axe 2
        (Z-axis) of the input array ``u``.
    
        When given, the input grids are used to set the extent of the
        displayed slices (see `matplotlib.imshow` documentation).
    
    origin : str in {'upper', 'lower'}, optional 
        Place the [0, 0] index of the array in the upper left or lower
        left corner of the Axes. When not given, the default setting
        is that of `matplotlib` (see key 'image.origin' of the
        matplotlib configuration parameter ``rcParams``).
    
    aspect : str in {'equal', 'auto'} or float or None, optional
        The aspect ratio of the Axes. This parameter is particularly
        relevant for images since it determines whether data pixels
        are square (see `matplotlib.imshow` documentation).
    
        When not given, the default setting is that of `matplotlib`
        (see key 'image.aspect' of the matplotlib configuration
        parameter ``rcParams``).
    
    boundaries : str in {'auto', 'same'} 
        Use ``boundaries = 'same'`` to give all subplots the same axes
        boundaries (in particular, this ensures that all slice images
        will be displayed on the screen using the same pixel size).

        Otherwise, set ``boundaries = 'auto'`` to use tight extent for
        each displayed slice image.

    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    indexes : sequence of int, optional
        When given, indexes must be a sequence of three int,
        ``index[j] = (id0, id1, id2)``, such that `id0`, `id1` and
        `id2` correspond to the indexes used along each axis of the 3D
        volume to extract the slices to be displayed (using ``None``
        to keep a particular index to its default value is possible).
        
        The default setting is ``indexes = (u.shape[0]//2,
        u.shape[1]//2, u.shape[2]//2)``.
    
    
    Return
    ------
    
    fg : sequence of <class 'matplotlib.image.AxesImage'>
        Sequence of produced image instance (one instance per subplot)
    
    
    See also
    --------
    
    update_display_monosrc_3d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
   
    # retrieve slice images
    if indexes is not None:
        xc = im.shape[1]//2 if indexes[1] is None else indexes[1]
        yc = im.shape[0]//2 if indexes[0] is None else indexes[0]
        zc = im.shape[2]//2 if indexes[2] is None else indexes[2]
    else:
        xc = im.shape[1]//2
        yc = im.shape[0]//2
        zc = im.shape[2]//2
    im_01 = im[:, :, zc]
    im_02 = im[:, xc, :]
    im_12 = im[yc, :, :]

    # compute imshow extents (if grids are given)
    if grids is not None:
        xgrid, ygrid, zgrid = grids[1], grids[0], grids[2]
        extent_01 = (xgrid[0], xgrid[-1], ygrid[0], ygrid[-1])
        extent_02 = (zgrid[0], zgrid[-1], ygrid[0], ygrid[-1])
        extent_12 = (zgrid[0], zgrid[-1], xgrid[0], xgrid[-1])
        extents = (extent_01, extent_02, extent_12)
        if origin != 'lower':
            extents = tuple((t[0], t[1], t[-1], t[-2]) for t in extents)
        xc = xgrid[xc] # slice index is changed into its actual coordinate
        yc = ygrid[yc] # slice index is changed into its actual coordinate
        zc = zgrid[zc] # slice index is changed into its actual coordinate
    else:
        extents = (None, None, None)

    # prepare figure
    #fg, ax = plt.subplots(1,3)    
    
    # update figsize (if needed)
    if figsize is not None:
        FG = plt.gcf()
        FG.set_figwidth(figsize[0])
        FG.set_figheight(figsize[1])
    
    # display XY slice (Z = zc)
    plt.subplot(1,3,1)
    fg1 = plt.imshow(im_01, cmap=cmap, extent=extents[0],
                     origin=origin, aspect=aspect)
    plt.title("XY slice (Z=%g)" % zc)

    # display ZY slice (X = xc)
    plt.subplot(1,3,2)
    fg2 = plt.imshow(im_02, cmap=cmap, extent=extents[1],
                     origin=origin, aspect=aspect)
    plt.title("ZY slice (X=%g)" % xc)

    # display ZX slice (Y = yc)
    plt.subplot(1,3,3)
    fg3 = plt.imshow(im_12, cmap=cmap, extent=extents[2],
                     origin=origin, aspect=aspect)
    plt.title("ZX slice (Y=%g)" % yc)

    # display axes labels (if needed)
    if display_labels:
        xlab = 'X' if units is None else ('X (%s)' % units)
        ylab = 'Y' if units is None else ('Y (%s)' % units)
        zlab = 'Z' if units is None else ('Z (%s)' % units)
        fg1.axes.set_xlabel(xlab)
        fg1.axes.set_ylabel(ylab)
        fg2.axes.set_xlabel(zlab)
        fg2.axes.set_ylabel(ylab)
        fg3.axes.set_xlabel(zlab)
        fg3.axes.set_ylabel(xlab)
    
    # if same pixel size is needed, give to all subplots the same axes
    # boundaries
    if boundaries == 'same':
        if grids is not None:
            xlim = (min(xgrid[0], zgrid[0]), max(xgrid[-1], zgrid[-1])) 
            ylim = (min(xgrid[0], ygrid[0]), max(xgrid[-1], ygrid[-1]))
        else:
            ny, nz, nx = u.shape
            xmin = 0.
            xmax = nx - 1.
            ymin = 0.
            ymax = ny - 1.
            zmin = 0.
            zmax = nz - 1.
            xlim = (min(xmin, zmin), max(xmax, zmax))
            ylim = (min(xmin, ymin), max(xmax, ymax))
        if origin != 'lower':
            ylim = (ylim[-1], ylim[-2])
        fg1.axes.set_xlim(xlim)
        fg1.axes.set_ylim(ylim)
        fg2.axes.set_xlim(xlim)
        fg2.axes.set_ylim(ylim)
        fg3.axes.set_xlim(xlim)
        fg3.axes.set_ylim(ylim)

    # aggregate imshow handles
    fg = (fg1, fg2, fg3)

    # pause an return
    if is_notebook:
        time.sleep(time_sleep)
    else:
        plt.pause(time_sleep)
    
    return fg

def update_display_monosrc_3d(u, fg, is_notebook=False, displayFcn=None, adjust_dynamic=True, time_sleep=0.01, indexes=None):
    """Update single 3D image display.
    
    Parameters
    ----------
    
    u : ndarray
        Three-dimensional array
    
    fg : <class 'matplotlib.image.AxesImage'>
        Image instance to be updated.
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 3D image ``u`` into another 3D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    adjust_dynamic : bool, optional
        Set ``adjust_dynamic = True`` to maximize the dynamic of the
        displayed slices during the updating process (the displayed
        dynamic will be [min, max] where min and max denote the min
        and max values among the three displayed slices), otherwise,
        set ``adjust_dynamic = False`` to keep the displayed dynamic
        unchanged.
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    indexes : sequence of int, optional
        When given, indexes must be a sequence of three int,
        ``index[j] = (id0, id1, id2)``, such that `id0`, `id1` and
        `id2` correspond to the indexes used along each axis of the 3D
        volume to extract the slices to be displayed (using ``None``
        to keep a particular index to its default value is possible).
        
        The default setting is ``indexes = (u.shape[0]//2,
        u.shape[1]//2, u.shape[2]//2)``.
    
    
    Return
    ------

    None

    
    See also
    --------
    
    init_display_monosrc_3d

    """
    
    # retrieve slice images
    im = u if displayFcn is None else displayFcn(u)
    if indexes is not None:
        xc = im.shape[1]//2 if indexes[1] is None else indexes[1]
        yc = im.shape[0]//2 if indexes[0] is None else indexes[0]
        zc = im.shape[2]//2 if indexes[2] is None else indexes[2]
    else:
        xc = im.shape[1]//2
        yc = im.shape[0]//2
        zc = im.shape[2]//2
    im_01 = im[:, :, zc]
    im_02 = im[:, xc, :]
    im_12 = im[yc, :, :]
    
    # draw images
    fg[0].set_data(im_01)
    fg[1].set_data(im_02)
    fg[2].set_data(im_12)
    
    # if needed, adjust dynamics
    if(adjust_dynamic):
        cmin = min((im_01.min(), im_02.min(), im_12.min()))
        cmax = max((im_01.max(), im_02.max(), im_12.max()))
        fg[0].set_clim(cmin, cmax)
        fg[1].set_clim(cmin, cmax)
        fg[2].set_clim(cmin, cmax)
    
    # deal with interactive notebook running environments
    if is_notebook:
        display.clear_output(wait=True)
        display.display(pl.gcf())    
        time.sleep(time_sleep)
    #else:
    #    plt.pause(time_sleep)
    
    return

def init_display_multisrc_2d(u, figsize=None, time_sleep=0.01,
                             units=None, display_labels=False,
                             displayFcn=None, cmap=None, grids=None,
                             origin='lower', aspect=None,
                             boundaries='auto', is_notebook=False,
                             src_labels=None):
    """Initialize display for a sequence of 2D images.
    
    Parameters
    ----------
    
    u : sequence of ndarray
        The sequence (tuple or list) of two-dimensional images to be
        displayed.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        taht ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    units : str, optional 
        Units associated to the X & Y axes of the different source
        images (handling of different axes units is not provided).
    
    display_labels : bool, optional
        Set ``display_labels = True`` to display axes labels (including
        units when given).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 3D image ``u`` into another 3D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
    grids : sequence, optional
        A sequence with same length as ``u``, such that ``grids[j]``
        is a sequence containing two monodimensional arrays
        (``grids[j][0]`` and ``grids[j][1]``) corresponding to the
        sampling nodes associated to axes 0 (Y-axis), axe 1 (X-axis)
        of the `j-th` source image ``u[j]``.
        
        When given, the input grids are used to set the extent of the
        displayed source images (see `matplotlib.imshow`
        documentation).
    
    origin : str in {'upper', 'lower'}, optional 
        Place the [0, 0] index of the array in the upper left or lower
        left corner of the Axes. When not given, the default setting
        is that of `matplotlib` (see key 'image.origin' of the
        matplotlib configuration parameter ``rcParams``).
    
    aspect : str in {'equal', 'auto'} or float or None, optional
        The aspect ratio of the Axes. This parameter is particularly
        relevant for images since it determines whether data pixels
        are square (see `matplotlib.imshow` documentation).
    
        When not given, the default setting is that of `matplotlib`
        (see key 'image.aspect' of the matplotlib configuration
        parameter ``rcParams``).
    
    boundaries : str in {'auto', 'same'} 
        Use ``boundaries = 'same'`` to give all subplots the same axes
        boundaries (in particular, this ensures that all source images
        will be displayed on the screen using the same pixel size).
    
        Otherwise, set ``boundaries = 'auto'`` to use tight extent for
        each displayed slice image.
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).

    src_labels : sequence of str, optional 
        When given, src_label must be a sequence with same length as
        ``u`` such that ``src_labels[j]`` corresponds to the label of
        the j-th source ``u[j]`` (that is, a str to be added to the
        j-th source suptitle).
    
    
    Return
    ------
    
    fg : sequence of <class 'matplotlib.image.AxesImage'>
        Sequence of produced image instance (one instance per
        subplot).
    
    
    See also
    --------
    
    update_display_multisrc_2d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
    
    # retrieve number of sources 
    nsrc = len(im)
    
    # display figure
    if figsize is not None:
        FG = plt.gcf()
        FG.set_figwidth(figsize[0])
        FG.set_figheight(figsize[1])
        
    # compute imshow extents (if grids are given)
    if grids is not None:
        extents = tuple((grid[1][0], grid[1][-1], grid[0][0], grid[0][-1])
                        for grid in grids)
    else:
        extents = (None,)*nsrc
        
    # if needed compute maximal extent along each axis
    if boundaries == 'same':
        x0 = min(tuple(grid[1][0] for grid in grids))
        x1 = max(tuple(grid[1][-1] for grid in grids))
        y0 = min(tuple(grid[0][0] for grid in grids))
        y1 = max(tuple(grid[0][-1] for grid in grids))
        xlim = (x0, x1)
        ylim = (y0, y1)
        if origin != 'lower':
            ylim = (ylim[-1], ylim[-2])
                
    # display source images
    fg = ()
    for j in range(nsrc):
        
        # display image
        plt.subplot(1,nsrc,j+1)
        fg_j = plt.imshow(im[j], cmap=cmap, extent=extents[j],
                          origin=origin, aspect=aspect)
        
        # display title: source index + source label (if given)
        if src_labels is not None and src_labels[j] is not None:
            plt.title("source #%d (%s)" % (j, src_labels[j]))
        else:
            plt.title("source #%d" % j)
            
        # if needed, display labels
        if display_labels:
            xlab = 'X' if units is None else ('X (%s)' % units)
            ylab = 'Y' if units is None else ('Y (%s)' % units)
            fg_j.axes.set_xlabel(xlab)
            fg_j.axes.set_ylabel(ylab)
        
        # if same pixel size is needed, give to all subplots the same axes
        # boundaries
        if boundaries == 'same':
            fg_j.axes.set_xlim(xlim)
            fg_j.axes.set_ylim(ylim)
        
        # aggregate imshow handles
        fg += (fg_j,)
        
    # pause and return
    if is_notebook:
        time.sleep(time_sleep)
    else:
        plt.pause(time_sleep)
    
    return fg

def update_display_multisrc_2d(u, fg, is_notebook=False, displayFcn=None, adjust_dynamic=True, time_sleep=0.01):
    """Update display for a sequence of 2D images.
    
    Parameters
    ----------
    
    u : sequence of ndarray
        The sequence (tuple or list) of two-dimensional images to be
        displayed.
    
    fg : sequence of <class 'matplotlib.image.AxesImage'>
        The sequence of image instances to be updated.
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the sequence ``u`` into another sequence of 2D images (with
        same length). When `displayFcn` is given, the displayed image
        is ``im = displayFcn(u)`` instead of ``u``.
    
    adjust_dynamic : bool, optional
        Set ``adjust_dynamic = True`` to maximize the dynamic of the
        displayed sequence of images during the updating process (the
        displayed dynamic will be [min, max] where min and max denote
        the min and max values among all the images in ``u``),
        otherwise, set ``adjust_dynamic = False`` to keep the
        displayed dynamic unchanged.
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    
    Return
    ------
    
    None
    
    
    See also
    --------
    
    init_display_multisrc_2d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
    
    # draw images (with or without dynamic update)
    if adjust_dynamic:
        cmin = min(tuple(v.min() for v in im))
        cmax = max(tuple(v.max() for v in im))
        for j, v in enumerate(im):
            fg[j].set_data(v)
            fg[j].set_clim(cmin, cmax)
    else:
        for j, v in enumerate(im):
            fg[j].set_data(v)
    
    # deal with interactive notebook running environments
    if is_notebook:
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(time_sleep)
        
    return


def init_display_multisrc_3d(u, figsize=None, time_sleep=0.01,
                             units=None, display_labels=False,
                             displayFcn=None, cmap=None, grids=None,
                             origin='lower', aspect=None,
                             boundaries='auto', is_notebook=False,
                             indexes=None, src_labels=None):
    """Initialize display for a sequence of 3D images.
    
    Parameters
    ----------
    
    u : sequence of ndarray
        The sequence (tuple or list) of three-dimensional images to be
        displayed.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        taht ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    units : str, optional 
        Units associated to the X, Y & Z axes of the different source
        images (handling of different axes units is not provided).
    
    display_labels : bool, optional
        Set ``display_labels = True`` to display axes labels (including
        units when given).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 3D image ``u`` into another 3D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
    grids : sequence, optional
        A sequence with same length as ``u``, such that ``grids[j]``
        is a sequence containing three monodimensional arrays
        (``grids[j][0]``, ``grids[j][1]``, ``grids[j][2]``)
        corresponding to the sampling nodes associated to axes 0
        (Y-axis), axe 1 (X-axis) and axe 2 (Z-axis) of the `j-th`
        source image ``u[j]``.
        
        When given, the input grids are used to set the extent of the
        displayed source images (see `matplotlib.imshow`
        documentation).
    
    origin : str in {'upper', 'lower'}, optional 
        Place the [0, 0] index of the array in the upper left or lower
        left corner of the Axes. When not given, the default setting
        is that of `matplotlib` (see key 'image.origin' of the
        matplotlib configuration parameter ``rcParams``).
    
    aspect : str in {'equal', 'auto'} or float or None, optional
        The aspect ratio of the Axes. This parameter is particularly
        relevant for images since it determines whether data pixels
        are square (see `matplotlib.imshow` documentation).
    
        When not given, the default setting is that of `matplotlib`
        (see key 'image.aspect' of the matplotlib configuration
        parameter ``rcParams``).
    
    boundaries : str in {'auto', 'same'} 
        Use ``boundaries = 'same'`` to give all subplots the same axes
        boundaries (in particular, this ensures that all source images
        will be displayed on the screen using the same pixel size).
    
        Otherwise, set ``boundaries = 'auto'`` to use tight extent for
        each displayed slice image.
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    indexes : sequence, optional
        When given, indexes must be a sequence with lenght ``nsrc``
        such that ``indexes[j] = (id0, id1, id2)`` is a sequence of
        three indexes corresponding to the indexes used along each
        axis of the j-th source image ``u[j]`` to extract the slices
        to be displayed (using ``None`` to keep a particular index to
        its default value is possible).
        
        The default setting is ``indexes = [[im.shape[0]//2,
        u.shape[1]//2, u.shape[2]//2] for im in u]``.
    
    src_labels : sequence of str, optional 
        When given, src_label must be a sequence with same length as
        ``u`` such that ``src_labels[j]`` corresponds to the label of
        the j-th source ``u[j]`` (that is, a str to be added to the
        j-th source suptitle).
    
    Return
    ------
    
    fg : sequence of sequence of <class 'matplotlib.image.AxesImage'>
        A sequence with same lenght as ``u`` and such that ``fg[j]``
        is as sequence of three <class 'matplotlib.image.AxesImage'>
        corresponding to the image instances produced when displaying
        the three slices of ``u[j]``.
    
    
    See also
    --------
    
    update_display_multisrc_3d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
    
    # retrieve number of sources
    nsrc = len(im)
    
    # compute central slice indexes
    t = indexes is None
    xc = [v.shape[1]//2 if t or indexes[j][1] is None else
          indexes[j][1] for j, v in enumerate(im)]
    yc = [v.shape[0]//2 if t or indexes[j][0] is None else
          indexes[j][0] for j, v in enumerate(im)]
    zc = [v.shape[2]//2 if t or indexes[j][2] is None else
          indexes[j][2] for j, v in enumerate(im)]
    slices = tuple((v[:, :, zc[j]], v[:, xc[j], :], v[yc[j], :, :])
                   for j, v in enumerate(im))
    
    # compute imshow extents (if grids are given)
    if grids is not None:
        extents = ()
        for j, grid in enumerate(grids):
            # compute source extent
            xgrid, ygrid, zgrid = grid[1], grid[0], grid[2]
            extent_01 = (xgrid[0], xgrid[-1], ygrid[0], ygrid[-1])
            extent_02 = (zgrid[0], zgrid[-1], ygrid[0], ygrid[-1])
            extent_12 = (zgrid[0], zgrid[-1], xgrid[0], xgrid[-1])
            _extents = (extent_01, extent_02, extent_12)
            if origin != 'lower':
                _extents = tuple((t[0], t[1], t[-1], t[-2]) for t in
                                 _extents)
            extents += (_extents,)
            # change source slice indexes into actual coordinates
            xc[j] = xgrid[xc[j]]
            yc[j] = ygrid[yc[j]]
            zc[j] = zgrid[zc[j]]
    else:
        extents = ((None, None, None),)*nsrc
    
    # retrieve current figure instance
    _fg_ = plt.gcf()
    
    # update figsize (if needed)
    if figsize is not None:
        _fg_.set_figwidth(figsize[0])
        _fg_.set_figheight(figsize[1])
    
    # prepare subfigures
    _fg_.set_layout_engine('constrained')
    subfigs = _fg_.subfigures(nsrc, 1)
    
    # deal with case boundaries == 'same'
    if boundaries == 'same':
        x0 = min(tuple(min(g[1][0], g[2][0]) for g in grids))
        x1 = max(tuple(max(g[1][-1], g[2][-1]) for g in grids))
        y0 = min(tuple(min(g[1][0], g[0][0]) for g in grids))
        y1 = max(tuple(max(g[1][-1], g[0][-1]) for g in grids))
        xlim = (x0, x1)
        ylim = (y0, y1)
        if origin != 'lower':
            ylim = (ylim[-1], ylim[-2])
    
    # display subfigures
    fg = ()
    for j, v in enumerate(im):
        
        # retrieve slices
        v_01, v_02, v_12 = slices[j]
        
        # prepare subplots
        ax = subfigs[j].subplots(1, 3)
        
        # display XY slice (Z = zc)
        fg1 = ax[0].imshow(v_01, cmap=cmap, extent=extents[j][0],
                           origin=origin, aspect=aspect)
        ax[0].set_title("XY slice (Z=%g)" % zc[j])
        
        # display ZY slice (X = xc)
        fg2 = ax[1].imshow(v_02, cmap=cmap, extent=extents[j][1],
                           origin=origin, aspect=aspect)
        ax[1].set_title("ZY slice (X=%g)" % xc[j])
        
        # display ZX slice (Y = yc)
        fg3 = ax[2].imshow(v_12, cmap=cmap, extent=extents[j][2],
                           origin=origin, aspect=aspect)
        ax[2].set_title("ZX slice (Y=%g)" % yc[j])
        
        # display title: source index + source label (if given)
        if src_labels is not None and src_labels[j] is not None:
            src_lab = "source #%d (%s)" % (j, src_labels[j])
        else:
            src_lab = "source #%d" % j
        subfigs[j].suptitle(src_lab, weight='demibold')
        
        # display axes labels (if needed)
        if display_labels:
            xlab = 'X' if units is None else ('X (%s)' % units)
            ylab = 'Y' if units is None else ('Y (%s)' % units)
            zlab = 'Z' if units is None else ('Z (%s)' % units)
            ax[0].set_xlabel(xlab)
            ax[0].set_ylabel(ylab)
            ax[1].set_xlabel(zlab)
            ax[1].set_ylabel(ylab)
            ax[2].set_xlabel(zlab)
            ax[2].set_ylabel(xlab)
        
        # if same pixel size is needed, give to all subplots the same
        # axes boundaries
        if boundaries == 'same':
            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)
            ax[1].set_xlim(xlim)
            ax[1].set_ylim(ylim)
            ax[2].set_xlim(xlim)
            ax[2].set_ylim(ylim)
        
        # aggregate imshow handles
        fg += ((fg1, fg2, fg3),)
    
    # pause and return
    if is_notebook:
        time.sleep(time_sleep)
    else:
        plt.pause(time_sleep)
        
    return fg

def update_display_multisrc_3d(u, fg, is_notebook=False, displayFcn=None, adjust_dynamic=True, time_sleep=0.01, indexes=None):
    """Update display for a sequence of 3D images.
    
    Parameters
    ----------
    
    u : sequence of ndarray
        The sequence (tuple or list) of two-dimensional images to be
        displayed.
    
    fg : sequence of sequence of <class 'matplotlib.image.AxesImage'>
        The sequence of sequences of image instances to be updated
        (see ``update_display_multisrc_3d`` output).
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the sequence ``u`` into another sequence of 2D images (with
        same length). When `displayFcn` is given, the displayed image
        is ``im = displayFcn(u)`` instead of ``u``.
    
    adjust_dynamic : bool, optional
        Set ``adjust_dynamic = True`` to maximize the dynamic of the
        displayed sequence of images during the updating process (the
        displayed dynamic will be [min, max] where min and max denote
        the min and max values among all displayed slices computed
        from ``u``), otherwise, set ``adjust_dynamic = False`` to keep
        the displayed dynamic unchanged.
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    indexes : sequence, optional
        When given, indexes must be a sequence with lenght ``nsrc``
        such that ``indexes[j] = (id0, id1, id2)`` is a sequence of
        three indexes corresponding to the indexes used along each
        axis of the j-th source image ``u[j]`` to extract the slices
        to be displayed (using ``None`` to keep a particular index to
        its default value is possible).
        
        The default setting is ``indexes = [[im.shape[0]//2,
        u.shape[1]//2, u.shape[2]//2] for im in u]``.
    
    
    Return
    ------
    
    None
    
    
    See also
    --------
    
    init_display_multisrc_3d

    """
    
    # compute image to be displayed
    im = u if displayFcn is None else displayFcn(u)
    
    # extract slices
    if indexes is not None:
        xc = [v.shape[1]//2 if indexes[j][1] is None else
              indexes[j][1] for j, v in enumerate(im)]
        yc = [v.shape[0]//2 if indexes[j][0] is None else
              indexes[j][0] for j, v in enumerate(im)]
        zc = [v.shape[2]//2 if indexes[j][2] is None else
              indexes[j][2] for j, v in enumerate(im)]
    else:
        xc = [v.shape[1]//2 for v in im]
        yc = [v.shape[0]//2 for v in im]
        zc = [v.shape[2]//2 for v in im]
    slices = tuple((v[:, :, zc[j]], v[:, xc[j], :], v[yc[j], :, :])
                   for j, v in enumerate(im))

    # draw images (with or without dynamic update)
    if adjust_dynamic:
        cmin = min(tuple(vv.min() for v in slices for vv in v))
        cmax = max(tuple(vv.max() for v in slices for vv in v))
        for j in range(len(im)):
            fg[j][0].set_data(slices[j][0])
            fg[j][1].set_data(slices[j][1])
            fg[j][2].set_data(slices[j][2])
            fg[j][0].set_clim(cmin, cmax)
            fg[j][1].set_clim(cmin, cmax)
            fg[j][2].set_clim(cmin, cmax)
    else:
        for j in range(len(im)):
            fg[j][0].set_data(slices[j][0])
            fg[j][1].set_data(slices[j][1])
            fg[j][2].set_data(slices[j][2])
    
    # pause and return
    if is_notebook:
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(time_sleep)
    #else:
    #    plt.pause(time_sleep)
        
    return


def create_2d_displayer(nsrc=1, figsize=None, displayFcn=None,
                        time_sleep=0.01, units=None,
                        adjust_dynamic=True, display_labels=False,
                        cmap=None, grids=None, origin='lower',
                        aspect=None, boundaries='auto', indexes=None,
                        src_labels=None):
    """Instantiate a single 2D image displayer.

    This function instantiate a ``pyepri.Displayer`` class instance
    using ndim=3 and passing all the other args & kwargs to the
    ``pyepri.displayers.Displayer`` default constructor (type
    ``help(pyepri.displayers)`` for more details).

    """
    ndim = 2
    return Displayer(nsrc, ndim, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     units=units, adjust_dynamic=adjust_dynamic,
                     display_labels=display_labels, cmap=cmap,
                     grids=grids, origin=origin, aspect=aspect,
                     boundaries=boundaries, indexes=indexes,
                     src_labels=src_labels)

def create_3d_displayer(nsrc=1, figsize=None, displayFcn=None,
                        time_sleep=0.01, units=None, extents=None,
                        adjust_dynamic=True, display_labels=False,
                        cmap=None, grids=None, origin='lower',
                        aspect=None, boundaries='auto',
                        indexes=None, src_labels=None):
    """Instantiate a single 3D image displayer.

    This function instantiate a ``pyepri.Displayer`` class instance
    using ndim=3 and passing all the other args & kwargs to the
    ``pyepri.displayers.Displayer`` default constructor (type
    ``help(pyepri.displayers)`` for more details).

    """
    ndim = 3
    return Displayer(nsrc, ndim, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     units=units, adjust_dynamic=adjust_dynamic,
                     display_labels=display_labels, cmap=cmap,
                     grids=grids, origin=origin, aspect=aspect,
                     boundaries=boundaries, indexes=indexes,
                     src_labels=src_labels)

def create(u, figsize=None, displayFcn=None, time_sleep=0.01,
           units=None, extents=None, adjust_dynamic=True,
           display_labels=False, cmap=None, grids=None,
           origin='lower', aspect=None, boundaries='auto',
           indexes=None, src_labels=None):
    """Instantiate a Displayer object suited to the input parameter.
    
    This function instantiate a ``pyepri.Displayer`` class instance
    using ``nsrc`` and ``ndim`` values inferred from ``u`` and passing
    all the other args & kwargs to the ``pyepri.displayers.Displayer``
    default constructor (type ``help(pyepri.displayers)`` for more
    details).

    """
    # check consistency for parameter u (other parameters will be
    # tested during the pyepri.displayers.Displayer object
    # instanciation)
    _check_inputs_(u=u)

    # retrieve number of sources (nsrc) and dimensions (ndim)
    if isinstance(u, (tuple, list)):
        nsrc = len(u)
        ndim = u[0].ndim
        force_multisrc = nsrc == 1
    else:
        nsrc = 1
        ndim = u.ndim
        force_multisrc = False

    # create & return Displayer object instance
    return Displayer(nsrc, ndim, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     units=units, adjust_dynamic=adjust_dynamic,
                     display_labels=display_labels, cmap=cmap,
                     grids=grids, origin=origin, aspect=aspect,
                     boundaries=boundaries,
                     force_multisrc=force_multisrc)

def get_number(fg):
    """Retrieve displayed figure number.
    
    Parameters
    ----------
    
    fg : <class 'matplotlib.image.AxesImage'> or sequence of <class \
    'matplotlib.image.AxesImage'>
        Image instance or sequence of image instances that belond to
        the same figure.

    Return
    ------
    
    fgnum : int 
        Figure number. 
    
    """
    
    if isinstance(fg,  (tuple, list)):
        if isinstance(fg[0], (tuple, list)):
            fgnum = fg[0][0].get_figure().get_figure().number
        else:
            fgnum = fg[0].get_figure().number
    else:
        fgnum = fg.get_figure().number
    
    return fgnum

def _check_inputs_(nsrc=None, ndim=None, displayFcn=None,
                   time_sleep=None, units=None, adjust_dynamic=None,
                   display_labels=None, cmap=None, grids=None,
                   origin=None, aspect=None, boundaries=None,
                   u=__EMPTY_ARRAY__, figsize=None, indexes=None,
                   src_labels=None):
    """Factorized consistency checks for functions in this :py:mod:`pyepri.displayers` submodule.

    """

    # type checks
    checks._check_type_(int, nsrc=nsrc, ndim=ndim)
    checks._check_type_(float, time_sleep=time_sleep)
    checks._check_type_(bool, adjust_dynamic=adjust_dynamic, display_labels=display_labels)
    checks._check_type_(str, units=units, cmap=cmap, origin=origin, aspect=aspect, boundaries=boundaries)
    checks._check_type_(types.FunctionType, displayFcn=displayFcn)
    
    # custom checks
    if cmap is not None and cmap not in plt.colormaps():
        raise ValueError(
            "Parameter `cmap` must be `None` or one of %s" % plt.colormaps()
        )
    if origin is not None and origin not in {'lower', 'upper'}:
        raise ValueError(
            "Parameter `origin` must be `None` or one of {'lower', 'upper'}"
        )
    if aspect is not None and aspect not in {'equal', 'auto'}:
        raise ValueError(
            "Parameter `aspect` must be `None` or one of {'equal', 'auto'}"
        )
    if boundaries is not None and boundaries not in {'auto', 'same'}:
        raise ValueError(
            "Parameter `boundaries` must be `None` or one of {'auto', 'same'}"
        )
    if figsize is not None:
        checks._check_seq_(t=float, n=2, figsize=figsize)
    if grids is not None:
        if not isinstance(grids, (tuple, list)):
            raise RuntimeError(        
                "Parameter `grids` must be a tuple or a list"
            )
        if nsrc == 1:
            if len(grids) != ndim:
                raise RuntimeError(
                    "For single source display, parameter `grids` must satisfy ``len(grids) == ndim``"
                )
            if not all((isinstance(g, np.ndarray) and g.ndim == 1 for g in grids)):
                raise RuntimeError(
                    "For single source display, all elements `g` in grids must be monodimensional arrays"
                )
        else:
            if len(grids) != nsrc:
                raise RuntimeError(
                    "For multiple sources display, parameter `grids` must satisfy ``len(grids) == nsrc``"
                )
            elif not all(len(g) == ndim for g in grids):
                raise RuntimeError(
                    "For multiple sources display, all elements `g` in `grids` must satisfy ``len(g) == ndim``"
                )
            if not all(tuple(isinstance(gg, np.ndarray) and gg.ndim == 1 for g in grids for gg in g)):
                raise RuntimeError(
                    "For multiple source display, for all `g` in `grids`, all elements of `g` must be monodimensional arrays"
                )
    monosrc = isinstance(u, np.ndarray)
    multisrc = isinstance(u, (tuple, list)) and all(isinstance(v, np.ndarray) for v in u)
    if not monosrc and not multisrc:
        raise RuntimeError(
            "Parameter `u` must be either a `ndarray` or a sequence of `ndarray`"
        )
    if indexes is not None:
        if 1 == nsrc:
            checks._check_seq_(t=int, n=ndim, indexes=indexes)
        else:
            checks._check_seq_of_seq_(t=int, len0=nsrc, len1=ndim, indexes=indexes)            
    if src_labels is not None:
        checks._check_seq_(t=str, n=nsrc, src_labels=src_labels)
    
    return True


class Displayer:
    """Class for display and update of different kind of images, in different running environments.
    
    Supported images
    ----------------
    
    + single 2D image : the input signal is a two-dimensional array
    
    + single 3D image : the input signal is a three-dimensional array
    
    + multisources 2D images : the input signal is a sequence of \
      two-dimensional arrays (each array being called a `source`)
    
    + multisources 3D images : the input signal is a sequence of \
      three-dimensional arrays (each array being called a `source`)
    
    
    Displaying rules 
    ----------------
    
    + single 2D image : the image is displayed using \
      `matplotlib.imshow`
    
    + single 3D image : the three central slices (along each axis) of \
      the image are drawn using `matplotlib.imshow` into a single row \ 
      of subplots.
    
    + multisources 2D images : the source images are drawn using \ 
      `matplotlib.imshow` into a single row of subplots.
    
    + multisources 3D images : each source image is represented using
      a row of subplots. Each row contains the three central slices of
      the considered source image.
    
    In all situations described above, several display customization
    are proposed (axes labels, axes boundaries, colormap, aspect, ...)
    through the kwargs of the default constructor.

    
    Class attributes
    ----------------
    
    init_display : <class 'function'>
        Function with prototype ``fg = init_display(u)`` that can be
        used to draw the input image ``u`` according to the rules
        described above. The returned ``fg`` is the produced image
        instance (when u is a single 2D image) or a sequence of image
        instances (when u is a single 3D image or a multisources 2D or
        3D image) corresponding to the image instances of each
        produced subplot.
    
    update_display : <class 'function'> 
        Function with prototype ``None = update_display(u, fg)`` that
        can be used to replace the image displayed in ``fg`` (the
        ouptut of the ``init_display`` attribute described above) by
        ``u``.

    get_number : <class 'function'>
        Function with prototype ``fgnum = get_number(fg)`` that return
        the figure number from the output of the ``init_display``
        attribute described above.
    
    title : <class 'function'> 
        Function with prototype ``None = title(str)`` that can be used
        to update the title (or suptitle when subplots are used) of
        the current figure.

    notebook : bool 
        A bool that specified whether the detected environment is an
        interactive notebook environments (``notebook = True``) or not
        (``notebook = False``)
    
    pause : <class 'function'>
        Function with prototype ``None = pause(t=time_sleep)`` used to
        pause (or sleep in interactive python environment) during of
        ``t`` seconds, the default value of ``time_sleep`` is defined
        during the ``pyepri.displayers.Display`` object instanciation.
    
    clear_output : <class 'function'>
        Function with prototype ``None = clear_output()`` used to
        clear the currently displayed image within an interactive
        notebook running environment.

    """
    
    def __init__(self, nsrc, ndim, figsize=None, displayFcn=None,
                 time_sleep=0.01, units=None, adjust_dynamic=True,
                 display_labels=False, cmap=None, grids=None,
                 origin='lower', aspect=None, boundaries='auto',
                 force_multisrc=False, indexes=None, src_labels=None):
        """Default constructor for ``pyepri.displayers.Displayer`` objects instanciation.
        
        
        Parameters
        ----------
        
        nsrc : int
            Number of source images to be displayed (must be >= 1).
        
        ndim : int in {1, 2, 3}
            Dimensions of the source images to be displayed.
        
        figsize : (float, float), optional
            When given, figsize must be a tuple with length two and
            such taht ``figsize[0]`` and ``figsize[1]`` are the width
            and height in inches of the figure to be displayed. When
            not given, the default setting is that of `matplotlib`
            (see key 'figure.figsize' of the matplotlib configuration
            parameter ``rcParams``).
        
        displayFcn : <class 'function'>, optional 
            Function with prototype ``im = displayFcn(v)`` that can
            change any source image ``v in u`` into another image with
            same number of dimensions (``im.ndim = v.ndim``). When
            `displayFcn` is given, the displayed source images will be
            ``(displayFcn(v) for v in u)`` instead of ``u``.
        
        time_sleep : float, optional 
            Duration in seconds of pause or sleep (depending on the
            running environment) to perform after image drawing.
        
        units : str, optional 
            Units associated to image(s) axes (the same unit will be
            use for all axes, the handling of different units is not
            provided).
        
        adjust_dynamic : bool, optional
            Set ``adjust_dynamic = True`` to maximize the dynamic of
            the displayed image during each update process, otherwise,
            set ``adjust_dynamic = False`` to keep the displayed
            dynamic unchanged.
        
        display_labels : bool, optional
            Set ``display_labels = True`` to display axes labels
            (including units when given).
        
        cmap : str, optional
            The registered colormap name used to map scalar data to
            colors in `matplotlib.imshow`.
        
        grids : sequence, optional
            A sequence (tuple or list) of sequence such that
            ``grids[i][j]`` is a monodimensional array containing the
            sampling nodes associated to the j-th axe of the i-th
            source image.
        
            When given, the input grids are used to set the extent of
            the displayed images (see `matplotlib.imshow`
            documentation).
        
        origin : str in {'upper', 'lower'}, optional 
            Place the [0, 0] index of the array in the upper left or
            lower left corner of the Axes. When not given, the default
            setting is that of `matplotlib` (see key 'image.origin' of
            the matplotlib configuration parameter ``rcParams``).
        
        aspect : str in {'equal', 'auto'} or float or None, optional
            The aspect ratio of the Axes. This parameter is
            particularly relevant for images since it determines
            whether data pixels are square (see `matplotlib.imshow`
            documentation).
        
            When not given, the default setting is that of
            `matplotlib` (see key 'image.aspect' of the matplotlib
            configuration parameter ``rcParams``).
        
        boundaries : str in {'auto', 'same'} 
            This parameter is only used when nsrc > 1 or ndim > 2. Use
            ``boundaries = 'same'`` to give all subplots the same axes
            boundaries (in particular, this ensures that all slice
            images will be displayed on the screen using the same
            pixel size).
        
            Otherwise, set ``boundaries = 'auto'`` to use tight extent
            for each displayed slice image.
        
        force_multisrc : bool, optional
            Force instanciation of a multi-source displayer (useful
            when, for some reasons, the user want to consider a
            multi-source framework with only one source, in this case,
            the source is not stored as an array but as a tuple
            containing a unique array).
        
        indexes : sequence, optional
            Used for 3D (monosrc or multisrc) displayers only. When
            given, indexes must be:
        
            + when ``nsrc == 1``: a sequence of three int, ``indexes =
              (id0, id1, id2)`` such that `id0`, `id1` and `id2`
              correspond to the indexes used along each axis of the 3D
              volume to extract the slices to be displayed (using
              ``None`` to keep a particular index to its default value
              is possible). The default setting in this situation is
              ``indexes = (u.shape[0]//2, u.shape[1]//2,
              u.shape[2]//2)``;

            + when ``nsrc > 1``: a sequence with lenght ``nsrc`` such
              that ``indexes[j] = (id0, id1, id2)`` is a sequence of
              three indexes corresponding to the indexes used along
              each axis of the j-th 3D source image ``u[j]`` to
              extract the slices to be displayed (using ``None`` to
              keep a particular index to its default value is
              possible). The default setting is ``indexes =
              [[im.shape[0]//2, u.shape[1]//2, u.shape[2]//2] for im
              in u]``.
        
        src_labels : sequence of str, optional
            Used for multisrc (2D or 3D) displayers only. When given,
            src_label must be a sequence with length ``nsrc`` such
            that ``src_labels[j]`` corresponds to the label of the
            j-th source (a str to be added to the j-th source
            suptitle).
        
        Return
        ------
        
        displayer : <class 'pyepri.displayers.Displayer'>
        
        
        See also
        --------
        
        create_2d_displayer
        create_3d_displayer
        create

        """
        
        # check consistency
        _check_inputs_(nsrc=nsrc, ndim=ndim, displayFcn=displayFcn,
                       time_sleep=time_sleep, units=units,
                       adjust_dynamic=adjust_dynamic,
                       display_labels=display_labels, cmap=cmap,
                       grids=grids, origin=origin, aspect=aspect,
                       boundaries=boundaries, figsize=figsize,
                       indexes=indexes, src_labels=src_labels)
        
        # configure display libraries according to the running
        # environment
        if is_notebook():
            self.notebook = True
            get_ipython().run_line_magic('matplotlib', 'inline')
            self.pause = lambda time_sleep=time_sleep : time.sleep(time_sleep)
            self.pause.__doc__ = "return time.sleep(time_sleep)"
        else:
            self.notebook = False
            self.pause = lambda time_sleep=time_sleep : plt.pause(time_sleep)
            self.pause.__doc__ = "return plt.pause(time_sleep)"
            plt.ion()
        
        # fill attributes
        self.clear_output = lambda wait=True : display.clear_output(wait=wait)
        self.clear_output.__doc__ = "return display.clear_output(wait=wait)"
        self.get_number = get_number
        
        # deal with title attribute (plt.title for monosource 2D
        # image, plt.suptitle otherwise)
        if nsrc == 1 and ndim == 2: 
            self.title = plt.title
        else:
            self.title = plt.suptitle
            
        # configure init_display and update_display attribute
        if nsrc == 1 and not force_multisrc: # monosrc
            if ndim == 2:
                self.init_display = \
                functools.partial(init_display_monosrc_2d,
                                  figsize=figsize,
                                  displayFcn=displayFcn,
                                  time_sleep=time_sleep, units=units,
                                  display_labels=display_labels,
                                  cmap=cmap, grids=grids,
                                  origin=origin, aspect=aspect,
                                  is_notebook=self.notebook)
                self.update_display = \
                functools.partial(update_display_monosrc_2d,
                                  is_notebook=self.notebook,
                                  displayFcn=displayFcn,
                                  adjust_dynamic=adjust_dynamic,
                                  time_sleep=time_sleep)
            elif ndim == 3:
                self.init_display = \
                functools.partial(init_display_monosrc_3d,
                                  figsize=figsize,
                                  displayFcn=displayFcn,
                                  time_sleep=time_sleep, units=units,
                                  display_labels=display_labels,
                                  cmap=cmap, grids=grids,
                                  origin=origin, aspect=aspect,
                                  boundaries=boundaries,
                                  indexes=indexes,
                                  is_notebook=self.notebook)
                self.update_display = \
                functools.partial(update_display_monosrc_3d,
                                  is_notebook=self.notebook,
                                  displayFcn=displayFcn,
                                  adjust_dynamic=adjust_dynamic,
                                  indexes=indexes,
                                  time_sleep=time_sleep)
        else: # multisrc
            if ndim == 2:
                self.init_display = \
                functools.partial(init_display_multisrc_2d,
                                  figsize=figsize,
                                  displayFcn=displayFcn,
                                  time_sleep=time_sleep, units=units,
                                  display_labels=display_labels,
                                  boundaries=boundaries,
                                  cmap=cmap, grids=grids,
                                  origin=origin, aspect=aspect,
                                  is_notebook=self.notebook,
                                  src_labels=src_labels)
                self.update_display = \
                functools.partial(update_display_multisrc_2d,
                                  is_notebook=self.notebook,
                                  displayFcn=displayFcn,
                                  adjust_dynamic=adjust_dynamic,
                                  time_sleep=time_sleep)
            elif ndim == 3:
                self.init_display = \
                functools.partial(init_display_multisrc_3d,
                                  figsize=figsize,
                                  displayFcn=displayFcn,
                                  time_sleep=time_sleep, units=units,
                                  display_labels=display_labels,
                                  cmap=cmap, grids=grids,
                                  origin=origin, aspect=aspect,
                                  boundaries=boundaries,
                                  indexes=indexes,
                                  is_notebook=self.notebook,
                                  src_labels=src_labels)
                self.update_display = \
                functools.partial(update_display_multisrc_3d,
                                  is_notebook=self.notebook,
                                  displayFcn=displayFcn,
                                  adjust_dynamic=adjust_dynamic,
                                  indexes=indexes,
                                  time_sleep=time_sleep)
