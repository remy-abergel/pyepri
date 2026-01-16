"""This module provides static & interactive tools for image display.

Interactive image displayers
----------------------------

Tools for displaying images (3D images & 4D spectral-spatial images)
and interacting with the display through mouse and keyboard
interactive commands (interactions are not available using notebooks).


Static (but updatable) image displayers
---------------------------------------

Static image displayers can be used to display different kind of
images (mono & multisource 2D & 3D images, spectral-spatial 4D images)
in different execution environments (console & notebooks).

They come with the possibility to update the displayed image at any
moment (useful in an iterative framework).

"""
import math
import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
import pyvista as pv
from IPython import display, get_ipython
import time
import types
import pyepri.checks as checks
import pyepri.utils as utils

__EMPTY_ARRAY__ = np.empty(0)

def is_notebook() -> bool:
    """Infer whether code is executed using IPython notebook or not."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        elif 'google.colab' in str(get_ipython()): # running on Google Colab
            return True
        else:
            return False # Other type (?)
    except NameError:
        return False     # Probably standard Python interpreter

    
def isosurf3d(u, isovalue=None, color='#f7fe00', cmap=None,
              show_grid=True, xlabel='X',
              ylabel='Y', zlabel='Z', xlim=None, ylim=None, zlim=None,
              config=None):
    """Interactive isosurface displayer for 3D images.
    
    Parameters
    ----------
    
    u : ndarray
        Three dimensional array containing the values of the
        3D image ordered as follows:
        
        + axis 0 = spatial vertical axis (or Y-axis);
        + axis 1 = spatial horizontal axis (or X-axis);
        + axis 2 = saptial depth axis (or Z-axis).
    
    isovalue : float, optional
        Isovalue to display (default setting corresponds to Otsu's
        threshold which maximizes the inter-class variance between
        background and foreground).
    
    color : str or RBG tuple or RGBA tuple, optional    
        Isosurface color, provided in PyVista compatible format, see
        several examples below:
        
        + predefined color names : "red", "green", "blue", "yellow",
          "magenta", "cyan", "orange", "purple", "black", "white", ...
        
        + hexadecimal color: '#F7FE00', "#FF5733", ...
        
        + RGB tuple : for instance (1.0, 0.5, 0.0)
        
        + RGBA tuple : for instance (0.2, 0.4, 0.6, 0.8)
        
        See PyVista documentation for more details.
    
    cmap : str or pyvista.LookupTable, optional
    
        Colormap to apply when rendering scalar values on the slices.
        
        - If `None`, the default colormap is used.
        - Can be any string recognized by Matplotlib, e.g., 'viridis',
          'jet', 'coolwarm', ...    
        - Can also be a `pyvista.LookupTable` for custom color mapping.
    
    show_grid : bool, optional
        Grid visibility at startup (toogle visibility using key 'g')
    
    xlabel : str, optional
        Label for the X axis
    
    ylabel : str, optional
        Label for the Y axis
    
    zlabel : str, optional
        Label for the Z axis
    
    xlim : tuple of float, optional
        Limits for the x-axis as a tuple ``(xmin, xmax)``. If
        provided, sets the visible range of the X-axis. If None, the
        limits are determined automatically based on the data.
    
    ylim : tuple of float, optional
        Limits for the x-axis as a tuple ``(ymin, ymax)``. If
        provided, sets the visible range of the Y-axis. If None, the
        limits are determined automatically based on the data.
    
    zlim : tuple of float, optional
        Limits for the x-axis as a tuple ``(zmin, zmax)``. If
        provided, sets the visible range of the Z-axis. If None, the
        limits are determined automatically based on the data.
    
    config : dict, optional        
        Dictionary containing custom settings. The following keys may
        be used:
        
        + 'surface_opacity' : float, initial value for the isosurface
          opacity slider
        
        + 'slices_opacity' : float, initial value for the slices
          opacity slider
        
        + 'step1' : float, proportion of the slider range to use at
          slider increment/decrement when the keys 'right' and 'left'
          are pressed
        
        + 'step2' : float, proportion of the slider range to use at
          slider increment/decrement when the keys 'ctrl+right' and
          'ctrl+left' are pressed
        
        Specifying only a subset of them is allowed.
    
    
    Return
    ------
    
    plotter : pyvista.Plotter
        The plotter instance containing the rendered scene.
    
    """
    
    # retrieve config if given, or set default config otherwise
    surface_opacity = slices_opacity = step1 = step2 = None    
    if config is not None:
        surface_opacity = config.get('surface_opacity')
        slices_opacity = config.get('surface_opacity')
        step1 = config.get('step1')
        step2 = config.get('step2')
    surface_opacity = 1 if surface_opacity is None else surface_opacity
    slices_opacity = 1 if slices_opacity is None else slices_opacity
    step1 = .05 if step1 is None else step1
    step2 = .1 if step2 is None else step2
    
    # prepare PyVista plotter
    grid = pv.ImageData()
    grid.dimensions = u.shape
    grid.point_data["values"] = u.flatten(order="F")
    plotter = pv.Plotter()
    
    # compute initial isosurface
    isoval = utils.otsu_threshold(u) if isovalue is None else isovalue
    contour = grid.contour([isoval])
    surface_actor = plotter.add_mesh(contour, name="isosurface",
                                     opacity=surface_opacity,
                                     color=color)
    
    # add slices (recall that dimensions (0, 1, 2) represent the (Y, X, Z) axes)
    n0, n1, n2 = grid.dimensions
    slice_0 = grid.slice(normal='x', origin=(n0//2, 0, 0))
    slice_1 = grid.slice(normal='y', origin=(0, n1//2, 0))
    slice_2 = grid.slice(normal='z', origin=(0, 0, n2//2))    
    slice_x_actor = plotter.add_mesh(slice_1, name="slice_x",
                                     opacity=slices_opacity,
                                     cmap=cmap)
    slice_y_actor = plotter.add_mesh(slice_0, name="slice_y",
                                     opacity=slices_opacity,
                                     cmap=cmap)
    slice_z_actor = plotter.add_mesh(slice_2, name="slice_z",
                                     opacity=slices_opacity,
                                     cmap=cmap)
    
    # compute grid
    labels = dict(xtitle=ylabel, ytitle=xlabel, ztitle=zlabel)
    xlim = [0, n1 - 1]
    ylim = [0, n0 - 1]
    zlim = [0, n2 - 1]
    bounds = [*ylim, *xlim, *zlim]
    ax = plotter.show_grid(**labels, bounds=bounds)
    ax.SetVisibility(show_grid)
    
    # camera position
    cam = plotter.camera
    cam.focal_point = (
        cam.focal_point[0] + .25 * n0,
        cam.focal_point[1],
        cam.focal_point[2]
    )
    
    # slider callbacks    
    def update_iso(value):
        contour = grid.contour([value])
        surface_actor.mapper.SetInputData(contour)
        plotter.render()
    
    def update_opacity(value, actors_list):
        for actor in actors_list:
            actor.GetProperty().SetOpacity(value)
        plotter.render()
    
    def update_slice_y(value):
        sl = grid.slice(normal='x', origin=(value, 0, 0))
        slice_x_actor.mapper.SetInputData(sl)
        plotter.render()
    
    def update_slice_x(value):
        sl = grid.slice(normal='y', origin=(0, value, 0))
        slice_y_actor.mapper.SetInputData(sl)
        plotter.render()
    
    def update_slice_z(value):
        sl = grid.slice(normal='z', origin=(0, 0, value))
        slice_z_actor.mapper.SetInputData(sl)
        plotter.render()
    
    def set_slider_color(slider, rgb):
        slider.GetRepresentation().GetTitleProperty().SetColor(*rgb)
        slider.GetRepresentation().GetTubeProperty().SetColor(*rgb)
    
    # isosurface slider
    iso_rng = [u.min(), u.max()]
    slider_iso = plotter.add_slider_widget(
        callback=update_iso,
        rng=iso_rng,
        value=isovalue,
        title="isovalue",
        pointa=(.01, .9),
        pointb=(.2, .9),
        color="black",
    )
    set_slider_color(slider_iso, (1, 0, 0))
    slider_iso.GetRepresentation().GetTitleProperty().SetFontSize(1)
    
    # isosurface opacity slider
    slider_isosurf_opacity = plotter.add_slider_widget(
        callback=lambda value: update_opacity(value, [surface_actor]),
        rng=[0, 1],
        value=1,
        title="surface opacity",
        pointa=(.01, .76),
        pointb=(.2, .76),
        color="black",
    )
    
    # X slice slider
    slider_x_slice = plotter.add_slider_widget(
        callback=update_slice_x,
        rng=[0, n1 - 1],
        value=n1//2,
        title="slice X",
        pointa=(0.01, 0.58),
        pointb=(0.2, 0.58)
    )
    
    # Y slice slider
    slider_y_slice = plotter.add_slider_widget(
        callback=update_slice_y,
        rng=[0, n0-1],
        value=n0//2,
        title="slice Y",
        pointa=(0.01, 0.44),
        pointb=(0.2, 0.44)
    )
    
    # Y slice slider
    slider_z_slice = plotter.add_slider_widget(
        callback=update_slice_z,
        rng=[0, n2-1],
        value=n2//2,
        title="slice Z",
        pointa=(0.01, 0.3),
        pointb=(0.2, 0.3)
    )
    
    # slices opacity slider
    slider_slices_opacity = plotter.add_slider_widget(
        callback=lambda value: update_opacity(value, [slice_x_actor, slice_y_actor, slice_z_actor]),
        rng=[0, 1],
        value=1,
        title="slices opacity",
        pointa=(0.01, 0.16),
        pointb=(0.2, 0.16)
    )

    # deal with slider cycling
    sliders = [slider_iso, slider_isosurf_opacity, slider_x_slice,
               slider_y_slice, slider_z_slice,
               slider_slices_opacity]
    
    sliders_callback = [
        update_iso,
        lambda value: update_opacity(value, [surface_actor]),
        update_slice_x, update_slice_y,
        update_slice_z,
        lambda value: update_opacity(value, [slice_x_actor, slice_y_actor, slice_z_actor])
    ]
    params = {'id': 0}
    
    # keyboard callbacks
    interactor = plotter.iren.interactor
    
    def on_left_arrow():
        rep = sliders[params['id']].GetRepresentation()
        rng = (rep.GetMinimumValue(), rep.GetMaximumValue())
        value = rep.GetValue()
        ctrl = interactor.GetControlKey()
        step = step1 if ctrl == 0 else step2
        new_val = max(rng[0], value - step * (rng[1] - rng[0]))
        rep.SetValue(new_val)
        sliders_callback[params['id']](new_val)
    
    def on_right_arrow():
        rep = sliders[params['id']].GetRepresentation()
        rng = (rep.GetMinimumValue(), rep.GetMaximumValue())
        value = rep.GetValue()
        ctrl = interactor.GetControlKey()
        step = step1 if ctrl == 0 else step2
        new_val = min(rng[1], value + step * (rng[1] - rng[0]))
        rep.SetValue(new_val)
        sliders_callback[params['id']](new_val)
    
    def change_slider():
        
        # set current slider color to black
        set_slider_color(sliders[params['id']], (0, 0, 0))
        
        # cycle slider
        ctrl = interactor.GetControlKey()
        step = 1 if ctrl == 0 else -1
        params['id'] += step
        params['id'] = params['id'] % len(sliders)
        
        # set new slider color to red
        set_slider_color(sliders[params['id']], (1, 0, 0))
        
        # refresh rendering
        plotter.render()
    
    def toogle_actor_visibility(actor, refresh=True):
        actor.SetVisibility(not actor.GetVisibility())
        if refresh:
            plotter.render()
    
    def toogle_slider_visibility(slider):
        slider.GetRepresentation().SetVisibility(not slider.GetRepresentation().GetVisibility())
    
    def on_c_pressed(ctrl=None):
        ctrl = interactor.GetControlKey() if ctrl is None else ctrl
        if ctrl == 0 : # show/hide slices
            toogle_actor_visibility(slice_x_actor, refresh=False)
            toogle_actor_visibility(slice_y_actor, refresh=False)
            toogle_actor_visibility(slice_z_actor, refresh=True)
        else: # show/hide isosurface
            toogle_actor_visibility(surface_actor, refresh=True)
    
    def on_s_pressed():
        ctrl = interactor.GetControlKey()
        if ctrl == 1: # screenshot ? (not implemented yet)
            pass
        else:
            for s in sliders:
                toogle_slider_visibility(s)
            plotter.render()
    
    def on_h_pressed():
        print("")
        print("Interactive controls (3D isosurface displayer)")
        print("==============================================\n")
        print("Mouse")
        print("-----\n")
        print("  - Wheel up : zoom in (*)")
        print("  - Wheel down : zoom out (*)")
        print("  - drag : rotate display (*)")
        print("  - shift + drag : translate volume in the focal plan (*)")
        print("")
        print("Keyboard")
        print("--------\n")
        print("  - space : cycle selected slider (downard direction)")
        print("  - ctrl + space : cycle selected slider (upward direction)")
        print("  - left : increase slider value by %g%s of its range" % (100 * step1, "%"))
        print("  - right : decrease slider value by %g%s of its range"% (100 * step1, "%"))
        print("  - ctrl + left : increase slider value by %g%s of its range" % (100 * step2, "%"))
        print("  - ctrl + right : decrease slider value by %g%s of its range" % (100 * step2, "%"))
        print("  - ctrl + c : show/hide isosurface")
        print("  - c : show/hide all slices")
        print("  - x : show/hide X slices")
        print("  - y : show/hide Y slices")
        print("  - z : show/hide Z slices")
        print("  - g : show/hide grid")
        print("  - s : show/hide sliders (useful before screenshot)")
        print("  - q : quit")
        print("  - h : display help")
        print("")
        print("(*) native interactions (inherited from PyVista)")

    # show/hide button
    if is_notebook():
        def onoff_isosurf_check(checked):
            on_c_pressed(ctrl=1)
            
        onoff_isosurf = plotter.add_checkbox_button_widget(
            lambda checked : on_c_pressed(ctrl=1),
            value=True,
            position=(10, 10),
            color_on=color,
            background_color='black',
            size=25,
        )
        onoff_slice_x = plotter.add_checkbox_button_widget(
            lambda checked : toogle_actor_visibility(slice_x_actor, refresh=True),
            value=True,
            position=(45, 10),
            size=25,
        )
        onoff_slice_y = plotter.add_checkbox_button_widget(
            lambda checked : toogle_actor_visibility(slice_y_actor, refresh=True),
            value=True,
            position=(80, 10),
            size=25,
        )
        onoff_slice_z = plotter.add_checkbox_button_widget(
            lambda checked : toogle_actor_visibility(slice_z_actor, refresh=True),
            value=True,
            position=(115, 10),
            size=25,
        )
    
    # key binding
    plotter.add_key_event("Left", on_left_arrow)
    plotter.add_key_event("Right", on_right_arrow)
    plotter.add_key_event("space", change_slider)
    plotter.add_key_event("c", on_c_pressed)
    plotter.add_key_event("x", lambda : toogle_actor_visibility(slice_x_actor, refresh=True))
    plotter.add_key_event("y", lambda : toogle_actor_visibility(slice_y_actor, refresh=True))
    plotter.add_key_event("z", lambda : toogle_actor_visibility(slice_z_actor, refresh=True))
    plotter.add_key_event("g", lambda : toogle_actor_visibility(ax, refresh=True))
    plotter.add_key_event("s", on_s_pressed)
    plotter.add_key_event("h", on_h_pressed)
    
    # show & return
    plotter.show()
    
    return plotter

def imshow3d(u, xgrid=None, ygrid=None, zgrid=None, indexes=None,
             units='', figsize=None, valfmt='%0.3g',
             show_colorbar=True, cmap=None, origin='lower',
             aspect='equal', boundaries='same',
             interpolation='nearest', sx_color=None, sy_color=None,
             sz_color=None, xlim=None, ylim=None, zlim=None):
    """Interactive slice displayer for 3D images.
    
    Display slices of a 3D image, and explore its content through many
    interactive commands (once the figure is displayed, press the `h`
    key of your keyboard to display the list of interactive commands,
    also listed below).
    
    Parameters
    ----------
    
    u : ndarray
        Three dimensional array containing the values of the
        3D image ordered as follows:
        
        + axis 0 = spatial vertical axis (or Y-axis);
        + axis 1 = spatial horizontal axis (or X-axis);
        + axis 2 = saptial depth axis (or Z-axis).
    
    xgrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[1]`` containing
        the sampling nodes associated to the X-axis (axis 1) of the
        3D image ``u``.
    
    ygrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[0]`` containing
        the sampling nodes associated to the Y-axis (axis 0) of the
        3D image ``u``.
    
    zgrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[2]`` containing
        the sampling nodes associated to the Z-axis (axis 2) of the
        3D image ``u``.
    
    indexes : sequence of int, optional    
        When given, indexes must be a sequence of three int, ``indexes
        = (id0, id1, id2)``, such that `id0`, `id1` and `id2`
        correspond to the indexes used along each axis of the 3D
        volume to extract the slices to be displayed (using ``None``
        to keep a particular index to its default value is possible).
        
        The default setting is ``indexes = (u.shape[0]//2,
        u.shape[1]//2, u.shape[2]//2)``.
    
    units : str, optional
        Units associated to the X, Y and Z axes (handling of different
        axes units is not provided).
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).
    
    valfmt : str, optional
        %-format string used to format the slider values.
    
    show_colorbar : bool, optional
        Specify whether a colorbar should be displayed next to each
        slice image.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
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
        will be displayed on the screen using the same pixel
        size). Otherwise, a tight extent is used for each displayed
        slice image.
    
    interpolation : str, optional
        The interpolation method used (see ``matplotlib``
        documentation for the possible choices).
    
    sx_color : str or None, optional    
        The name of a matplotlib color for the X-slice slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    sy_color : str or None, optional    
        The name of a matplotlib color for the Y-slice slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    sz_color : str or None, optional    
        The name of a matplotlib color for the Z-slice slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    xlim : tuple of float, optional    
        Limits for the x-axis as a tuple ``(xmin, xmax)``. If
        provided, sets the visible range of the x-axis. If None, the
        limits are determined automatically based on the data. Note
        that the use of this option jointly with ``boundaries='same'``
        is discouraged and may lead to unexpected behavior.
    
    ylim : tuple of float, optional
        Limits for the y-axis as a tuple ``(ymin, ymax)``. If
        provided, sets the visible range of the y-axis. If None, the
        limits are determined automatically based on the data. Note
        that the use of this option jointly with ``boundaries='same'``
        is discouraged and may lead to unexpected behavior.
    
    zlim : tuple of float, optional
        Limits for the z-axis as a tuple ``(zmin, zmax)``. If
        provided, sets the visible range of the z-axis. If None, the
        limits are determined automatically based on the data. Note
        that the use of this option jointly with ``boundaries='same'``
        is discouraged and may lead to unexpected behavior.

    
    Return
    ------
    
    params : dict
        A dictionary containing all graphical objects and state
        parameters.

    
    Mouse and keyboard Interactive commands
    ---------------------------------------
    
    - x : select the X-slice slider
    - y : select the Y-slice slider
    - z : select the Z-slice slider
    - left : move the active slider back by one step
    - right : move the active slider forward by one step
    - ctrl + left : move the active slider back by 10% of its range
    - ctrl + right : move the active slider forward by 10% of its
      range
    - shift + left : move the active slider back by 5% of its range
    - shift + right : move the active slider forward by 5% of its
      range
    - up : toogle forward the slider selection (X -> Y -> Z)
    - down : toogle back the slider selection (Z -> Y -> X)
    - c : maximize the contrast among the three displayed slices
    - h : display help

    """

    # local functions to handle interactions
    def slider_update(params, dim, id):
        
        # retrieve slider, grid & unit
        slider = params['s' + dim]
        if slider.is_updating:
            return
        slider.is_updating = True
        strunit = params[dim + 'unit']
        
        # update label
        val = params[dim + 'grid'][id]
        str = slider.valfmt + ' %s'
        slider.valtext.set_text(str % (val, strunit))
        
        # update displayed slice
        u = params["u"]
        if slider == params['sx']:
            im = params["im_uyz"]
            im.set_data(u[:, slider.val, :])
        elif slider == params['sy']:
            im = params["im_uxz"]
            im.set_data(u[slider.val, :, :])
        elif slider == params['sz']:
            im = params["im_uyx"]
            im.set_data(u[:, :, slider.val])
        params['fig'].canvas.draw_idle()
        slider.is_updating = False
    
    def keypressed(params, event):
        
        # deal with key events
        if event.key in ('x', 'y', 'z'): # toogle selected slider
            dim = params['active_dim']
            r = params['r' + event.key]
            s = params['s' + event.key]
            params['r' + dim].set_visible(False)
            r.set_visible(True)
            params['active_dim'] = event.key
            plt.draw()
            params['fig'].canvas.draw_idle()
        elif event.key == 'left': # 1 step decrease for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            newval = s.val - 1
            if newval >= 0:
                s.set_val(newval)
                slider_update(params, dim, s.val)
        elif event.key == 'right': # 1 step increase for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            newval = s.val + 1
            if newval < s.nval:
                s.set_val(newval)
                slider_update(params, dim, s.val)
        elif event.key in ('ctrl+right', 'shift+right'): # 10% or 5% increase for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            val = s.val
            step = s.nval//(10 if event.key == 'ctrl+right' else 20)
            newval = min(s.nval -1, s.val + step)
            if newval != val:
                s.set_val(newval)
                slider_update(params, dim, s.val)
        elif event.key in ('ctrl+left', 'shift+left'): # 10% or 5% decrease for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            val = s.val
            step = s.nval//(10 if event.key == 'ctrl+left' else 20)
            newval = max(0, s.val - step)
            if newval != val:
                s.set_val(newval)
                slider_update(params, dim, s.val)
        elif event.key == 'up': # forward cycling for the selected slider
            dim = params['active_dim']
            params['r' + dim].set_visible(False)
            dim = params['next_dim'][dim]
            params['r' + dim].set_visible(True)
            params['active_dim'] = dim
            plt.draw()
            params['fig'].canvas.draw_idle()
        elif event.key == 'down': # backward cycling for the selected slider
            dim = params['active_dim']
            params['r' + dim].set_visible(False)
            dim = params['prev_dim'][dim]
            params['r' + dim].set_visible(True)
            params['active_dim'] = dim
            plt.draw()
            params['fig'].canvas.draw_idle()
        elif event.key == 'c': # maximize contrast among the displayed slices
            u = params['u']
            u_yz = u[:, params['sx'].val, :]
            u_xz = u[params['sy'].val, :, :]
            u_yx = u[:, :, params['sz'].val]
            cmin = min((u_yz.min(), u_xz.min(), u_yx.min()))
            cmax = max((u_yz.max(), u_xz.max(), u_yx.max()))
            params['im_uyz'].set_clim(cmin, cmax)
            params['im_uxz'].set_clim(cmin, cmax)
            params['im_uyx'].set_clim(cmin, cmax)
        elif event.key == 'h': # print interactive help
            print("")
            print("Interactive controls (3D image displayer)")
            print("=========================================\n")
            print("Keyboard")
            print("--------\n")
            print("  - x : select the X-slice slider")
            print("  - y : select the Y-slice slider")
            print("  - z : select the Z-slice slider")
            print("  - left : move the active slider back by one step")
            print("  - right : move the active slider forward by one step")
            print("  - ctrl + left : move the active slider back by 10% of its range")
            print("  - ctrl + right : move the active slider forward by 10% of its range")
            print("  - shift + left : move the active slider back by 5% of its range")
            print("  - shift + right : move the active slider forward by 5% of its range")
            print("  - up : toogle forward the slider selection (X -> Y -> Z)")
            print("  - down : toogle back the slider selection (Z -> Y -> X)")
            print("  - c : maximize the contrast among the three displayed slices")
            print("  - h : display help")
            print("")
    
    def get_k(params, event):
        
        # if the mouse pointer lies within a displayed slice, retrieve
        # the corresponding voxel indexes
        kx = ky = kz = -1
        ax = event.inaxes
        if ax == params['ax_uyz']:
            z = params['zgrid']
            y = params['ygrid']
            dz = params['dz']
            dy = params['dy']            
            kz = math.floor(.5 + (event.xdata - z[0]) / dz)
            ky = math.floor(.5 + (event.ydata - y[0]) / dy)
            kx = params['sx'].val
        elif ax == params['ax_uxz']:
            z = params['zgrid']
            x = params['xgrid']
            dz = params['dz']
            dx = params['dx']
            kz = math.floor(.5 + (event.xdata - z[0]) / dz)
            kx = math.floor(.5 + (event.ydata - x[0]) / dx)
            ky = params['sy'].val
        elif ax == params['ax_uyx']:
            y = params['ygrid']
            x = params['xgrid']
            dy = params['dy']
            dx = params['dx']
            kx = math.floor(.5 + (event.xdata - x[0]) / dx)
            ky = math.floor(.5 + (event.ydata - y[0]) / dy)
            kz = params['sz'].val
        
        # check whether the indexes are within the image domain or not
        valid_x = (0 <= kx < params['xgrid'].size)
        valid_y = (0 <= ky < params['ygrid'].size)
        valid_z = (0 <= kz < params['zgrid'].size)
        valid = all([valid_x, valid_y, valid_z])
        
        return kx, ky, kz, valid
    
    # retrieve image dimensions
    Ny, Nx, Nz = u.shape
    
    # create discrete indexes
    idx = np.arange(Nx, dtype='int32')
    idy = np.arange(Ny, dtype='int32')
    idz = np.arange(Nz, dtype='int32')
    
    # set default grids (if not provided)
    if xgrid is None:
        xgrid = idx
    if ygrid is None:
        ygrid = idy
    if zgrid is None:
        zgrid = idz
    
    # retrieve sampling steps
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]
    
    # get slices indexes
    if indexes is not None:
        x0 = u.shape[1]//2 if indexes[1] is None else indexes[1]
        y0 = u.shape[0]//2 if indexes[0] is None else indexes[0]
        z0 = u.shape[2]//2 if indexes[2] is None else indexes[2]
    else:
        x0 = u.shape[1]//2
        y0 = u.shape[0]//2
        z0 = u.shape[2]//2
    
    # retrieve slices
    u_yz = u[:, x0, :] #02
    u_xz = u[y0, :, :] #12
    u_yx = u[:, :, z0] #01
    
    # retrieve slices extents
    if origin == 'lower':
        extent_yx = (xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx, ygrid[0] - .5 * dy, ygrid[-1] + .5 * dy)
        extent_yz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, ygrid[0] - .5 * dy, ygrid[-1] + .5 * dy)
        extent_xz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx)
    else:
        extent_yx = (xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx, ygrid[-1] + .5 * dy, ygrid[0] - .5 * dy)
        extent_yz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, ygrid[-1] + .5 * dy, ygrid[0] - .5 * dy)
        extent_xz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, xgrid[-1] + .5 * dx, xgrid[0] - .5 * dx)
    
    # prepare figure & axes
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 5], hspace=0)
    ax_sx = fig.add_subplot(gs[0, 0])
    ax_sy = fig.add_subplot(gs[0, 1])
    ax_sz = fig.add_subplot(gs[0, 2])
    ax_uyz = fig.add_subplot(gs[1, 0])
    ax_uxz = fig.add_subplot(gs[1, 1])
    ax_uyx = fig.add_subplot(gs[1, 2])
    
    # display YZ-slice
    im_uyz = ax_uyz.imshow(u_yz, extent=extent_yz, origin=origin,
                           aspect=aspect, cmap=cmap,
                           interpolation=interpolation)
    ax_uyz.set_xlabel("Z")
    ax_uyz.set_ylabel("Y")
    
    # display XZ-slice
    im_uxz = ax_uxz.imshow(u_xz, extent=extent_xz, origin=origin,
                           aspect=aspect, cmap=cmap,
                           interpolation=interpolation)
    ax_uxz.set_xlabel("Z")
    ax_uxz.set_ylabel("X")
    
    # display YX-slice
    im_uyx = ax_uyx.imshow(u_yx, extent=extent_yx, origin=origin,
                           aspect=aspect, cmap=cmap,
                           interpolation=interpolation)
    ax_uyx.set_xlabel("X")
    ax_uyx.set_ylabel("Y")
    
    # deal with boundaries (if same pixel size is needed, give to all
    # subplots the same axes boundaries)
    if boundaries == 'same':
        Dxlim = max(xgrid[-1] + .5 * dx, zgrid[-1] + .5 * dz) - min(xgrid[0] - .5 * dx, zgrid[0] - .5 * dx)
        Dylim = max(xgrid[-1] + .5 * dx, ygrid[-1] + .5 * dy) - min(xgrid[0] - .5 * dx, ygrid[0] - .5 * dy)
        Dx = xgrid[-1] - xgrid[0]
        Dy = ygrid[-1] - ygrid[0]
        Dz = zgrid[-1] - zgrid[0]
        xlim_uyz = (zgrid[0] - .5 * (Dxlim - Dz), zgrid[-1] + .5 * (Dxlim - Dz))
        ylim_uyz = (ygrid[0] - .5 * (Dylim - Dy), ygrid[-1] + .5 * (Dylim - Dy))
        xlim_uxz = (zgrid[0] - .5 * (Dxlim - Dz), zgrid[-1] + .5 * (Dxlim - Dz))
        ylim_uxz = (xgrid[0] - .5 * (Dylim - Dx), xgrid[-1] + .5 * (Dylim - Dx))
        xlim_uyx = (xgrid[0] - .5 * (Dxlim - Dx), xgrid[-1] + .5 * (Dxlim - Dx))
        ylim_uyx = (ygrid[0] - .5 * (Dylim - Dy), ygrid[-1] + .5 * (Dylim - Dy))
        if origin != 'lower':
            ylim_uyz = (ylim_uyz[-1], ylim_uyz[-2])
            ylim_uxz = (ylim_uxz[-1], ylim_uxz[-2])
            ylim_uyx = (ylim_uyx[-1], ylim_uyx[-2])
        ax_uyz.set_xlim(xlim_uyz)
        ax_uxz.set_xlim(xlim_uxz)
        ax_uyx.set_xlim(xlim_uyx)
        ax_uyz.set_ylim(ylim_uyz)
        ax_uxz.set_ylim(ylim_uxz)
        ax_uyx.set_ylim(ylim_uyx)
    
    # deal with xlim/ylim/zlim options
    if xlim is not None:
        ax_uxz.set_ylim(xlim)
        ax_uyx.set_xlim(xlim)
    if ylim is not None:
        ax_uyz.set_ylim(ylim)
        ax_uyx.set_ylim(ylim)
    if zlim is not None:
        ax_uyz.set_xlim(zlim)
        ax_uxz.set_xlim(zlim)
    
    # add sliders
    plt.subplots_adjust(top=.95, bottom=0.05, left=0.07, right=0.93)
    sx = Slider(ax_sx, "X", idx[0], idx[-1], valinit=idx[x0], valstep=idx, color=sx_color, valfmt=valfmt)
    sy = Slider(ax_sy, "Y", idy[0], idy[-1], valinit=idy[y0], valstep=idy, color=sy_color, valfmt=valfmt)
    sz = Slider(ax_sz, "Z", idz[0], idz[-1], valinit=idz[z0], valstep=idz, color=sz_color, valfmt=valfmt)
    sx.nval = Nx
    sy.nval = Ny
    sz.nval = Nz
    sx.is_updating = sy.is_updating = sz.is_updating = False
    sx.valtext.set_text((valfmt + ' %s') % (xgrid[x0], units))
    sy.valtext.set_text((valfmt + ' %s') % (ygrid[y0], units))
    sz.valtext.set_text((valfmt + ' %s') % (zgrid[z0], units))
    ax_sx.set_title('X-slice', pad=0, y=0.93)
    ax_sy.set_title('Y-slice', pad=0, y=0.93)
    ax_sz.set_title('Z-slice', pad=0, y=0.93)
    
    # add slider rectangles
    r = []
    for ax in [ax_sx, ax_sy, ax_sz]:
        x, y, ww, hh = ax.get_position().bounds
        cof = .5
        rect = patches.Rectangle((x, y + hh * (1 - cof) / 2),
                                 width=ww, height=hh*cof, linewidth=2,
                                 edgecolor='black', facecolor='none',
                                 visible=False)
        fig.add_artist(rect)
        r.append(rect)
    rx, ry, rz = r
    rx.set_visible(True)
    plt.draw()
    
    # deal with colorbar display
    if show_colorbar:
        #
        # usual color bar may move the figure (this typically happens
        # when the image width is smaller than the image height),
        # leading to unaesthetic uncentered display
        #
        #plt.colorbar(im_uyz, ax=ax_uyz)
        #plt.colorbar(im_uxz, ax=ax_uxz)
        #plt.colorbar(im_uyx, ax=ax_uyx)
        #
        # this fixes the issue presented above
        d_uyz = make_axes_locatable(ax_uyz)
        d_uxz = make_axes_locatable(ax_uxz)
        d_uyx = make_axes_locatable(ax_uyx)
        cax_uyz = d_uyz.append_axes("right", size="7%", pad=0.05)
        cax_uxz = d_uxz.append_axes("right", size="7%", pad=0.05)
        cax_uyx = d_uyx.append_axes("right", size="7%", pad=0.05)
        plt.colorbar(im_uyz, cax=cax_uyz)
        plt.colorbar(im_uxz, cax=cax_uxz)
        plt.colorbar(im_uyx, cax=cax_uyx)
    
    # gather parameters
    params = {
        'fig': fig,
        'ax_sx': ax_sx,
        'ax_sy': ax_sy,
        'ax_sz': ax_sz,
        'ax_uyz': ax_uyz,
        'ax_uxz': ax_uxz,
        'ax_uyx': ax_uyx,
        'cax_uyz': cax_uyz,
        'cax_uxz': cax_uxz,
        'cax_uyx': cax_uyx,
        'rx': rx,
        'ry': ry,
        'rz': rz,
        'im_uyz': im_uyz,
        'im_uxz': im_uxz,
        'im_uyx': im_uyx,
        'sx': sx,
        'sy': sy,
        'sz': sz,
        'u': u,
        'xgrid': xgrid,
        'ygrid': ygrid,
        'zgrid': zgrid,
        'active_dim': 'x',
        'xunit': units,
        'yunit': units,
        'zunit': units,
        'next_dim' : {'x': 'y', 'y': 'z', 'z': 'x'},
        'prev_dim' : {'z': 'y', 'y': 'x', 'x': 'z'},
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'ready_to_follow': False,
    }
    
    # set callback functions
    sx.on_changed(functools.partial(slider_update, params, 'x'))
    sy.on_changed(functools.partial(slider_update, params, 'y'))
    sz.on_changed(functools.partial(slider_update, params, 'z'))
    fig.canvas.mpl_connect('key_press_event', functools.partial(keypressed, params))
    
    return params

def imshow4d(u, xgrid=None, ygrid=None, zgrid=None, Bgrid=None,
             spatial_unit='', B_unit='', figsize=None, valfmt='%0.3g',
             show_legend=True, legend_loc='upper right',
             show_colorbar=True, cmap=None, origin='lower',
             aspect='equal', boundaries='same',
             interpolation='nearest', sx_color=None, sy_color=None,
             sz_color=None, sb_color='g', xlim=None, ylim=None,
             zlim=None):
    """Interactive displayer for 4D spectral-spatial images.
    
    Display slices & spectra of a 4D spectral-spatial image, and
    explore its content through many interactive commands (once the
    figure is displayed, press the `h` key of your keyboard to display
    the list of interactive commands, also listed below).
    
    Parameters
    ----------
    
    u : ndarray
        Four dimensional array containing the values of the 4D
        spectral-spatial image ordered as follows:
        
        + axis 0 = homogeneous magnetic field intensity axis (or B-axis);
        + axis 1 = spatial vertical axis (or Y-axis);
        + axis 2 = spatial horizontal axis (or X-axis);
        + axis 3 = saptial depth axis (or Z-axis).
    
    xgrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[2]`` containing
        the sampling nodes associated to the X-axis (axis 2) of the
        4D spectral-spatial image ``u``.
    
    ygrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[1]`` containing
        the sampling nodes associated to the Y-axis (axis 1) of the
        4D spectral-spatial image ``u``.
    
    zgrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[3]`` containing
        the sampling nodes associated to the Z-axis (axis 3) of the
        4D spectral-spatial image ``u``.
    
    Bgrid : ndarray, optional
        Monodimensional ndarray with length ``u.shape[0]`` containing
        the sampling nodes associated to the B-axis (axis 0) of the
        4D spectral-spatial image ``u``.
    
    spatial_unit : str, optional
        Units associated to the X, Y and Z axes (handling of different
        axes units is not provided).
    
    B_unit : str, optional
        Units associated to the homogeneous magnetic field intensity
        (B) axis.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).
    
    valfmt : str, optional
        %-format string used to format the slider values.
    
    show_legend : bool, optional
        Decide whether the legend in the spectrum display area should
        be visible or not when the figure is drawn (note that once the
        figure is drawn, you can always show or hide the legend by
        pressing the 'S' key on your keyboard).
    
    legend_loc : str, optional
        The location of the legend in the spectrum display area
        (see ``matplotlib`` documentation for possible choices).
    
    show_colorbar : bool, optional
        Specify whether a colorbar should be displayed next to each
        slice image.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
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
        will be displayed on the screen using the same pixel
        size). Otherwise, a tight extent is used for each displayed
        slice image.
    
    interpolation : str, optional
        The interpolation method used (see ``matplotlib``
        documentation for the possible choices).
    
    sx_color : str or None, optional    
        The name of a matplotlib color for the X-slice slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    sy_color : str or None, optional    
        The name of a matplotlib color for the Y-slice slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    sz_color : str or None, optional    
        The name of a matplotlib color for the Z-slice slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    sb_color : str or None, optional    
        The name of a matplotlib color for the B slider progress
        bar (the default color is used when this optionnal input is
        set to ``None``).
    
    xlim : tuple of float, optional    
        Limits for the x-axis as a tuple ``(xmin, xmax)``. If
        provided, sets the visible range of the x-axis. If None, the
        limits are determined automatically based on the data. Note
        that the use of this option jointly with ``boundaries='same'``
        is discouraged and may lead to unexpected behavior.
    
    ylim : tuple of float, optional
        Limits for the y-axis as a tuple ``(ymin, ymax)``. If
        provided, sets the visible range of the y-axis. If None, the
        limits are determined automatically based on the data. Note
        that the use of this option jointly with ``boundaries='same'``
        is discouraged and may lead to unexpected behavior.
    
    zlim : tuple of float, optional
        Limits for the z-axis as a tuple ``(zmin, zmax)``. If
        provided, sets the visible range of the z-axis. If None, the
        limits are determined automatically based on the data. Note
        that the use of this option jointly with ``boundaries='same'``
        is discouraged and may lead to unexpected behavior.

    
    Return
    ------
    
    params : dict
        A dictionary containing all graphical objects and state
        parameters.
    
    
    Mouse and keyboard Interactive commands
    ---------------------------------------
    
    - single left click : keep the display for the spectrum under the
      mouse cursor    
    - x : select the X-slice slider
    - y : select the Y-slice slider
    - z : select the Z-slice slider
    - b : select the B-value slider
    - left : move the active slider back by one step
    - right : move the active slider forward by one step
    - ctrl + left : move the active slider back by 10% of its range
    - ctrl + right : move the active slider forward by 10% of its
      range
    - shift + left : move the active slider back by 5% of its range
    - shift + right : move the active slider forward by 5% of its
      range
    - up : toogle forward the slider selection (B -> X -> Y -> Z)
    - down : toogle back the slider selection (Z -> Y -> X -> B)
    - space : keep the display for spectrum under the mouse cursor
    - r : maximize the dynamic range of the last displayed spectrum
    - R : maximize the dynamic range of for all currently displayed
      spectra
    - c : maximize the contrast among the three displayed slices
    - d : remove the last displayed spectrum
    - D : remove all currently displayed spectra
    - S : show/hide legend
    - h : display help

    """
    # def local functions (callbacks)
    def slider_update(params, dim, id):
        
        # retrieve slider, grid & unit
        slider = params['s' + dim]
        if slider.is_updating:
            return
        slider.is_updating = True
        strunit = params[dim + 'unit']
        
        # update label
        val = params[dim + 'grid'][id]
        str = slider.valfmt + ' %s'
        slider.valtext.set_text(str % (val, strunit))
        
        # update displayed slice
        u = params["u"]
        if slider == params['sx']:
            im = params["im_uyz"]
            im.set_data(u[params["sb"].val, :, slider.val, :])
        elif slider == params['sy']:
            im = params["im_uxz"]
            im.set_data(u[params["sb"].val, slider.val, :, :])
        elif slider == params['sz']:
            im = params["im_uyx"]
            im.set_data(u[params["sb"].val, :, :, slider.val])
        else: # slider == param['sb']
            im_uyz = params["im_uyz"]
            im_uxz = params["im_uxz"]
            im_uyx = params["im_uyx"]
            im_uyz.set_data(u[slider.val, :, params["sx"].val, :])
            im_uxz.set_data(u[slider.val, params["sy"].val, :, :])
            im_uyx.set_data(u[slider.val, :, :, params["sz"].val])
        params['fig'].canvas.draw_idle()
        slider.is_updating = False
    
    def update_legend(params):
        b = params['ax_h'].get_legend().get_visible()
        params['ax_h'].legend(loc=params['legend_loc'])
        params['ax_h'].get_legend().set_visible(b)
        
    def keypressed(params, event):
        redisplay_spectrum = False
        
        # deal with key events
        if event.key in ('x', 'y', 'z', 'b'): # toogle selected slider
            dim = params['active_dim']
            r = params['r' + event.key]
            s = params['s' + event.key]
            params['r' + dim].set_visible(False)
            r.set_visible(True)
            params['active_dim'] = event.key
            plt.draw()
            params['fig'].canvas.draw_idle()
        elif event.key == 'left': # 1 step decrease for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            newval = s.val - 1
            if newval >= 0:
                s.set_val(newval)
                slider_update(params, dim, s.val)
                redisplay_spectrum = s is not params['sb']
        elif event.key == 'right': # 1 step increase for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            newval = s.val + 1
            if newval < s.nval:
                s.set_val(newval)
                slider_update(params, dim, s.val)
                redisplay_spectrum = s is not params['sb']
        elif event.key in ('ctrl+right', 'shift+right'): # 10% or 5% increase for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            val = s.val
            step = s.nval//(10 if event.key == 'ctrl+right' else 20)
            newval = min(s.nval -1, s.val + step)
            if newval != val:
                s.set_val(newval)
                slider_update(params, dim, s.val)
                redisplay_spectrum = s is not params['sb']
        elif event.key in ('ctrl+left', 'shift+left'): # 10% or 5% decrease for the active slider
            dim = params['active_dim']
            s = params['s' + dim]
            val = s.val
            step = s.nval//(10 if event.key == 'ctrl+left' else 20)
            newval = max(0, s.val - step)
            if newval != val:
                s.set_val(newval)
                slider_update(params, dim, s.val)
                redisplay_spectrum = s is not params['sb']
        elif event.key == 'up': # forward cycling for the selected slider
            dim = params['active_dim']
            params['r' + dim].set_visible(False)
            dim = params['next_dim'][dim]
            params['r' + dim].set_visible(True)
            params['active_dim'] = dim
            plt.draw()
            params['fig'].canvas.draw_idle()
        elif event.key == 'down': # backward cycling for the selected slider
            dim = params['active_dim']
            params['r' + dim].set_visible(False)
            dim = params['prev_dim'][dim]
            params['r' + dim].set_visible(True)
            params['active_dim'] = dim
            plt.draw()
            params['fig'].canvas.draw_idle()
        elif event.key == 'c': # maximize contrast among the displayed slices
            u = params['u']
            u_yz = u[params["sb"].val, :, params['sx'].val, :]
            u_xz = u[params["sb"].val, params['sy'].val, :, :]
            u_yx = u[params["sb"].val, :, :, params['sz'].val]
            cmin = min((u_yz.min(), u_xz.min(), u_yx.min()))
            cmax = max((u_yz.max(), u_xz.max(), u_yx.max()))
            params['im_uyz'].set_clim(cmin, cmax)
            params['im_uxz'].set_clim(cmin, cmax)
            params['im_uyx'].set_clim(cmin, cmax)
        elif event.key == 'r': # rescale yaxis to maximize the dynamic of the currently followed spectrum
            ymin = params['lines'][-1].get_ydata().min()
            ymax = params['lines'][-1].get_ydata().max()
            sgmin = -1 if ymin < 0 else 1
            sgmax = -1 if ymax < 0 else 1
            ymin = sgmin * (sgmin * ymin * 1.05)
            ymax = sgmax * (sgmax * ymax * 1.05)
            params['ax_h'].set_ylim((ymin, ymax))
        elif event.key == 'R': # rescale yaxis to maximize the dynamic of all displayed spectra
            if len(params['lines']) >= 1:
                cmin, cmax = math.inf, -math.inf
                for line in params['lines']:
                    ymin = line.get_ydata().min()
                    ymax = line.get_ydata().max()
                    sgmin = -1 if ymin < 0 else 1
                    sgmax = -1 if ymax < 0 else 1
                    ymin = sgmin * (sgmin * ymin * 1.05)
                    ymax = sgmax * (sgmax * ymax * 1.05)
                    cmin = min(cmin, ymin)
                    cmax = max(cmax, ymax)
                params['ax_h'].set_ylim((cmin, cmax))
        elif event.key == ' ': # same as left click (draw spectrum)
            on_click(params, event)
        elif event.key == 'd': # delete last plot
            lines = params['lines']
            n = len(lines)
            if n >= 2:
                line = lines[-2]
                col = line.get_color()
                lines.remove(line)
                line.remove()
                lines[-1].set_color(col)
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                id = (n % len(colors)) - 1
                rotated = colors[id:] + colors[:id]
                params['ax_h'].set_prop_cycle(color=rotated)
                update_legend(params)
        elif event.key == 'D': # delete all plots
            lines = params['lines']
            for line in lines[:(len(lines)-1)]:
                lines.remove(line)
                line.remove()
            lines[-1].set_color(params['default_color'])
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            rotated = colors[1:] + colors[:1]
            params['ax_h'].set_prop_cycle(color=rotated)
            update_legend(params)
        elif event.key == 'S': # show/hide legend
            b = params['ax_h'].get_legend().get_visible()
            params['ax_h'].get_legend().set_visible(not b)
            params['fig'].canvas.draw_idle()
        elif event.key == 'h': # print interactive help
            print("")
            print("Interactive controls (spectral-spatial 4D image displayer)")
            print("==========================================================\n")
            print("Mouse")
            print("-----\n")
            print("  - single left click : keep the display for the spectrum under the mouse cursor")
            print("")
            print("Keyboard")
            print("--------\n")
            print("  - x : select the X-slice slider")
            print("  - y : select the Y-slice slider")
            print("  - z : select the Z-slice slider")
            print("  - b : select the B-value slider")
            print("  - left : move the active slider back by one step")
            print("  - right : move the active slider forward by one step")
            print("  - ctrl + left : move the active slider back by 10% of its range")
            print("  - ctrl + right : move the active slider forward by 10% of its range")
            print("  - shift + left : move the active slider back by 5% of its range")
            print("  - shift + right : move the active slider forward by 5% of its range")
            print("  - up : toogle forward the slider selection (B -> X -> Y -> Z)")
            print("  - down : toogle back the slider selection (Z -> Y -> X -> B)")
            print("  - space : keep the display for spectrum under the mouse cursor")
            print("  - r : maximize the dynamic range of the last displayed spectrum")
            print("  - R : maximize the dynamic range of for all currently displayed spectra")
            print("  - c : maximize the contrast among the three displayed slices")
            print("  - d : remove the last displayed spectrum")
            print("  - D : remove all currently displayed spectra")
            print("  - S : show/hide legend")
            print("  - h : display help")
            print("")
         
        # if needed, redisplay spectrum
        if redisplay_spectrum:
            on_mouse_move(params, event)
    
    def get_k(params, event):
        
        # if the mouse pointer lies within a displayed slice, retrieve
        # the corresponding voxel indexes
        kx = ky = kz = -1
        ax = event.inaxes
        if ax == params['ax_uyz']:
            z = params['zgrid']
            y = params['ygrid']
            dz = params['dz']
            dy = params['dy']            
            kz = math.floor(.5 + (event.xdata - z[0]) / dz)
            ky = math.floor(.5 + (event.ydata - y[0]) / dy)
            kx = params['sx'].val
        elif ax == params['ax_uxz']:
            z = params['zgrid']
            x = params['xgrid']
            dz = params['dz']
            dx = params['dx']
            kz = math.floor(.5 + (event.xdata - z[0]) / dz)
            kx = math.floor(.5 + (event.ydata - x[0]) / dx)
            ky = params['sy'].val
        elif ax == params['ax_uyx']:
            y = params['ygrid']
            x = params['xgrid']
            dy = params['dy']
            dx = params['dx']
            kx = math.floor(.5 + (event.xdata - x[0]) / dx)
            ky = math.floor(.5 + (event.ydata - y[0]) / dy)
            kz = params['sz'].val
        
        # check whether the indexes are within the image domain or not
        valid_x = (0 <= kx < params['xgrid'].size)
        valid_y = (0 <= ky < params['ygrid'].size)
        valid_z = (0 <= kz < params['zgrid'].size)
        valid = all([valid_x, valid_y, valid_z])
        
        return kx, ky, kz, valid
    
    def on_mouse_move(params, event):
        
        # retrieve integer voxel indexes (if the pointer lies within a
        # displayed slice)
        kx, ky, kz, valid = get_k(params, event)
        line = params['lines'][-1]
        params['ready_to_follow'] = params['ready_to_follow'] or valid
        if params['ready_to_follow'] and params['ax_h'].get_legend().get_visible():
            line.set_visible(valid)
        
        # if the pointer lies within a displayed slice, update the
        # displayed spectrum
        if valid:
            h = params['u'][:, ky, kx, kz]
            line.set_ydata(h)
            label = 'u[:, %d, %d, %d]' % (ky, kx, kz)
            line.set_label(label)
            update_legend(params)
            params['fig'].canvas.draw_idle()
        elif params['ready_to_follow'] and params['ax_h'].get_legend().get_visible(): 
            line.set_label(' ')
            params['ax_h'].legend(loc=params['legend_loc'])
    
    def on_click(params, event):
        
        # retrieve integer voxel indexes (if the pointer lies within a
        # displayed slice)
        kx, ky, kz, valid = get_k(params, event)
        
        # if the pointer lies within a displayed slice, plot the
        # corresponding spectrum
        if valid:
            h = params['u'][:, ky, kx, kz]
            line, = params['ax_h'].plot(params['bgrid'], h)
            label = 'u[:, %d, %d, %d]' % (ky, kx, kz)
            line.set_label(label)
            update_legend(params)
            params['lines'].append(line)
            params['fig'].canvas.draw_idle()
    
    # retrieve image dimensions
    Nb, Ny, Nx, Nz = u.shape
    
    # create discrete indexes
    idx = np.arange(Nx, dtype='int32')
    idy = np.arange(Ny, dtype='int32')
    idz = np.arange(Nz, dtype='int32')
    idB = np.arange(Nb, dtype='int32')
    
    # set default grids (if not provided)
    if xgrid is None:
        xgrid = idx
    if ygrid is None:
        ygrid = idy
    if zgrid is None:
        zgrid = idz
    if Bgrid is None:
        Bgrid = idB
    
    # retrieve sampling steps
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]
    
    # get central slices
    x0, y0, z0, B0 = Nx//2, Ny//2, Nz//2, Nb//2
    u_yz = u[B0, :, x0, :] #02
    u_xz = u[B0, y0, :, :] #12
    u_yx = u[B0, :, :, z0] #01
    if origin == 'lower':
        extent_yx = (xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx, ygrid[0] - .5 * dy, ygrid[-1] + .5 * dy)
        extent_yz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, ygrid[0] - .5 * dy, ygrid[-1] + .5 * dy)
        extent_xz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx)
    else:
        extent_yx = (xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx, ygrid[-1] + .5 * dy, ygrid[0] - .5 * dy)
        extent_yz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, ygrid[-1] + .5 * dy, ygrid[0] - .5 * dy)
        extent_xz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, xgrid[-1] + .5 * dx, xgrid[0] - .5 * dx)
    
    # prepare figure & axes
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 3, width_ratios=[1, 1, 1], height_ratios=[1, 5, 2/3, 1, 1/3, 5], hspace=0)
    ax_sx = fig.add_subplot(gs[0, 0])
    ax_sy = fig.add_subplot(gs[0, 1])
    ax_sz = fig.add_subplot(gs[0, 2])
    ax_uyz = fig.add_subplot(gs[1, 0])
    ax_uxz = fig.add_subplot(gs[1, 1])
    ax_uyx = fig.add_subplot(gs[1, 2])
    ax_sb = fig.add_subplot(gs[3, :])
    ax_h = fig.add_subplot(gs[5, :])
    
    # display YZ-slice
    im_uyz = ax_uyz.imshow(u_yz, extent=extent_yz, origin=origin,
                           aspect=aspect, cmap=cmap,
                           interpolation=interpolation)
    ax_uyz.set_xlabel("Z")
    ax_uyz.set_ylabel("Y")
    
    # display XZ-slice    
    im_uxz = ax_uxz.imshow(u_xz, extent=extent_xz, origin=origin,
                           aspect=aspect, cmap=cmap,
                           interpolation=interpolation)
    ax_uxz.set_xlabel("Z")
    ax_uxz.set_ylabel("X")
    
    # display YX-slice
    im_uyx = ax_uyx.imshow(u_yx, extent=extent_yx, origin=origin,
                           aspect=aspect, cmap=cmap,
                           interpolation=interpolation)
    ax_uyx.set_xlabel("X")
    ax_uyx.set_ylabel("Y")
    
    # display spectrum
    label = 'u[:, %d, %d, %d]' % (y0, x0, z0)
    l0, = ax_h.plot(Bgrid, u[:, y0, x0, z0], label=label)
    ax_h.set_xlabel("B")
    ax_h.set_ylabel("local spectrum")
    ax_h.set_xlim((Bgrid[0], Bgrid[-1]))
    leg = ax_h.legend(loc=legend_loc)
    leg.set_visible(show_legend)
    
    # deal with boundaries (if same pixel size is needed, give to all
    # subplots the same axes boundaries)
    if boundaries == 'same':
        Dxlim = max(xgrid[-1] + .5 * dx, zgrid[-1] + .5 * dz) - min(xgrid[0] - .5 * dx, zgrid[0] - .5 * dx)
        Dylim = max(xgrid[-1] + .5 * dx, ygrid[-1] + .5 * dy) - min(xgrid[0] - .5 * dx, ygrid[0] - .5 * dy)
        Dx = xgrid[-1] - xgrid[0]
        Dy = ygrid[-1] - ygrid[0]
        Dz = zgrid[-1] - zgrid[0]
        xlim_uyz = (zgrid[0] - .5 * (Dxlim - Dz), zgrid[-1] + .5 * (Dxlim - Dz))
        ylim_uyz = (ygrid[0] - .5 * (Dylim - Dy), ygrid[-1] + .5 * (Dylim - Dy))
        xlim_uxz = (zgrid[0] - .5 * (Dxlim - Dz), zgrid[-1] + .5 * (Dxlim - Dz))
        ylim_uxz = (xgrid[0] - .5 * (Dylim - Dx), xgrid[-1] + .5 * (Dylim - Dx))
        xlim_uyx = (xgrid[0] - .5 * (Dxlim - Dx), xgrid[-1] + .5 * (Dxlim - Dx))
        ylim_uyx = (ygrid[0] - .5 * (Dylim - Dy), ygrid[-1] + .5 * (Dylim - Dy))
        if origin != 'lower':
            ylim_uyz = (ylim_uyz[-1], ylim_uyz[-2])
            ylim_uxz = (ylim_uxz[-1], ylim_uxz[-2])
            ylim_uyx = (ylim_uyx[-1], ylim_uyx[-2])
        ax_uyz.set_xlim(xlim_uyz)
        ax_uxz.set_xlim(xlim_uxz)
        ax_uyx.set_xlim(xlim_uyx)
        ax_uyz.set_ylim(ylim_uyz)
        ax_uxz.set_ylim(ylim_uxz)
        ax_uyx.set_ylim(ylim_uyx)

    # deal with xlim/ylim/zlim/Blim options
    if xlim is not None:
        ax_uxz.set_ylim(xlim)
        ax_uyx.set_xlim(xlim)
    if ylim is not None:
        ax_uyz.set_ylim(ylim)
        ax_uyx.set_ylim(ylim)
    if zlim is not None:
        ax_uyz.set_xlim(zlim)
        ax_uxz.set_xlim(zlim)
    
    # add sliders
    plt.subplots_adjust(top=.95, bottom=0.05, left=0.07, right=0.93)
    sx = Slider(ax_sx, "X", idx[0], idx[-1], valinit=idx[x0], valstep=idx, color=sx_color, valfmt=valfmt)
    sy = Slider(ax_sy, "Y", idy[0], idy[-1], valinit=idy[y0], valstep=idy, color=sy_color, valfmt=valfmt)
    sz = Slider(ax_sz, "Z", idz[0], idz[-1], valinit=idz[z0], valstep=idz, color=sz_color, valfmt=valfmt)
    sb = Slider(ax_sb, "B", idB[0], idB[-1], valinit=idB[B0], valstep=idB, color=sb_color, valfmt=valfmt)    
    sx.nval = Nx
    sy.nval = Ny
    sz.nval = Nz
    sb.nval = Nb
    sx.is_updating = sy.is_updating = sz.is_updating = sb.is_updating = False
    sx.valtext.set_text((valfmt + ' %s') % (xgrid[x0], spatial_unit))
    sy.valtext.set_text((valfmt + ' %s') % (ygrid[y0], spatial_unit))
    sz.valtext.set_text((valfmt + ' %s') % (zgrid[z0], spatial_unit))
    sb.valtext.set_text((valfmt + ' %s') % (Bgrid[B0], B_unit))
    ax_sx.set_title('X-slice')
    ax_sy.set_title('Y-slice')
    ax_sz.set_title('Z-slice')
    
    # add slider rectangles
    r = []
    for ax in [ax_sx, ax_sy, ax_sz, ax_sb]:
        x, y, ww, hh = ax.get_position().bounds
        cof = .5
        rect = patches.Rectangle((x, y + hh * (1 - cof) / 2),
                                 width=ww, height=hh*cof, linewidth=2,
                                 edgecolor='black', facecolor='none',
                                 visible=False)
        fig.add_artist(rect)
        r.append(rect)    
    rx, ry, rz, rb = r
    rb.set_visible(True)
    plt.draw()
    
    # deal with colorbar display
    if show_colorbar:
        #
        # usual color bar may move the figure (this typically happens
        # when the image width is smaller than the image height),
        # leading to unaesthetic uncentered display
        #
        #plt.colorbar(im_uyz, ax=ax_uyz)
        #plt.colorbar(im_uxz, ax=ax_uxz)
        #plt.colorbar(im_uyx, ax=ax_uyx)
        #
        # this fixes the issue presented above
        d_uyz = make_axes_locatable(ax_uyz)
        d_uxz = make_axes_locatable(ax_uxz)
        d_uyx = make_axes_locatable(ax_uyx)
        cax_uyz = d_uyz.append_axes("right", size="7%", pad=0.05)
        cax_uxz = d_uxz.append_axes("right", size="7%", pad=0.05)
        cax_uyx = d_uyx.append_axes("right", size="7%", pad=0.05)
        plt.colorbar(im_uyz, cax=cax_uyz)
        plt.colorbar(im_uxz, cax=cax_uxz)
        plt.colorbar(im_uyx, cax=cax_uyx)
    
    # gather parameters
    params = {
        'fig': fig,
        'ax_sx': ax_sx,
        'ax_sy': ax_sy,
        'ax_sz': ax_sz,
        'ax_uyz': ax_uyz,
        'ax_uxz': ax_uxz,
        'ax_uyx': ax_uyx,
        'ax_sb': ax_sb,
        'ax_h': ax_h,
        'cax_uyz': cax_uyz,
        'cax_uxz': cax_uxz,
        'cax_uyx': cax_uyx,
        'rx': rx,
        'ry': ry,
        'rz': rz,
        'rb': rb,
        'im_uyz': im_uyz,
        'im_uxz': im_uxz,
        'im_uyx': im_uyx,
        'lines': [l0],
        'sx': sx,
        'sy': sy,
        'sz': sz,
        'sb': sb,
        'u': u,
        'xgrid': xgrid,
        'ygrid': ygrid,
        'zgrid': zgrid,
        'bgrid': Bgrid,
        'active_dim': 'b',
        'xunit': spatial_unit,
        'yunit': spatial_unit,
        'zunit': spatial_unit,
        'bunit': B_unit,
        'next_dim' : {'b': 'x', 'x': 'y', 'y': 'z', 'z': 'b'},
        'prev_dim' : {'b': 'z', 'z': 'y', 'y': 'x', 'x': 'b'},
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'legend_loc': legend_loc,
        'ready_to_follow': False,
        'default_color': l0.get_color(),
    }
    
    # set callback functions
    sx.on_changed(functools.partial(slider_update, params, 'x'))
    sy.on_changed(functools.partial(slider_update, params, 'y'))
    sz.on_changed(functools.partial(slider_update, params, 'z'))
    sb.on_changed(functools.partial(slider_update, params, 'b'))
    fig.canvas.mpl_connect('button_press_event', functools.partial(on_click, params))
    fig.canvas.mpl_connect('key_press_event', functools.partial(keypressed, params))
    fig.canvas.mpl_connect('motion_notify_event', functools.partial(on_mouse_move, params))
    
    return params


def init_display_monosrc_2d(u, newfig=True, figsize=None,
                            time_sleep=0.01, units=None,
                            display_labels=False, displayFcn=None,
                            cmap=None, grids=None, origin='lower',
                            aspect=None, is_notebook=False):
    """Initialize display for a single 2D image.
    
    Parameters
    ----------
    
    u : ndarray
        Two-dimensional array
    
    newfig : bool, optional
        Specify whether the display must be done into a new figure or
        not.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
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

    # draw a new figure (if needed)
    if newfig:
        plt.figure(figsize=figsize)
    
    # draw image
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
        

def init_display_monosrc_3d(u, newfig=True, figsize=None,
                            time_sleep=0.01, units=None,
                            display_labels=False, displayFcn=None,
                            cmap=None, grids=None, origin='lower',
                            aspect=None, boundaries='auto',
                            is_notebook=False, indexes=None):
    """Initialize display for a single 3D image (display the three central slices of a 3D volume).
    
    Parameters
    ----------
    
    u : ndarray
        Three-dimensional array
    
    newfig : bool, optional
        Specify whether the display must be done into a new figure or
        not.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
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
        When given, indexes must be a sequence of three int, ``index =
        (id0, id1, id2)``, such that `id0`, `id1` and `id2` correspond
        to the indexes used along each axis of the 3D volume to
        extract the slices to be displayed (using ``None`` to keep a
        particular index to its default value is possible).
        
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
    
    # draw a new figure (if needed)
    if newfig:
        plt.figure(figsize=figsize)
    
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

def init_display_spectralspatial_4d(u, newfig=True, figsize=None,
                                    time_sleep=0.01, displayFcn=None,
                                    cmap=None, grids=None,
                                    display_labels=True,
                                    origin='lower', aspect=None,
                                    boundaries='auto',
                                    is_notebook=False,
                                    interpolation='nearest',
                                    slice_indexes=None,
                                    spec_indexes=None,
                                    show_legend=True,
                                    legend_loc='upper right',
                                    custom_spec=[],
                                    spec_normalization=False):
    """Initialize display for a spectral-spatial 4D image.
    
    Parameters
    ----------
    
    u : ndarray
        Three-dimensional array
    
    newfig : bool, optional
        Specify whether the display must be done into a new figure or
        not.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
        height in inches of the figure to be displayed. When not
        given, the default setting is that of `matplotlib` (see key
        'figure.figsize' of the matplotlib configuration parameter
        ``rcParams``).
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 4D image ``u`` into another 4D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    cmap : str, optional
        The registered colormap name used to map scalar data to colors
        in `matplotlib.imshow`.
    
    grids : sequence, optional    
        A sequence (tuple or list) of four monodimensional ndarrays,
        such that grids[0], grids[1], grids[2] and grids[3] contain
        the sampling nodes associated to axes 0 (B-axis), axe 1
        (Y-axis), axe 2 (X-axis) and axe 3 (Z-axis) of the input array
        ``u``.
        
        When given, the input grids are used to set the extent of the
        displayed slices (see `matplotlib.imshow` documentation).
    
    display_labels : bool, optional
        Set ``display_labels = True`` to display axes labels.
    
    origin : str in {'upper', 'lower'}, optional    
        This parameter only affects the 2D slice image axes. Place the
        [0, 0] index of the array in the upper left or lower left
        corner of the Axes. When not given, the default setting is
        that of `matplotlib` (see key 'image.origin' of the matplotlib
        configuration parameter ``rcParams``).
    
    aspect : str in {'equal', 'auto'} or float or None, optional    
        This parameter only affects the 2D slice image axes. Set
        aspect ratio of the Axes. This parameter is particularly
        relevant for images since it determines whether data pixels
        are square (see `matplotlib.imshow` documentation).
        
        When not given, the default setting is that of `matplotlib`
        (see key 'image.aspect' of the matplotlib configuration
        parameter ``rcParams``).
    
    boundaries : str in {'auto', 'same'}    
        This parameter only affects the 2D slice image axes. Use
        ``boundaries = 'same'`` to give all subplots the same axes
        boundaries (in particular, this ensures that all slice images
        will be displayed on the screen using the same pixel size).
    
        Otherwise, set ``boundaries = 'auto'`` to use tight extent for
        each displayed slice image.
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    interpolation : str, optional    
        The interpolation method used (see ``matplotlib``
        documentation for the possible choices).
    
    slice_indexes : sequence of four int, optional    
        Indexes used to extract slices of spectral-spatial image, the
        sequence must contain four integer indexes, ``slice_indexes =
        (id0, id1, id2, id3)``, that are used to extract and display
        2D slices from the 4D spectral-spatial image:
        
        + ``u[id0, :, id2, :]`` : YZ slice (for ``B = Bgrid[id0]``
          and ``X = xgrid[id2]``)
        
        + ``u[id0, id1, :, :]`` : ZX slice (for ``B = Bgrid[id0]``
          and ``Y = ygrid[id1]``)
        
        + ``u[id0, :, :, id3]`` : YX slice (for ``B = Bgrid[id0]``
          and ``Z = zgrid[id3]``)
        
        When not given, the default setting is ``slice_indexes[i] =
        u.shape[i]//2``, The slice indexes can be partially given,
        e.g., using ``slice_indexes = (5, None, 10, None)``, in this
        case, the not given indexes will be automatically replaced by
        their default values described above.
        
    spec_indexes : sequence of sequences of 3 int, optional    
        Each element ``spec_indexes[i]`` is a sequence containing 3
        integers, corresponding to the spatial indexes along the three
        spatial axes of the 4D image, and use to extract and display
        profiles.
        
        More precisely, the extracted and displayed profiles are
        the ``u[:, id[0], id[1], id[2]] for id in spec_indexes``.
        
        When not given, the default setting is ``spec_indexes =
        ((u.shape[1]//2, u.shape[2]//2, u.shape[3]//2))``.
    
    show_legend : bool, optional
        Decide whether the legend in the spectrum display area should
        be visible or not when the figure is drawn.
    
    legend_loc : str, optional
        The location of the legend in the spectrum display area (see
        ``matplotlib`` documentation for possible choices).
    
    custom_spec : sequence of dict, optional    
        This parameter can be used to specify some custom profiles to
        be displayed in the bottom axes.
        
        When given, each element of ``custom_spec`` must be a
        dictionary with the following key-values:
        
        + 'spec' : the custom profile to be displayed (monodimensional
          numpy array)
        
        + 'B' : the sampling grid (B axis) associated to the custom
          profile (monodimensional numpy array)
        
        + 'label' : label of the custom profile (str)
    
    spec_normalization : bool, optional    
        Decide whether the displayed spectra should be normalized or
        not. When spectra normalization is enabled, each displayed
        spectrum is divided by its maximum value (normalization occurs
        at display only, it does not affect the latent image)
    
    Return
    ------
    
    fg : dict
        Contains axes and handles of the produced image instance
    
    
    See also
    --------
    
    update_display_spectralspatial_4d

    """
    
    # prepare image
    im = u if displayFcn is None else displayFcn(u)
    
    # retrieve image dimensions
    Nb, Ny, Nx, Nz = im.shape
    
    # create discrete indexes
    idx = np.arange(Nx, dtype='int32')
    idy = np.arange(Ny, dtype='int32')
    idz = np.arange(Nz, dtype='int32')
    idB = np.arange(Nb, dtype='int32')
    
    # set default grids (if not provided)
    if grids is not None:
        Bgrid, ygrid, xgrid, zgrid = grids
    else:
        Bgrid = ygrid = xgrid = zgrid = None
    if Bgrid is None:
        Bgrid = idB
    if xgrid is None:
        xgrid = idx
    if ygrid is None:
        ygrid = idy
    if zgrid is None:
        zgrid = idz
    if Bgrid is None:
        Bgrid = idB
    
    # retrieve sampling steps
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]
    dB = Bgrid[1] - Bgrid[0]
    
    # get slices indexes
    if slice_indexes is not None:
        B0, y0, x0, z0 = slice_indexes
    else:
        B0 = y0 = x0 = z0 = None
    B0 = Nb//2 if B0 is None else B0
    x0 = Nx//2 if x0 is None else x0
    y0 = Ny//2 if y0 is None else y0
    z0 = Nz//2 if z0 is None else z0
    
    # extract slices
    slice_yz = im[B0, :, x0, :] #02
    slice_xz = im[B0, y0, :, :] #12
    slice_yx = im[B0, :, :, z0] #01
    if origin == 'lower':
        extent_yx = (xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx, ygrid[0] - .5 * dy, ygrid[-1] + .5 * dy)
        extent_yz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, ygrid[0] - .5 * dy, ygrid[-1] + .5 * dy)
        extent_xz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx)
    else:
        extent_yx = (xgrid[0] - .5 * dx, xgrid[-1] + .5 * dx, ygrid[-1] + .5 * dy, ygrid[0] - .5 * dy)
        extent_yz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, ygrid[-1] + .5 * dy, ygrid[0] - .5 * dy)
        extent_xz = (zgrid[0] - .5 * dz, zgrid[-1] + .5 * dz, xgrid[-1] + .5 * dx, xgrid[0] - .5 * dx)
    
    # prepare figure & axes
    fg = plt.figure(figsize=figsize)
    gs = GridSpec(5, 3, width_ratios=[1, 1, 1], height_ratios=[1, 5, 3, .1, 8], hspace=0)
    ax_slices_title = fg.add_subplot(gs[0, :])
    ax_im_yz = fg.add_subplot(gs[1, 0])
    ax_im_xz = fg.add_subplot(gs[1, 1])
    ax_im_yx = fg.add_subplot(gs[1, 2])
    ax_spec_title = fg.add_subplot(gs[3, :])
    ax_h = fg.add_subplot(gs[4, :])
    
    # display YZ-slice
    im_yz = ax_im_yz.imshow(slice_yz, extent=extent_yz, origin=origin,
                            aspect=aspect, cmap=cmap,
                            interpolation=interpolation)
    if display_labels:
        ax_im_yz.set_xlabel("Z")
        ax_im_yz.set_ylabel("Y")
        ax_im_yz.set_title("X = %g" % xgrid[x0])
    
    # display XZ-slice    
    im_xz = ax_im_xz.imshow(slice_xz, extent=extent_xz, origin=origin,
                            aspect=aspect, cmap=cmap,
                            interpolation=interpolation)
    if display_labels:
        ax_im_xz.set_xlabel("Z")
        ax_im_xz.set_ylabel("X")
        ax_im_xz.set_title("Y = %g" % ygrid[y0])
    
    # display YX-slice
    im_yx = ax_im_yx.imshow(slice_yx, extent=extent_yx, origin=origin,
                            aspect=aspect, cmap=cmap,
                            interpolation=interpolation)
    if display_labels:
        ax_im_yx.set_xlabel("X")
        ax_im_yx.set_ylabel("Y")
        ax_im_yx.set_title("Z = %g" % zgrid[z0])
    
    # set slices title
    ax_slices_title.axis("off")
    ax_slices_title.set_title("Extracted slices (B = %g)" % Bgrid[B0])

    # deal with spec normalization
    if spec_normalization:
        display_spec = lambda h : h / h.max()
    else:
        display_spec = lambda h : h
    
    # retrieve spec indexes
    if spec_indexes is None:
        spec_indexes = [[y0, x0, z0]]
    
    # display custom spec
    smin, smax = math.inf, -math.inf
    for cspec in custom_spec:
        s = display_spec(cspec['spec'])
        l0, = ax_h.plot(cspec['B'], s, label=cspec['label'])
        smin = min(smin, s.min())
        smax = max(smax, s.max())
    
    # display spec
    spec_hdl = []
    for id in spec_indexes:        
        label = 'u[:, %d, %d, %d]' % (id[0], id[1], id[2])
        l0, = ax_h.plot(Bgrid, display_spec(im[:, id[0], id[1], id[2]]), label=label)
        spec_hdl.append(l0)
    ax_h.set_xlim((Bgrid[0], Bgrid[-1]))
    leg = ax_h.legend(loc=legend_loc)
    leg.set_visible(show_legend)
    if display_labels:
        ax_h.set_xlabel("B")
    
    # compute total number of displayed spec
    Nspec = len(spec_indexes) + len(custom_spec)
    
    # set spec title
    ax_spec_title.axis("off")
    spec_title = "Extracted profile%s" % ("s" if Nspec > 1 else "")
    if spec_normalization:
        spec_title += " (normalized)"
    ax_spec_title.set_title(spec_title)
    
    # deal with boundaries (if same pixel size is needed, give to all
    # subplots the same axes boundaries)
    if boundaries == 'same':
        Dxlim = max(xgrid[-1] + .5 * dx, zgrid[-1] + .5 * dz) - min(xgrid[0] - .5 * dx, zgrid[0] - .5 * dx)
        Dylim = max(xgrid[-1] + .5 * dx, ygrid[-1] + .5 * dy) - min(xgrid[0] - .5 * dx, ygrid[0] - .5 * dy)
        Dx = xgrid[-1] - xgrid[0]
        Dy = ygrid[-1] - ygrid[0]
        Dz = zgrid[-1] - zgrid[0]
        xlim_im_yz = (zgrid[0] - .5 * (Dxlim - Dz), zgrid[-1] + .5 * (Dxlim - Dz))
        ylim_im_yz = (ygrid[0] - .5 * (Dylim - Dy), ygrid[-1] + .5 * (Dylim - Dy))
        xlim_im_xz = (zgrid[0] - .5 * (Dxlim - Dz), zgrid[-1] + .5 * (Dxlim - Dz))
        ylim_im_xz = (xgrid[0] - .5 * (Dylim - Dx), xgrid[-1] + .5 * (Dylim - Dx))
        xlim_im_yx = (xgrid[0] - .5 * (Dxlim - Dx), xgrid[-1] + .5 * (Dxlim - Dx))
        ylim_im_yx = (ygrid[0] - .5 * (Dylim - Dy), ygrid[-1] + .5 * (Dylim - Dy))
        if origin != 'lower':
            ylim_im_yz = (ylim_im_yz[-1], ylim_im_yz[-2])
            ylim_im_xz = (ylim_im_xz[-1], ylim_im_xz[-2])
            ylim_im_yx = (ylim_im_yx[-1], ylim_im_yx[-2])
        ax_im_yz.set_xlim(xlim_im_yz)
        ax_im_xz.set_xlim(xlim_im_xz)
        ax_im_yx.set_xlim(xlim_im_yx)
        ax_im_yz.set_ylim(ylim_im_yz)
        ax_im_xz.set_ylim(ylim_im_xz)
        ax_im_yx.set_ylim(ylim_im_yx)
    
    # gather outputs
    hdl = {
        'fg': fg,
        'im_yz': im_yz,
        'im_xz': im_xz,
        'im_yx': im_yx,
        'ax_spec': ax_h,
        'spec_hdl': spec_hdl,
        'ylim_custom_spec': (smin, smax),
    }
    
    # pause an return
    if is_notebook:
        time.sleep(time_sleep)
    else:
        plt.pause(time_sleep)
    
    return hdl

def update_display_spectralspatial_4d(u, hdl, is_notebook=False,
                                      displayFcn=None,
                                      adjust_dynamic=True,
                                      time_sleep=0.01,
                                      slice_indexes=None,
                                      spec_indexes=None,
                                      spec_normalization=False):
    """Update spectral-spatial 4D image display.
    
    Parameters
    ----------
    
    u : ndarray
        Three-dimensional array
    
    hdl : dict
        Contains the handles and axes of the image instance to be
        updated (output of
        ``py:func:init_display_spectralspatial_4d``)
    
    is_notebook : bool, optional
        Indicate whether the running environment is an interactive
        notebook (``is_notebook = True``) or not (``is_notebook =
        False``).
    
    displayFcn : <class 'function'>, optional 
        Function with prototype ``im = displayFcn(u)`` that changes
        the 4D image ``u`` into another 4D image. When `displayFcn` is
        given, the displayed image is ``im = displayFcn(u)`` instead
        of ``u``.
    
    adjust_dynamic : bool, optional
        Set ``adjust_dynamic = True`` to maximize the dynamic of the
        displayed slices and profiles during the updating
        process. Otherwise, set ``adjust_dynamic = False`` to keep the
        displayed dynamic unchanged.
    
    time_sleep : float, optional 
        Duration in seconds of pause or sleep (depending on the
        running environment) to perform after image drawing.
    
    slice_indexes : sequence of four int, optional    
        Indexes used to extract slices of spectral-spatial image, the
        sequence must contain four integer indexes, ``slice_indexes =
        (id0, id1, id2, id3)``, that are used to extract and display
        2D slices from the 4D spectral-spatial image:
        
        + ``u[id0, :, id2, :]`` : YZ slice (for ``B = Bgrid[id0]``
          and ``X = xgrid[id2]``)
        
        + ``u[id0, id1, :, :]`` : ZX slice (for ``B = Bgrid[id0]``
          and ``Y = ygrid[id1]``)
        
        + ``u[id0, :, :, id3]`` : YX slice (for ``B = Bgrid[id0]``
          and ``Z = zgrid[id3]``)
        
        When not given, the default setting is ``slice_indexes[i] =
        u.shape[i]//2``, The slice indexes can be partially given,
        e.g., using ``slice_indexes = (5, None, 10, None)``, in this
        case, the not given indexes will be automatically replaced by
        their default values described above.
        
    spec_indexes : sequence of sequences of 3 int, optional    
        Each element ``spec_indexes[i]`` is a sequence containing 3
        integers, corresponding to the spatial indexes along the three
        spatial axes of the 4D image, and use to extract and display
        profiles.
        
        More precisely, the extracted and displayed profiles are
        the ``u[:, id[0], id[1], id[2]] for id in spec_indexes``.
        
        When not given, the default setting is ``spec_indexes =
        ((u.shape[1]//2, u.shape[2]//2, u.shape[3]//2))``.
    
    spec_normalization : bool, optional    
        Decide whether the displayed spectra should be normalized or
        not. When spectra normalization is enabled, each displayed
        spectrum is divided by its maximum value (normalization occurs
        at display only, it does not affect the latent image)

    
    Return
    ------
    
    None
    
    
    See also
    --------
    
    init_display_spectralspatial_4d
    
    """
    
    # retrieve slice images
    im = u if displayFcn is None else displayFcn(u)
    
    # retrieve image dimensions
    Nb, Ny, Nx, Nz = im.shape
    
    # get slices indexes
    if slice_indexes is not None:
        B0, y0, x0, z0 = slice_indexes
    else:
        B0 = y0 = x0 = z0 = None
    B0 = Nb//2 if B0 is None else B0
    x0 = Nx//2 if x0 is None else x0
    y0 = Ny//2 if y0 is None else y0
    z0 = Nz//2 if z0 is None else z0
    
    # get spec indexes
    if spec_indexes is None:
        spec_indexes = [[y0, x0, z0]]
    
    # extract slices
    slice_yz = im[B0, :, x0, :] #02
    slice_xz = im[B0, y0, :, :] #12
    slice_yx = im[B0, :, :, z0] #01
    
    # draw images
    hdl['im_yz'].set_data(slice_yz)
    hdl['im_xz'].set_data(slice_xz)
    hdl['im_yx'].set_data(slice_yx)
    
    # if needed, adjust dynamics
    if(adjust_dynamic):
        cmin = min((slice_yz.min(), slice_xz.min(), slice_yx.min()))
        cmax = max((slice_yz.max(), slice_xz.max(), slice_yx.max()))
        hdl['im_yz'].set_clim(cmin, cmax)
        hdl['im_xz'].set_clim(cmin, cmax)
        hdl['im_yx'].set_clim(cmin, cmax)
    
    # deal with spec normalization
    if spec_normalization:
        display_spec = lambda h : h / h.max()
    else:
        display_spec = lambda h : h
    
    # draw specs
    if adjust_dynamic:
        #ymin, ymax = math.inf, -math.inf
        ymin, ymax = hdl['ylim_custom_spec']
        for i, id in enumerate(spec_indexes):
            label = 'u[:, %d, %d, %d]' % (id[0], id[1], id[2])
            s = display_spec(im[:, id[0], id[1], id[2]])
            ymin = min((ymin, s.min()))
            ymax = max((ymax, s.max()))
            hdl['spec_hdl'][i].set_ydata(s)
        ymax *= (1 + np.sign(ymax) * .05)
        ymin *= (1 - np.sign(ymin) * .05)
        hdl['ax_spec'].set_ylim((ymin, ymax))
            
    else:
        for i, id in enumerate(spec_indexes):
            label = 'u[:, %d, %d, %d]' % (id[0], id[1], id[2])
            hdl['spec_hdl'][i].set_ydata(display_spec(im[:, id[0], id[1], id[2]]))
    
    # deal with interactive notebook running environments
    if is_notebook:
        display.clear_output(wait=True)
        display.display(pl.gcf())    
        time.sleep(time_sleep)
    #else:
    #    plt.pause(time_sleep)
    
    return

def init_display_multisrc_2d(u, newfig=True, figsize=None,
                             time_sleep=0.01, units=None,
                             display_labels=False, displayFcn=None,
                             cmap=None, grids=None, origin='lower',
                             aspect=None, boundaries='auto',
                             is_notebook=False, src_labels=None):
    """Initialize display for a sequence of 2D images.
    
    Parameters
    ----------
    
    u : sequence of ndarray
        The sequence (tuple or list) of two-dimensional images to be
        displayed.
        
    newfig : bool, optional
        Specify whether the display must be done into a new figure or
        not.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
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
    
    # draw a new figure (if needed)
    if newfig:
        plt.figure(figsize=figsize)
    
    # set figure size (if given)
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


def init_display_multisrc_3d(u, newfig=True, figsize=None,
                             time_sleep=0.01, units=None,
                             display_labels=False, displayFcn=None,
                             cmap=None, grids=None, origin='lower',
                             aspect=None, boundaries='auto',
                             is_notebook=False, indexes=None,
                             src_labels=None):
    """Initialize display for a sequence of 3D images.
    
    Parameters
    ----------
    
    u : sequence of ndarray
        The sequence (tuple or list) of three-dimensional images to be
        displayed.
    
    newfig : bool, optional
        Specify whether the display must be done into a new figure or
        not.
    
    figsize : (float, float), optional
        When given, figsize must be a tuple with length two and such
        that ``figsize[0]`` and ``figsize[1]`` are the width and
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
    
    # draw a new figure or retrieve the current one
    _fg_ = plt.figure(figsize=figsize) if newfig else plt.gcf()
    
    # update figsize (if needed)
    if (not newfig) and (figsize is not None):
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


def create_2d_displayer(nsrc=1, newfig=True, figsize=None,
                        displayFcn=None, time_sleep=0.01, units=None,
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
    return Displayer(nsrc, ndim, newfig=newfig, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     units=units, adjust_dynamic=adjust_dynamic,
                     display_labels=display_labels, cmap=cmap,
                     grids=grids, origin=origin, aspect=aspect,
                     boundaries=boundaries, indexes=indexes,
                     src_labels=src_labels)

def create_3d_displayer(nsrc=1, newfig=True, figsize=None,
                        displayFcn=None, time_sleep=0.01, units=None,
                        extents=None, adjust_dynamic=True,
                        display_labels=False, cmap=None, grids=None,
                        origin='lower', aspect=None,
                        boundaries='auto', indexes=None,
                        src_labels=None):
    """Instantiate a single 3D image displayer.

    This function instantiate a ``pyepri.Displayer`` class instance
    using ndim=3 and passing all the other args & kwargs to the
    ``pyepri.displayers.Displayer`` default constructor (type
    ``help(pyepri.displayers)`` for more details).

    """
    ndim = 3
    return Displayer(nsrc, ndim, newfig=newfig, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     units=units, adjust_dynamic=adjust_dynamic,
                     display_labels=display_labels, cmap=cmap,
                     grids=grids, origin=origin, aspect=aspect,
                     boundaries=boundaries, indexes=indexes,
                     src_labels=src_labels)

def create_spectralspatial_4d_displayer(newfig=True, figsize=None,
                                        displayFcn=None,
                                        time_sleep=0.01,
                                        adjust_dynamic=True,
                                        cmap=None, grids=None,
                                        display_labels=True,
                                        origin='lower', aspect=None,
                                        boundaries='auto',
                                        interpolation='nearest',
                                        slice_indexes=None,
                                        spec_indexes=None,
                                        show_legend=True,
                                        legend_loc='upper right',
                                        custom_spec=[],
                                        spec_normalization=False):
    nsrc = 1
    ndim = 4
    return Displayer(nsrc, ndim, newfig=newfig, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     adjust_dynamic=adjust_dynamic, cmap=cmap,
                     grids=grids, display_labels=display_labels,
                     origin=origin, aspect=aspect,
                     boundaries=boundaries,
                     interpolation=interpolation,
                     slice_indexes=slice_indexes,
                     spec_indexes=spec_indexes,
                     show_legend=show_legend, legend_loc=legend_loc,
                     custom_spec=custom_spec,
                     spec_normalization=spec_normalization)

def create(u, newfig=True, figsize=None, displayFcn=None,
           time_sleep=0.01, units=None, extents=None,
           adjust_dynamic=True, display_labels=False, cmap=None,
           grids=None, origin='lower', aspect=None, boundaries='auto',
           indexes=None, src_labels=None, interpolation='nearest',
           slice_indexes=None, spec_indexes=None, show_legend=True,
           legend_loc='upper right', custom_spec=[],
           spec_normalization=False):
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
    return Displayer(nsrc, ndim, newfig=newfig, figsize=figsize,
                     displayFcn=displayFcn, time_sleep=time_sleep,
                     units=units, adjust_dynamic=adjust_dynamic,
                     display_labels=display_labels, cmap=cmap,
                     grids=grids, origin=origin, aspect=aspect,
                     boundaries=boundaries,
                     force_multisrc=force_multisrc,
                     interpolation=interpolation,
                     slice_indexes=slice_indexes,
                     spec_indexes=spec_indexes,
                     show_legend=show_legend, legend_loc=legend_loc,
                     custom_spec=custom_spec,
                     spec_normalization=spec_normalization)

def get_number(fg):
    """Retrieve displayed figure number.
    
    Parameters
    ----------
    
    fg : <class 'matplotlib.image.AxesImage'> or sequence of <class \
    'matplotlib.image.AxesImage'> or dict
        Image instance or sequence of image instances that belong to
        the same figure or dict with 'fg' key and <class
        'matplotlib.figure.Figure'> associated value.

    
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
    elif isinstance(fg, dict):
        fgnum = fg['fg'].get_figure().number
    else:
        fgnum = fg.get_figure().number
    
    return fgnum

def _check_inputs_(nsrc=None, ndim=None, displayFcn=None,
                   time_sleep=None, units=None, adjust_dynamic=None,
                   display_labels=None, cmap=None, grids=None,
                   origin=None, aspect=None, boundaries=None,
                   u=__EMPTY_ARRAY__, newfig=None, figsize=None,
                   indexes=None, src_labels=None, interpolation=None,
                   legend_loc=None, show_legend=None,
                   spec_normalization=None, slice_indexes=None,
                   spec_indexes=None, custom_spec=None):
    """Factorized consistency checks for functions in this :py:mod:`pyepri.displayers` submodule.

    """
    
    # type checks
    checks._check_type_(int, nsrc=nsrc, ndim=ndim)
    checks._check_type_(float, time_sleep=time_sleep)
    checks._check_type_(bool, adjust_dynamic=adjust_dynamic, display_labels=display_labels, newfig=newfig, show_legend=show_legend, spec_normalization=spec_normalization)
    checks._check_type_(str, units=units, cmap=cmap, origin=origin, aspect=aspect, boundaries=boundaries, interpolation=interpolation, legend_loc=legend_loc)
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
    
    # additional spectral-spatial inputs
    if slice_indexes is not None:
        checks._check_seq_(t=int, n=ndim, slice_indexes=slice_indexes)
    
    if spec_indexes is not None:
        checks._check_seq_of_seq_(t=int, len1=3, spec_indexes=spec_indexes)
    
    if custom_spec is not None:
        checks._check_seq_(t=dict, custom_spec=custom_spec)
        for d in custom_spec:
            if "B" not in d.keys():
                raise RuntimeError(
                    "All elements in `custom_spec` must be dictionaries and contain a key: 'B'"
                )
            if "spec" not in d.keys():
                raise RuntimeError(
                    "All elements in `custom_spec` must be dictionaries and contain a key: 'spec'"
                )
            if "label" not in d.keys():
                raise RuntimeError(
                    "All elements in `custom_spec` must be dictionaries and contain a key: 'label'"
                )
            if not isinstance(d["B"], np.ndarray) or 1 != d["B"].ndim:
                raise RuntimeError(
                    "All ellements in `custom_spec` with key 'B' must be monodimensional numpy arrays"
                )
            if not isinstance(d["spec"], np.ndarray) or 1 != d["B"].ndim:
                raise RuntimeError(
                    "All ellements in `custom_spec` with key 'spec' must be monodimensional numpy arrays"
                )
            if not isinstance(d["label"], str):
                raise RuntimeError(
                    "All ellements in `custom_spec` with key 'label' must have str type"
                )
            if not len(d["B"]) == len(d["spec"]):
                raise RuntimeError(
                    "For all dictionary d in `custom_spec`, we must have len(d['B']) == len(d['spec'])"
                )
    
    return True


class Displayer:
    """Class for display and update of different kind of images, in different running environments.
    
    Supported images
    ----------------
    
    + single 2D image : the input signal is a two-dimensional array
    
    + single 3D image : the input signal is a three-dimensional array
    
    + multisources 2D images : the input signal is a sequence of
      two-dimensional arrays (each array being called a `source`)
    
    + multisources 3D images : the input signal is a sequence of
      three-dimensional arrays (each array being called a `source`)
    
    
    Displaying rules 
    ----------------
    
    + single 2D image : the image is displayed using
      `matplotlib.imshow`
    
    + single 3D image : the three central slices (along each axis) of
      the image are drawn using `matplotlib.imshow` into a single row
      of subplots.
    
    + multisources 2D images : the source images are drawn using
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
    
    def __init__(self, nsrc, ndim, newfig=True, figsize=None,
                 displayFcn=None, time_sleep=0.01, units=None,
                 adjust_dynamic=True, display_labels=False, cmap=None,
                 grids=None, origin='lower', aspect=None,
                 boundaries='auto', force_multisrc=False,
                 indexes=None, src_labels=None,
                 interpolation='nearest', slice_indexes=None,
                 spec_indexes=None, show_legend=True,
                 legend_loc='upper right', custom_spec=[],
                 spec_normalization=False):
        """Default constructor for ``pyepri.displayers.Displayer`` objects instanciation.
        
        
        Parameters
        ----------
        
        nsrc : int
            Number of source images to be displayed (must be >= 1).
        
        ndim : int in {1, 2, 3}
            Dimensions of the source images to be displayed.
        
        newfig : bool, optional
            Specify whether the display must be done into a new figure
            or not.
        
        figsize : (float, float), optional
            When given, figsize must be a tuple with length two and
            such that ``figsize[0]`` and ``figsize[1]`` are the width
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
        
        interpolation : str, optional        
            Used for spectral-spatial displayers only. The
            interpolation method used (see ``matplotlib``
            documentation for the possible choices).
        
        slice_indexes : sequence of four int, optional        
            Used for spectral-spatial displayers only. Indexes used to
            extract slices of spectral-spatial image, the sequence
            must contain four integer indexes, ``slice_indexes = (id0,
            id1, id2, id3)``, that are used to extract and display 2D
            slices from the 4D spectral-spatial image:
            
            + ``u[id0, :, id2, :]`` : YZ slice (for ``B = Bgrid[id0]``
              and ``X = xgrid[id2]``)
            
            + ``u[id0, id1, :, :]`` : ZX slice (for ``B = Bgrid[id0]``
              and ``Y = ygrid[id1]``)
            
            + ``u[id0, :, :, id3]`` : YX slice (for ``B = Bgrid[id0]``
              and ``Z = zgrid[id3]``)
            
            When not given, the default setting is ``slice_indexes[i]
            = u.shape[i]//2``, The slice indexes can be partially
            given, e.g., using ``slice_indexes = (5, None, 10,
            None)``, in this case, the not given indexes will be
            automatically replaced by their default values described
            above.
        
        spec_indexes : sequence of sequences of 3 int, optional        
            Used for spectral-spatial displayers only. Each element
            ``spec_indexes[i]`` is a sequence containing 3 integers,
            corresponding to the spatial indexes along the three
            spatial axes of the 4D image, and use to extract and
            display profiles.
            
            More precisely, the extracted and displayed profiles are
            the ``u[:, id[0], id[1], id[2]] for id in spec_indexes``.
            
            When not given, the default setting is ``spec_indexes =
            ((u.shape[1]//2, u.shape[2]//2, u.shape[3]//2))``.
        
        show_legend : bool, optional
            Used for spectral-spatial displayers only. Decide whether
            the legend in the spectrum display area should be visible
            or not when the figure is drawn.
        
        legend_loc : str, optional        
            Used for spectral-spatial displayers only. The location of
            the legend in the spectrum display area (see
            ``matplotlib`` documentation for possible choices).
        
        custom_spec : sequence of dict, optional        
            Used for spectral-spatial displayers only. This parameter
            can be used to specify some custom profiles to be
            displayed in the bottom axes.
            
            When given, each element of ``custom_spec`` must be a
            dictionary with the following key-values:
            
            + 'spec' : the custom profile to be displayed (monodimensional
              numpy array)
            
            + 'B' : the sampling grid (B axis) associated to the custom
              profile (monodimensional numpy array)
            
            + 'label' : label of the custom profile (str)
        
        spec_normalization : bool, optional
            Used for spectral-spatial displayers only. Decide whether
            the displayed spectra should be normalized or not. When
            spectra normalization is enabled, each displayed spectrum
            is divided by its maximum value (normalization occurs at
            display only, it does not affect the latent image)
        
        
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
                       adjust_dynamic=adjust_dynamic, newfig=newfig,
                       display_labels=display_labels, cmap=cmap,
                       grids=grids, origin=origin, aspect=aspect,
                       boundaries=boundaries, figsize=figsize,
                       indexes=indexes, src_labels=src_labels,
                       interpolation=interpolation,
                       legend_loc=legend_loc, show_legend=show_legend,
                       spec_normalization=spec_normalization,
                       slice_indexes=slice_indexes,
                       spec_indexes=spec_indexes,
                       custom_spec=custom_spec)
        
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
            #plt.ion() # removed in PR #29
        
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
                                  newfig=newfig,
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
                                  newfig=newfig,
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
            elif ndim == 4:
                self.init_display = \
                functools.partial(init_display_spectralspatial_4d,
                                  newfig=newfig, figsize=figsize,
                                  displayFcn=displayFcn,
                                  time_sleep=time_sleep, cmap=cmap,
                                  grids=grids, origin=origin,
                                  aspect=aspect,
                                  boundaries=boundaries,
                                  is_notebook=self.notebook,
                                  interpolation=interpolation,
                                  slice_indexes=slice_indexes,
                                  spec_indexes=spec_indexes,
                                  show_legend=show_legend,
                                  legend_loc=legend_loc,
                                  custom_spec=custom_spec,
                                  spec_normalization=spec_normalization)
                self.update_display = \
                functools.partial(update_display_spectralspatial_4d,
                                  is_notebook=self.notebook,
                                  displayFcn=displayFcn,
                                  adjust_dynamic=adjust_dynamic,
                                  time_sleep=time_sleep,
                                  slice_indexes=slice_indexes,
                                  spec_indexes=spec_indexes,
                                  spec_normalization=spec_normalization)
        
        else: # multisrc
            if ndim == 2:
                self.init_display = \
                functools.partial(init_display_multisrc_2d,
                                  newfig=newfig,
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
                                  newfig=newfig,
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
