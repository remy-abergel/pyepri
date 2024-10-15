"""This module contains generic solvers for optimization problems. See
the :py:mod:`pyepri.processing` module for higher level usage of those
generic solvers to address EPR imaging reconstructions.

"""
import math
import types
import matplotlib.pyplot as plt
import pyepri.checks as checks
import pyepri.displayers as displayers

def tvsolver_cp2011(y, lbda, A, adjA, LA, grad, div, Lgrad,
                    backend=None, init=None, tol=1e-7, nitermax=1000,
                    gain=1., verbose=False, video=False,
                    eval_energy=False, displayer=None, Ndisplay=1,
                    notest=False):
    r"""Generic solver for inverse problems with quadratic data-fidelity and discrete total variation regularity.
    
    Compute a minimizer of
    
       .. math:: E(u) := \frac{1}{2} \|A(u)-y\|^2 + \lambda \cdot
                 \mathrm{TV}(u)
    
    where :math:`u` is a mono or multidimensional signal, :math:`y` is
    an input mono or multidimensional signal, :math:`\mathrm{TV}`
    denotes a discrete total variation regularizer, and
    :math:`\lambda` is a positive scalar. The minimization of
    :math:`E` is handled using Chambolle-Pock Algorithm
    :cite:p:`Chambolle_Pock_2011`.
    
    Parameters
    ----------
    
    y : array_like (with type `backend.cls`)
        Input mono or multidimensional signal.
    
    lbda : float
        Regularization parameter (TV weight :math:`\lambda` in the
        energy :math:`E` defined above).
    
    A : <class 'function'>
        Function with prototype ``v = A(u)`` and such that
        ``A(u).shape == y.shape``, corresponding to the linear
        operator to be inverted involved in E (see above).
    
    adjA : <class 'function'>
        Function with prototype ``u = adjA(v)`` corresponding to the
        adjoint of the linear operator A.
    
    LA : float
        An upper bound for the l2 induced norm of the operator ``A``.
    
    grad : <class 'function'>
        Function with prototype ``g = grad(u)``, used to evaluate the
        discrete gradient of the (mono or multidimensional) signal `u`
        involved in `E`. The signal ``g`` returned by this function
        must have shape ``(u.ndim,) + u.shape`` and be such that
        ``g[j]`` represents the discrete gradient of ``u`` along its
        j-th axis.
    
    div : <class 'function'>
        Function with prototype ``d = div(g)``, corresponding to the
        opposite adjoint of the ``grad`` operator.
    
    Lgrad : float
        An upper bound for the l2 induced norm of the ``grad``
        operator.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input array ``y``.
    
    init : array_like (with type `backend.cls`)
        Initializer for the primal variables of the numerical
        optimization scheme.
    
    tol : float, optional
        A tolerance parameter used to stop the scheme iterations when
        the relative error between the latent signal ``u`` and its
        previous value is less that ``tol``.
    
    nitermax : int, optional
        Maximal number of iterations for the numerical optimization
        scheme.
    
    gain : float, optional
        Parameter that can be used to reweight the default primal and
        dual time steps (parameters tau and sigma of the
        Chambolle-Pock Algorithm), using
        
        + tau = gain * .99 / L
        + sigma = (1. / gain) * .99 / L 
        
        where ``L = np.sqrt(LA**2 + (lbda*Lgrad)**2)``
        
        The convergence of the numerical scheme is ensured for any
        positive value of the gain, however, the setting of this
        parameter may drastically affect the practical convergence
        speed of the algorithm.
    
    verbose : bool, optional
        Enable / disable verbose mode. Set ``verbose = True`` to
        display the latent iteration index :math:`k`, the relative
        error between :math:`u^{(k)}` and :math:`u^{(k-1)}`, and, when
        ``evalE`` is given, :math:`E(u^{(k)})`), each time the
        iteration index :math:`k` is a multiple of ``Ndisplay``.
    
    video : bool, optional 
        Enable / disable video mode.

    eval_energy : bool, optional
        Enable / disable energy computation (set ``eval_energy=True``
        to compute the energy of the latent variable ``u`` each
        ``Ndisplay`` iteration).
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional
        When video is True, the attribute ``displayer.init_display``
        and ``displayer.update_display`` will be used to display the
        latent array_like ``u`` along the iteration of the numerical
        scheme.
    
        When not given (``displayer=None``), a default displayer will
        be instantiated (supported signals are 2D or 3D
        array_like). In this situation, the appropriate signal
        displayer is inferred from ``init`` (see
        :py:mod:`pyepri.displayers` documentation for more details).
    
    Ndisplay : int, optional
        Can be used to limit energy evaluation (when ``evalE`` is
        given), image display (when video mode is enabled), and
        verbose mode to the iteration indexes ``k`` that are multiples
        of Ndisplay (useful to speed-up the process).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : dict
        A dictionary with content ``{'u': u, 'ubar': ubar, 'p': p,
        'q':q, 'E': E, 'iter': iter}`` where
        
        + ``u``: (array_like with type `backend.cls`) is the output
          signal :math:`u` involved in the optimization scheme (when
          convergence is reached, this is a minimizer of :math:`E`).
        
        + ``ubar``: (array_like with type `backend.cls`) is the output
          signal :math:`\overline{u}` involved in the optimization
          scheme (when convergence is reached, ``ubar`` is the same as
          ``u``).
        
        + ``p``: (array_like with type `backend.cls`) is the output
          dual signal :math:`p` involved in the optimization scheme.
        
        + ``q``: (array_like with type `backend.cls`) is the output
          dual signal :math:`q` involved in the optimization scheme,
          the couple ``(p, q)`` is a solution of a dual formulation of
          the initial problem).
        
        + ``E``: (array_like with type `backend.cls` or ``None``) when
          ``eval_energy`` is ``True``, ``E`` is a one dimensional
          array containing the energy values computed each
          ``Ndisplay`` iterations, (``E[k]`` is the energy of
          :math:`u^{(n(k))}` where ``n(k) = k *
          Ndisplay``). Otherwise, when ``eval_energy`` if ``False``,
          ``E`` takes the value ``None``.
        
        + ``iter``: (int) the iteration index when the scheme was
          stopped.
    
    Notes
    -----
    
    The numerical scheme is designed to address a primal-dual
    reformulation of the initial problem involving two dual variables
    (one coming from the TV term, and one coming from the quadratic
    data-fidelity term) denoted below by :math:`p` and :math:`q`. 

    Given :math:`u^{(0)}` (whose value is set through the ``init``
    input parameter), set :math:`\overline{u}^{(0)} = u^{(0)}`,
    :math:`p^{(0)} = 0`, :math:`q^{(0)} = 0`, and iterate for
    :math:`k \geq 0`
    
    .. math :: 
       p^{(k+1)} &= \Pi_{\mathcal{B}}(p^{(k)} + \sigma \lambda
       \nabla \overline{u}^{(k)})
    
       q^{(k+1)} &= \frac{q^{(k)} + \sigma (A \overline{u}^{(k)} -
       y)}{1 + \sigma}
       
       u^{(k+1)} &= u^{(k)} + \tau \lambda \mathrm{div}(p^{(k+1)}) -
       \tau A^*q^{(k+1)}
       
       \overline{u}^{(k+1)} &= 2 u^{(k+1)} - u^{(k)}
    
    where 
    
    + :math:`\nabla u^{(k)}` corresponds to the discrete gradient
      (input parameter ``grad``) of :math:`u^{(k)}`;
    
    + :math:`\mathrm{div}` corresponds to the opposite of the adjoint
      of :math:`\nabla` (input parameter ``div``);
    
    + :math:`A^*` corresponds to the adjoint of the :math:`A` operator
      (input parameter ``adjA``);
    
    + :math:`\Pi_{\mathcal{B}}` corresponds to the orthogonal
      projection over a convex and closed dual unit ball `B` related
      to the TV regularity term involved in :math:`E`;
    
    + :math:`\tau` and :math:`\sigma` correspond to the primal and
      dual steps of the scheme (convergence toward a solution of the
      problem is guaranteed when :math:`\tau \sigma < |||A|||^2 +
      \lambda^2 |||\nabla|||^2`, denoting by :math:`|||\cdot|||` the
      l2 induced norm).
    
    
    See also
    --------
    
    pyepri.displayers
    tvsolver_cp2016

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(y=y)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(True, backend, y=y, lbda=lbda, A=A,
                          adjA=adjA, LA=LA, grad=grad, div=div,
                          Lgrad=Lgrad, init=init, tol=tol,
                          nitermax=nitermax, gain=gain,
                          verbose=verbose, video=video,
                          eval_energy=eval_energy,
                          displayer=displayer, Ndisplay=Ndisplay)
    
    # initialize primal variables
    if init is None :
        u = adjA(y)
        u.flags.writeable = True
    else :
        u = backend.copy(init)
    ubar = backend.copy(u)
    
    # retrieve y data-type in str format
    dtype = backend.lib_to_str_dtypes[y.dtype]
    
    # initialize dual variables
    ushape = u.shape
    yshape = y.shape
    udim = len(u.shape)
    p = backend.zeros((udim,) + ushape, dtype=dtype)
    q = backend.zeros(yshape, dtype=dtype)
    
    # compute primal and dual time steps
    L = backend.sqrt(LA**2 + (Lgrad * lbda)**2)
    tau = gain * .99 / L
    sigma = (1. / gain) * .99 / L
    
    # precompute g = grad(ubar) and delta = A(ubar) - y
    g = grad(ubar)
    delta = A(ubar) - y
    
    # allocate memory for E (if needed)
    if eval_energy:
        E = backend.zeros((1+nitermax//Ndisplay,),dtype=dtype)
        E[0] = .5*(delta**2).sum() + lbda*backend.sqrt((g**2).sum(axis=0)).sum()
    else:
        E = None
    
    # deal with video mode
    if video:
        if displayer is None:
            displayer = displayers.create(backend.to_numpy(u))
        fg = displayer.init_display(backend.to_numpy(u))
        fgnum = displayer.get_number(fg)
    
    # main loop
    iter = 0
    stop = iter >= nitermax
    while not stop:
        
        # update dual variable p
        p += (sigma * lbda) * g
        p /= backend.maximum(backend.sqrt((p**2).sum(axis=0)), 1)
        
        # update dual variable q
        q += sigma * delta
        q /= (1 + sigma)
        
        # update u and ubar from (px,py) and q
        ubar = -u
        u += tau * lbda * div(p) - tau * adjA(q)
        ubar += 2. * u
        
        # update g = discrete gradient of ubar
        g = grad(ubar)
        
        # update delta = A(ubar) - y
        delta = A(ubar) - y
        
        # compute relative error between u^{iter+1} and u^{iter} (use
        # ubar = 2*u^{iter+1} - u^{iter})
        deltasquare = ((ubar - u)**2).sum().item()
        usquare = (u**2).sum().item()

        # update stopping criterion
        iter += 1
        stop = (iter >= nitermax) or (deltasquare < tol**2 * usquare)
        
        # deal with energy evaluation
        if eval_energy and (0 == iter % Ndisplay) :
            E[iter//Ndisplay] = .5 * (delta**2).sum() + \
            lbda*(backend.sqrt((g**2).sum(axis=0))).sum()
        
        # deal with verbose and video modes
        if (verbose or video) and 0 == iter%Ndisplay :
            rel = math.sqrt(deltasquare / usquare)
            if eval_energy:
                msg = "iteration %d : rel = %.2e, E = %.17e" \
                    % (iter, rel, E[iter//Ndisplay])
            else:
                msg = "iteration %d : rel = %.2e" % (iter, rel)
            if verbose:
                print(msg)
            if video and plt.fignum_exists(fgnum):
                displayer.update_display(backend.to_numpy(u), fg)
                displayer.title(msg)
                displayer.pause()
        
        # check stopping criterion
        if stop or (video and not plt.fignum_exists(fgnum)) :
            if E is not None:
                E = E[0:iter//Ndisplay+1]
            break
    
    # close figure when code is running on interactive notebook
    if video and displayer.notebook:
        displayer.clear_output()
    
    # prepare and return output
    out = {'u': u, 'ubar': ubar, 'p': p, 'q': q, 'E': E, 'iter': iter}
    
    return out
    
def tvsolver_cp2016(init, gradf, Lf, lbda, grad, div, Lgrad,
                    backend=None, tol=1e-7, nitermax=1000, evalE=None,
                    verbose=False, video=False, displayer=None,
                    Ndisplay=1, notest=False):
    r"""Generic solver for inverse problems with Lipschitz differentiable data-fidelity term and discrete total variation regularization.
    
    Compute a minimizer of
    
       .. math:: E(u) := f(u) + \lambda \cdot \mathrm{TV}(u)
    
    where :math:`u` is a mono or multidimensional signal, :math:`f` is
    a Lipschitz differentiable data-fidelity term, :math:`\mathrm{TV}`
    denotes a discrete total variation regularizer, and
    :math:`\lambda` is a positive scalar.
    
    The minimization of :math:`E` is handled by a Conda-Vũ (see
    :cite:p:`Condat_2013` and :cite:p:`Vu_2013`) like optimization
    scheme presented in :cite:p:`Chambolle_Pock_2016`.
    
    Parameters
    ----------
    
    init : array_like (with type `backend.cls`)
        Initializer for the primal variable :math:`u` of the scheme.
    
    gradf : <class 'function'>
        Function with prototype ``y = gradf(u)``, used to evaluate the
        gradient of the data-fidelity function (``f``) at ``u``.
    
    Lf : float
        A Lipschitz constant for the ``gradf`` operator.
    
    lbda : float 
        Regularization parameter (TV weight :math:`\lambda` in the
        energy E defined above).
    
    grad : <class 'function'>
        Function with prototype ``g = grad(u)``, used to evaluate the
        discrete gradient of the (mono or multidimensional) signal `u`
        involved in `E`. The signal ``g`` returned by this function
        must have shape ``(u.ndim,) + u.shape`` and be such that
        ``g[j]`` represents the discrete gradient of ``u`` along its
        j-th axis.
    
    div : <class 'function'>
        Function with prototype ``d = div(g)``, corresponding to the
        opposite adjoint of the ``grad`` operator.
    
    Lgrad : float 
        An upper bound for the l2 induced norm of the ``grad``
        operator.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input array ``init``.
    
    tol : float, optional
        A tolerance parameter used to stop the scheme iterations when
        the relative error between the latent signal ``u`` and its
        previous value is less that ``tol``.
    
    nitermax : int, optional
        Maximal number of iterations for the numerical optimization
        scheme.
    
    evalE : <class 'function'>
        A function with prototype ``e = evalE(u)`` that takes as input
        an array_like ``u`` with same size as ``init`` and returns
        ``E(u)``.
    
    verbose : bool, optional
        Enable / disable verbose mode. Set ``verbose = True`` to
        display the latent iteration index :math:`k`, the relative
        error between :math:`u^{(k)}` and :math:`u^{(k-1)}`, and, when
        ``evalE`` is given, :math:`E(u^{(k)})`), each time the
        iteration index :math:`k` is a multiple of ``Ndisplay``.
    
    video : bool, optional 
        Enable / disable video mode.
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional
        When video is True, the attribute ``displayer.init_display``
        and ``displayer.update_display`` will be used to display the
        latent array_like ``u`` along the iteration of the numerical
        scheme.
    
        When not given (``displayer=None``), a default displayer will
        be instantiated (supported signals are 2D or 3D
        array_like). In this situation, the appropriate signal
        displayer is inferred from ``init`` (see
        :py:mod:`pyepri.displayers` documentation for more details).
    
    Ndisplay : int, optional
        Can be used to limit energy evaluation (when ``evalE`` is
        given), image display (when video mode is enabled), and
        verbose mode to the iteration indexes ``k`` that are multiples
        of Ndisplay (useful to speed-up the process).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : dict 
        A dictionary with content ``{'u': u, 'ubar': ubar, 'p': p,
        'E': E, 'iter': iter}`` where
    
        + ``u``: (array_like with type `backend.cls`) is the output
          signal :math:`u` involved in the optimization scheme (when
          convergence is reached, this is a minimizer of :math:`E`).
    
        + ``ubar``: (array_like with type `backend.cls`) is the output
          signal :math:`\overline{u}` involved in the optimization
          scheme (when convergence is reached, ``ubar`` is the same as
          ``u``).
        
        + ``p``: (array_like with type `backend.cls`) is the output
          signal :math:`p` involved in the optimization scheme (when
          convergence is reached, ``p`` is a solution of a dual
          formulation of the initial problem).
    
        + ``E``: (array_like with type `backend.cls`) is, when
          ``evalE`` is given, a one dimensional array containing the
          energy values computed each ``Ndisplay`` iterations,
          (``E[k]`` is the energy of :math:`u^{(n(k))}` where ``n(k) = k
          * Ndisplay``). When ``evalE`` is not given, the returned
          value of ``E`` is ``None``.
        
        + ``iter``: (int) the iteration index when the scheme was
          stopped.
    
    
    Notes
    -----
    
    This function implements the following numerical scheme. Given
    :math:`u^{(0)}` (whose value is set through the ``init`` input
    parameter), set :math:`\overline{u}^{(0)} = u^{(0)}`,
    :math:`p^{(0)} = 0`, and iterate for :math:`k\geq 0`
    
    .. math :: 
       p^{(k+1)} &= \Pi_{\mathcal{B}}(p^{(k)} + \sigma \lambda
       \nabla \overline{u}^{(k)})
       
       u^{(k+1)} &= u^{(k)} + \tau \left(\lambda
       \mathrm{div}(p^{(k+1)}) \nabla f(u^{(k)})\right)
       
       \overline{u}^{(k+1)} &= 2 u^{(k+1)} - u^{(k)}
    
    where 
    
    + :math:`\nabla u^{(k)}` corresponds to the discrete gradient
      (input parameter ``grad``) of :math:`u^{(k)}`;
    
    + :math:`\mathrm{div}` corresponds to the opposite of the adjoint
      of :math:`\nabla` (input parameter ``div``);
    
    + :math:`\nabla f(u^{(k)})` corresponds to the value at the point
      :math:`u^{(k)}` of the gradient of the data-fidelity term
      :math:`f` involved in the definition of :math:`E` (input
      parameter ``gradf``);
    
    + :math:`\Pi_{\mathcal{B}}` corresponds to the orthogonal
      projection over a convex and closed dual unit ball `B` related
      to the TV regularity term involved in :math:`E`;
    
    + :math:`\tau = \frac{1}{L_f}` corresponds to the primal step of
      the scheme (and :math:`L_f` is a Lipschitz constant of
      :math:`\nabla f` (input parameter ``Lf``);
    
    + :math:`\sigma = \frac{Lf}{(\mathrm{\lambda} L_g)^2}` corresponds
      to the dual step of the scheme (and :math:`L_g` is an upper
      bound of the l2-induced norm of the :math:`\nabla` operator
      (input parameter ``Lgrad``)).
        
    
    See also
    --------
    
    pyepri.displayers
    tvsolver_cp2016_multisrc

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(init=init)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(True, init=init, gradf=gradf, Lf=Lf,
                          lbda=lbda, grad=grad, div=div, Lgrad=Lgrad,
                          backend=backend, tol=tol, nitermax=nitermax,
                          evalE=evalE, verbose=verbose, video=video,
                          displayer=displayer, Ndisplay=Ndisplay)
    
    # initialize primal variables u and ubar
    u = backend.copy(init)
    ubar = backend.copy(u)
    
    # retrieve data type in str format
    dtype = backend.lib_to_str_dtypes[init.dtype]
    
    # initialize dual variable p
    ushape = u.shape
    udim = len(u.shape)    
    p = backend.zeros((udim,)+ushape, dtype=dtype)
    
    # compute primal and dual time steps
    tau = .5 / Lf
    sigma = Lf / ((Lgrad * lbda)**2)
    
    # if needed, allocate memory for E and compute E[0]
    if evalE is not None:
        E = backend.zeros((1+nitermax//Ndisplay,), dtype=dtype)
        E[0] = evalE(u)
    else:
        E = None
    
    # deal with video mode
    if video:
        if displayer is None:
            displayer = displayers.create(backend.to_numpy(u))
        fg = displayer.init_display(backend.to_numpy(u))
        fgnum = displayer.get_number(fg)
            
    # main loop
    iter = 0
    stop = iter >= nitermax
    while not stop:
        
        # update dual variable p
        p += (sigma*lbda)*grad(ubar)
        p /= backend.maximum(backend.sqrt((p**2).sum(0)), 1)
        
        # update u and ubar from (px,py) and q
        ubar = -u
        u += tau*(lbda*div(p) - gradf(u))
        ubar += 2.*u
        
        # compute relative error between u^{iter+1} and u^{iter} (use
        # ubar = 2*u^{iter+1} - u^{iter})
        deltasquare = ((ubar - u)**2).sum().item()
        usquare = (u**2).sum().item()

        # update stopping criterion
        iter += 1
        stop = (iter >= nitermax) or (deltasquare < tol**2 * usquare)
        
        # deal with energy evaluation
        if 0 == iter % Ndisplay and evalE is not None:
            E[iter//Ndisplay] = evalE(u)
        
        # deal with verbose and video modes
        if (verbose or video) and 0 == iter % Ndisplay:
            rel = math.sqrt(deltasquare / usquare)
            if evalE is not None:
                msg = "iteration %d : rel = %.2e, E = %.10e" % \
                    (iter, rel, E[iter//Ndisplay])
            else:
                msg = "iteration %d : rel = %.2e" % (iter, rel)
            if verbose:
                print(msg)
            if video and plt.fignum_exists(fgnum):
                displayer.update_display(backend.to_numpy(u), fg)
                displayer.title(msg)
                displayer.pause()
        
        # check stopping criterion
        if stop or (video and not plt.fignum_exists(fgnum)):
            if E is not None:
                E = E[0:iter//Ndisplay+1]
            break
    
    # close figure when code is running on interactive notebook
    if video and displayer.notebook:
        displayer.clear_output()
    
    # prepare and return output
    out = {'u': u, 'ubar': ubar, 'p': p, 'E': E, 'iter': iter}
    
    return out


def tvsolver_cp2016_multisrc(init, gradf, Lf, lbda, grad, div, Lgrad,
                             backend=None, tol=1e-7, nitermax=1000,
                             evalE=None, verbose=False, video=False,
                             displayer=None, Ndisplay=1, notest=False):
    r"""Generic solver for inverse problems similar to :py:func:`tvsolver_cp2016` but with primal variable defined as a sequence of subvariables. 
    
    Compute a minimizer of
    
       .. math:: E(u) := f(u) + \lambda \cdot \sum_{j = 1}^{K} \mathrm{TV}(u_j)
    
    where :math:`u = (u_1, u_2, \dots, u_K)` is a sequence of mono or
    multidimensional signals, :math:`f` is a Lipschitz differentiable
    data-fidelity term, :math:`\mathrm{TV}` denotes a discrete total
    variation regularizer, and :math:`\lambda` is a positive scalar.
    
    The minimization of :math:`E` is handled by a Conda-Vũ (see
    :cite:p:`Condat_2013` and :cite:p:`Vu_2013`) like optimization
    scheme presented in :cite:p:`Chambolle_Pock_2016`.
    
    
    Parameters
    ----------
    
    init : sequence of array_like (with type `backend.cls`)
        Sequence ``init = (init1, init2, ..., initK)`` of initializers
        for the primal variable :math:`u` of the scheme.
    
    gradf : <class 'function'>
        Function with prototype ``y = gradf(u)``, used to evaluate the
        gradient of the data-fidelity function (``f``) at points ``u``.
    
    Lf : float
        A Lipschitz constant for the ``gradf`` operator.
    
    lbda : float 
        Regularization parameter (TV weight :math:`\lambda` in the
        energy E defined above).
    
    grad : <class 'function'>
        Function with prototype ``g = grad(v)``, used to evaluate the
        discrete gradient of the mono or multidimensional signals
        :math:`u_j` (i.e., the arrays ``v in u``) involved in `E`.
    
    div : <class 'function'>
        Function with prototype ``d = div(g)``, corresponding to the
        opposite adjoint of the ``grad`` operator.
    
    Lgrad : float 
        An upper bound for the l2 induced norm of the ``grad``
        operator.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input array ``init[0]``.
    
    tol : float, optional
        A tolerance parameter used to stop the scheme iterations when
        the relative error between the latent signal ``u`` and its
        previous value is less that ``tol``.
    
    nitermax : int, optional
        Maximal number of iterations for the numerical optimization
        scheme.
    
    evalE : <class 'function'>
        A function with prototype ``e = evalE(u)`` that takes as input
        a sequence of array_like ``u = [u[j] for j in range(K)]`` and
        returns the value of ``E(u)``.
    
    verbose : bool, optional
        Enable / disable verbose mode. Set ``verbose = True`` to
        display the latent iteration index :math:`k`, the relative
        error between :math:`u^{(k)}` and :math:`u^{(k-1)}`, and, when
        ``evalE`` is given, :math:`E(u^{(k)})`), each time the
        iteration index :math:`k` is a multiple of ``Ndisplay``.
    
    video : bool, optional 
        Enable / disable video mode.
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional
        When video is True, the attribute ``displayer.init_display``
        and ``displayer.update_display`` will be used to display the
        latent array_like ``u`` along the iteration of the numerical
        scheme.
    
        When not given (``displayer=None``), a default displayer will
        be instantiated (supported signals are sequences of 2D or 3D
        array_like). In this situation, the appropriate signal
        displayer is inferred from ``init`` (see
        :py:mod:`pyepri.displayers` documentation for more details).
    
    Ndisplay : int, optional
        Can be used to limit energy evaluation (when ``evalE`` is
        given), image display (when video mode is enabled), and
        verbose mode to the iteration indexes ``k`` that are multiples
        of Ndisplay (useful to speed-up the process).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
        
    
    Return
    ------
    
    out : dict 
        A dictionary with content ``{'u': u, 'ubar': ubar, 'p': p,
        'E': E, 'iter': iter}`` where
    
        + ``u``: (array_like with type `backend.cls`) is the output
          signal :math:`u` involved in the optimization scheme (when
          convergence is reached, this is a minimizer of :math:`E`).
    
        + ``ubar``: (array_like with type `backend.cls`) is the output
          signal :math:`\overline{u}` involved in the optimization
          scheme (when convergence is reached, ``ubar`` is the same as
          ``u``).
        
        + ``p``: (array_like with type `backend.cls`) is the output
          signal :math:`p` involved in the optimization scheme (when
          convergence is reached, ``p`` is a solution of a dual
          formulation of the initial problem).
    
        + ``E``: (array_like with type `backend.cls`) is, when
          ``evalE`` is given, a one dimensional array containing the
          energy values computed each ``Ndisplay`` iterations,
          (``E[k]`` is the energy of :math:`u^{(n(k))}` where ``n(k) =
          k * Ndisplay``). When ``evalE`` is not given, the returned
          value of ``E`` is ``None``.
        
        + ``iter``: (int) the iteration index when the scheme was
          stopped.
    
    
    See also
    --------
    
    pyepri.displayers
    tvsolver_cp2016_multisrc

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(init=init[0])

    # consistency checks
    if not notest:
        _check_nd_inputs_(False, init=init, gradf=gradf, Lf=Lf,
                          lbda=lbda, grad=grad, div=div, Lgrad=Lgrad,
                          backend=backend, tol=tol, nitermax=nitermax,
                          evalE=evalE, verbose=verbose, video=video,
                          displayer=displayer, Ndisplay=Ndisplay)
    
    # initialize primal variables u and ubar
    u = list(backend.copy(v) for v in init)
    ubar = list(backend.copy(v) for v in init)
    
    # retrieve data type in str format
    dtype = backend.lib_to_str_dtypes[init[0].dtype]
    
    # initialize dual variable p
    ushape = tuple(im.shape for im in u)
    udim = u[0].ndim
    p = list(backend.zeros((udim,)+s, dtype=dtype) for s in ushape)
    
    # compute primal and dual time steps
    tau = .5 / Lf
    sigma = Lf / ((Lgrad * lbda)**2)
    
    # if needed, allocate memory for E and compute E[0]
    if evalE is not None:
        E = backend.zeros((1+nitermax//Ndisplay,), dtype=dtype)
        E[0] = evalE(u)
    else:
        E = None
    
    # deal with video mode
    if video:
        u_np = tuple(backend.to_numpy(im) for im in u)
        if displayer is None:
            displayer = displayers.create(u_np)
        fg = displayer.init_display(u_np)
        fgnum = displayer.get_number(fg)

    # precompute gradf(u)
    gfu = gradf(u)
    
    # main loop
    iter = 0
    stop = iter >= nitermax
    while not stop:
        
        # initialize error terms 
        deltasquare = 0.
        usquare = 0.
        
        # source-wise update or primal & dual variables
        for j in range(len(u)):
            
            # update dual variable in p[j]
            p[j] += (sigma * lbda) * grad(ubar[j])
            p[j] /= backend.maximum(backend.sqrt((p[j]**2).sum(0)), 1)
            
            # update u[j] and ubar[j] from p[j]
            ubar[j] = -u[j]
            u[j] += tau * (lbda * div(p[j]) - gfu[j])
            ubar[j] += 2. * u[j]
            
            # compute relative error between u[j]^{iter+1} and
            # u[j]^{iter} (use ubar[j] = 2*u[j]^{iter+1} - u^{iter})
            # and accumulate the result into deltasquare
            deltasquare += ((ubar[j] - u[j])**2).sum().item()
            
            # compute square l2 norm of u[j] and accumulate the result
            # into usquare
            usquare += (u[j]**2).sum().item()
        
        # update stopping criterion
        iter += 1
        stop = (iter >= nitermax) or (deltasquare < tol**2 * usquare)
        
        # deal with energy evaluation
        if 0 == iter % Ndisplay and evalE is not None:
            E[iter//Ndisplay] = evalE(u)
        
        # deal with verbose and video modes
        if (verbose or video) and 0 == iter % Ndisplay:
            rel = math.sqrt(deltasquare / usquare)
            if evalE is not None:
                msg = "iteration %d : rel = %.2e, E = %.10e" % \
                    (iter, rel, E[iter//Ndisplay])
            else:
                msg = "iteration %d : rel = %.2e" % (iter, rel)
            if verbose:
                print(msg)
            if video and plt.fignum_exists(fgnum):
                u_np = tuple(backend.to_numpy(im) for im in u)
                displayer.update_display(u_np, fg)
                displayer.title(msg)
                displayer.pause()
        
        # check stopping criterion
        if stop or (video and not plt.fignum_exists(fgnum)):
            if E is not None:
                E = E[0:iter//Ndisplay+1]
            break
        
        # update gfu
        gfu = gradf(u)
    
    # close figure when code is running on interactive notebook
    if video and displayer.notebook:
        displayer.clear_output()
    
    # prepare and return output
    out = {'u': u, 'ubar': ubar, 'p': p, 'E': E, 'iter': iter}
    
    return out

def _check_nd_inputs_(is_monosrc, backend, y=None, lbda=None, A=None,
                      adjA=None, LA=None, init=None, gradf=None,
                      Lf=None, grad=None, div=None, Lgrad=None,
                      tol=None, nitermax=None, gain=None,
                      verbose=None, video=None, displayer=None,
                      Ndisplay=None, eval_energy=None, evalE=None):
    """Factorized consistency checks for functions in the :py:mod:`pyepri.optimization` submodule."""

    # check backend consistency
    if is_monosrc:
        checks._check_backend_(backend, y=y, init=init)
    else:
        checks._check_seq_(t=backend.cls, init=init)

    # lbda: must be a nonnegative float
    if (lbda is not None) and ((not isinstance(lbda, (float, int))) or (lbda < 0)):
        raise RuntimeError(            
            "Parameter `lbda` must be a nonnegative float scalar number (int is also tolerated)."
        )

    # LA: must be a nonnegative float
    if (LA is not None) and (not isinstance(LA, (float, int)) or LA < 0):
        raise RuntimeError(            
            "Parameter `LA` must be a nonnegative float scalar number (int is also tolerated)."
        )

    # tol: must be a float
    if not isinstance(tol, (float, int)):
        raise RuntimeError(            
            "Parameter `tol` must be a float scalar number (int is also tolerated)."
        )
    
    # Ndisplay: must be a positive integer
    if (not isinstance(Ndisplay, int)) or Ndisplay <= 0:
        raise RuntimeError(            
            "Parameter `Ndisplay` must be a positive int scalar number."
        )
    
    # gain: must be None or a positive float
    if gain and ((not isinstance(gain, (float, int))) or gain <= 0):
        raise RuntimeError(            
            "Parameter `gain` must be a positive float scalar number (int is also tolerated)."
        )
    
    # A, adjA, grad, div, gradf: must be some functions
    checks._check_type_(types.FunctionType, A=A, adjA=adjA, grad=grad, div=div, gradf=gradf, evalE=evalE)

    # verbose, video: must be bool
    checks._check_type_(bool, verbose=verbose, video=video)
    
    # nitermax: must be a nonnegative integer
    if not math.isinf(nitermax) and ((not isinstance(nitermax, int)) or nitermax < 0):
        raise RuntimeError(            
            "Parameter `nitermax` must be a nonnegative int scalar number."
        )
    needE = evalE or eval_energy
    if needE and math.isinf(nitermax):
        raise RuntimeError(            
            "Parameter `nitermax` must be finite when energy computation is required."
        )

    return True
