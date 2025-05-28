.. _mathematical_processing:

Mathematical processing
========================

.. _mathematical_processing_fbp:

Filtered backprojection
-----------------------

In this section, we give mathematical details about the
filtered-backprojection features implemented in the PyEPRI package.
Within all this section, we assume that the sample is made of a single
EPR species :math:`X`, with reference spectrum :math:`h_X^c :
\mathbb{R}\to\mathbb{R}` and concentration mapping :math:`U_X^c:
\mathbb{R}^d \to \mathbb{R}` (reconstruction will be addressed in the
continuous setting, and discretized afterwards).

Inversion formula in the continuous setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the :ref:`Mathematical definitions <mathematical_definitions>`
section, we defined a projection :math:`\mathcal{P}_{X,\gamma}^{c}`
with field gradient :math:`\gamma = \mu \cdot e_\theta` as the
convolution between the reference spectrum :math:`h_X^c` and the
dilatation with factor :math:`-\mu` of the Radon transform of
:math:`U_X^c` in the direction :math:`e_\theta`, that is, the signal
:math:`\mathcal{R}_\theta^\mu(U_X^c) = B \mapsto \frac{1}{\mu} \,
\mathcal{R}_\theta(U_X^c)(-B/\mu)` (see Equation
:eq:`continuous-projection` and Equation :eq:`spin-profile-continuous`
of the :ref:`Mathematical definitions <mathematical_definitions>`
section). Therefore, taking the Fourier transform of such a projection
yields

 .. math ::
   :label: fourier_continuous_projection
 
   \forall \xi \in \mathbb{R}\,,\quad
   \mathcal{F}(\mathcal{P}_{X,\gamma}^c)(\xi) =
   \mathcal{F}(h_X^c)(\xi) \cdot
   \mathcal{F}(\mathcal{R}_\theta(U_X^c))(-\mu \xi)\,.

Since the reference spectrum can be modeled the derivative of an
absorption profile :math:`g_X^c`, we have
:math:`\mathcal{F}(h_X^c)(\xi) = i \xi \cdot
\mathcal{F}(g_X^c)(\xi)`. In the following, we assume that
:math:`\mathcal{F}(g_X^c)` does not vanishes. Besides, using Fourier
Slice Theorem, we have
:math:`\mathcal{F}(\mathcal{R}_\theta(U_X^c))(-\mu\xi) =
\mathcal{F}(U_X^c)(-\mu \xi e_\theta)`. Therefore, we have

 .. math ::
   :label: fourier_continuous_projection2
 
   \forall \xi \in \mathbb{R}\,,\quad \mathcal{F}(\mathcal{P}_{X,\gamma}^c)(\xi) = i \xi \cdot \mathcal{F}(g_X^c)(\xi) \cdot \mathcal{F}(U_X^c)(-\mu \xi e_\theta)\,.

Thanks to :eq:`fourier_continuous_projection2` we have an explicit
link between the Fourier transform of the image :math:`U_X^c` and that
of the observed projections :math:`\mathcal{P}_{X,\gamma}^c` that we
shall use to derive an explicit inversion formula.

2D setting
..........

We assume here that the image dimension is :math:`d=2`. In this
setting, the vector :math:`e_\theta = (\cos{\theta},\sin{\theta}) \in
\mathbb{R}^2` is parametrized by a monodimensional angle :math:`\theta
\in \mathbb{R}` (the :ref:`polar angle
<polar_and_spherical-systems>`). For any position :math:`x \in
\mathbb{R}^2`, we have

 .. math ::
    :label: build_fbp2d_continuous_inversion_1

    U_X^c(x) = \mathcal{F}^{-1}(\mathcal{F}(U_X^c))(x) =
    \frac{1}{(2\pi)^2} \int_{0}^{\pi}
    \int_{-\infty}^{+\infty} \mathcal{F}(U_X^c)(\rho e_\theta) \, e^{i
    \langle \rho e_\theta , x \rangle} \, |\rho| \, d\rho d\theta\,.

Then, using the variable change :math:`\xi = -\frac{\rho}{\mu}`, we get

 .. math ::
    :label: build_fbp2d_continuous_inversion_2

    U_X^c(x) = 
    \frac{\mu^2}{(2\pi)^2} \int_{0}^{\pi}
    \int_{-\infty}^{+\infty} \mathcal{F}(U_X^c)(-\mu \xi e_\theta) \, e^{i \xi 
    \langle -\mu e_\theta , x \rangle} \, |\xi| \, d\xi d\theta\,.

Using :eq:`fourier_continuous_projection2` and denoting :math:`\gamma_\mu(\theta) = \mu \cdot e_\theta`, we get

 .. math ::
    :label: build_fbp2d_continuous_inversion_3

    U_X^c(x) = 
    \frac{\mu^2}{(2\pi)^2} \int_{0}^{\pi}
    \int_{-\infty}^{+\infty} \mathcal{F}(\mathcal{P}_{X,\gamma_\mu(\theta)}^c)(\xi) \, \frac{-i\cdot \mathrm{sign(\xi)}}{\mathcal{F}(g_X^c)(\xi)} \, e^{i \xi 
    \langle -\gamma_\mu(\theta) , x \rangle} \, d\xi d\theta\,.

Now, let us set

 .. math ::
    :label: fbp_integral_2d

    \forall \gamma\in\mathbb{R}^2\,,\quad \forall r \in \mathbb{R}\,,\quad \mathcal{I}_{X,\gamma}^{c}(r) = \frac{1}{2\pi} \int_{-\infty}^{+\infty} \mathcal{F}(\mathcal{P}_{X,\gamma}^c)(\xi) \, \frac{-i\cdot \mathrm{sign(\xi)}}{\mathcal{F}(g_X^c)(\xi)} \, e^{i \xi r} \, d\xi \,,

which corresponds to the convolution between the projection
:math:`\mathcal{P}_{X,\gamma}^c` and the filter :math:`\mathcal{W}_X^c :=
\mathcal{F}^{-1}\left(\xi \mapsto \tfrac{-i\cdot
\mathrm{sign(\xi)}}{\mathcal{F}(g_X^c)(\xi)} \right)`, i.e.,

 .. math ::
    :label: fbp_integral_2d_convol

    \forall r \in \mathbb{R}\,,\quad \mathcal{I}_{X,\gamma}^{c}(r) = \left(\mathcal{P}_{X,\gamma}^c * \mathcal{W}_{X}^{c}\right)(r)\,,

and can thus be interpreted as a filtering of the projection
:math:`\mathcal{P}_{X,\gamma}^{c}`. Finally, injecting
:eq:`fbp_integral_2d` into :eq:`build_fbp2d_continuous_inversion_3` we
end-up with the 2D inversion formula

 .. math ::
    :label: fbp2d_continuous_formula

    \forall x \in \mathbb{R}^2 \,,\quad U_X^c(x) = 
    \frac{\mu^2}{2\pi} \int_{0}^{\pi} \mathcal{I}_{X,\gamma_\mu(\theta)}^{c}(\langle -\gamma_\mu(\theta) , x \rangle) \, d\theta\,,

which consists in integrating filtered projections (which explain the
naming of the reconstruction method). 

3D setting
..........

In the 3D setting (:math:`d=3`), the orientation vector
:math:`e_\theta =
(\cos{\theta_1}\sin{\theta_2},\sin{\theta_1}\sin{\theta_2},
\cos{\theta_2}) \in \mathbb{R}^3` is parametrized by two angles
:math:`(\theta_1,\theta_2) \in \mathbb{R}^2` corresponding to the
longitudinal (:math:`\theta_1`) and latitudinal (:math:`\theta_2`)
angles of the :ref:`spherical coordinate system
<polar_and_spherical-systems>`. The 3D inversion formula can be
derived using the same methodology as in the 2D setting, starting from
the spherical coordinate system integral formulation of the 3D inverse
Fourier transform. Indeed, for any :math:`x \in \mathbb{R}^3`, we have

 .. math ::
    :label: build_fbp3d_continuous_inversion_1

    \begin{array}{cl} U_X^c(x) &=
    \displaystyle{\mathcal{F}^{-1}(\mathcal{F}(U_X^c))(x)}\\
    &=\displaystyle{ \frac{1}{(2\pi)^3} \int_{0}^{\pi} \int_{0}^{\pi}
    \int_{-\infty}^{+\infty} \mathcal{F}(U_X^c)(\rho e_\theta) \, e^{i
    \langle \rho e_\theta , x \rangle} \, \rho^2 \sin{(\theta_2)} \,
    d\rho \, d\theta_1 \, d\theta_2}\,.  \end{array}

Setting

 .. math ::
    :label: fbp_integral_3d

    \forall \gamma\in\mathbb{R}^3\,,~ \forall r \in \mathbb{R}\,,\quad
    \mathcal{J}_{X,\gamma}^{c}(r) = \frac{1}{2\pi}
    \int_{-\infty}^{+\infty}
    \mathcal{F}(\mathcal{P}_{X,\gamma}^c)(\xi) \, \frac{-i\cdot \xi
    \sin{(\theta_2)}}{\mathcal{F}(g_X^c)(\xi)} \, e^{i \xi r} \, d\xi
    \,,

or, equivalently,

 .. math ::
    
    \mathcal{J}_{X,\gamma}^{c}(r) = \left(\mathcal{P}_{X,\gamma}^{c} *
    \mathcal{K}_{X,\gamma}^{c}\right)(r)\quad\text{where}\quad
    \mathcal{K}_{X,\gamma}^{c} = \mathcal{F}^{-1}\left(\xi \mapsto
    \frac{-i\cdot \xi
    \sin{(\theta_2)}}{\mathcal{F}(g_X^c)(\xi)}\right)
  
and setting again :math:`\gamma_\mu(\theta) = \mu e_\theta`, we can
easily rewrite :eq:`build_fbp3d_continuous_inversion_1` into the 3D
inversion formula

 .. math ::
    :label: fbp3d_continuous_formula

    \forall x \in \mathbb{R}^3 \,,\quad U_X^c(x) =
    \frac{\mu^3}{4\pi^2} \int_{0}^{\pi} \int_{0}^{\pi}
    \mathcal{J}_{X,\gamma_\mu(\theta)}^{c}(\langle -\gamma_\mu(\theta)
    , x \rangle) \, d\theta_1 \, d\theta_2\,,

which consists again in integrating some filtered projections (each
projection :math:`\mathcal{P}_{X,\gamma}^{c}` being filtered by the
:math:`\mathcal{K}_{X,\gamma}^{c}` filter).

      
Discretization scheme
~~~~~~~~~~~~~~~~~~~~~

Using the inversion formula :eq:`fbp2d_continuous_formula` (in the 2D
setting) or :eq:`fbp3d_continuous_formula` (in the 3D setting)
require to have access to the continuous projections
:math:`\mathcal{P}_{X,\gamma_\mu(\theta)}^{c}` for all orientation
:math:`\theta`, which is not possible in practice. For that reason,
practical filtered backprojection techniques rely on discretization
schemes for approaching the integrals :eq:`fbp2d_continuous_formula`
and :eq:`fbp3d_continuous_formula` from a finite number of
measurements. Many discretization strategies can be considered, we
shall describe now that currently implemented in the PyEPRI package.

In the following, we consider again a sequence containing :math:`N`
discrete projections :math:`p = (p_1, p_2, \dots p_N) \in
\left(\mathbb{R}^{I_{N_B}}\right)^N` acquired with field gradients
:math:`(\gamma_1, \gamma_2,\dots, \gamma_N) \in (\mathbb{R}^d)^N` and
sampling step :math:`\delta_B`. We denote again by :math:`u_X` the
discrete image to be reconstructed, by :math:`\delta` the associated
spatial sampling step (or pixel size), and by :math:`N_1, N_2, \dots,
N_d` the number of pixels of :math:`u_X` along each axis. We denote by
:math:`g_X` the discrete absorption profile with sampling step
:math:`\delta_B` (this signal can be estimated from the acquired
reference spectrum :math:`h_X` by using numerical integration).

2D setting
..........

A natural idea is to approach the continuous integral
:eq:`fbp2d_continuous_formula` by a Riemann sum, leading to

 .. math ::
    :label: fbp2d_build1

    \forall k \in I_{N_1} \times I_{N_2}\,,\quad u_X(k) \approx
    U_X^c(k \delta) \approx \frac{1}{2 N} \sum_{n = 1}^{N}
    \|\gamma_n\|^2 \cdot \mathcal{I}_{X, \gamma_n}^{c}(\langle
    -\gamma_n , k\delta\rangle)\,.

In this framework, it remains to evaluate the terms
:math:`\mathcal{I}_{X, \gamma_n}^{c}(\langle -\gamma_n ,
k\delta\rangle)`, which is done in two steps. First, the integrals
:math:`\mathcal{I}_{X, \gamma_n}^{c}(r)` are evaluated for values of
:math:`r` lying in a regular grid, more precisely, for :math:`r \in
\delta_B \cdot I_{N_B}`. Then, the values of the integrals
:math:`\mathcal{I}_{X,\gamma_n}(\langle -\gamma_n, k\delta \rangle)`
are evaluated by interpolating those evaluated on the the regular grid
:math:`(r_\ell := \ell \cdot \delta_B)_{\ell \in I_{N_B}}`. The
integrals :math:`\mathcal{I}_{X, \gamma_n}(r_\ell)` are approached
using another Riemann sum, by computing

 .. math ::
    :label: fbp2d_build2

    I_n(\ell) := \frac{1}{N_B \delta_B} \sum_{\alpha \in I_{N_B}}
    \mathrm{DFT}(p_n)(\alpha) \cdot \frac{-i \cdot
    \mathrm{sign}(\alpha)}{\mathrm{DFT}{(g_X)}(\alpha)} \cdot
    e^{\frac{2 i \pi \alpha \ell}{N_B}} \approx \mathcal{I}_{X,
    \gamma_n}^{c}(r_\ell).

The interest of this approach is that all values :math:`I_n(\ell)` (for
:math:`\ell \in I_{N_B}`) can be computed at once using FFT algorithms
since we have

 .. math ::
    :label: fbp2d_build3
	    
    I_n(\ell) = \frac{1}{\delta_B} \,
    \mathrm{IDFT}\left(\mathrm{DFT}(p_n) \cdot
    \widehat{w_X}\right)(\ell) \quad \text{where} \quad \widehat{w_X}
    = \alpha \mapsto \frac{-i \cdot
    \mathrm{sign}(\alpha)}{\mathrm{DFT}{(g_X)}(\alpha)}\,.


Since in practice the measured projections :math:`p_n` are corrupted
by noise, dramatic noise amplification can occur during the evaluation
of :math:`I_n(\ell)` with :eq:`fbp2d_build3` due to the presence of
Fourier coefficients :math:`\mathrm{DFT}(g_X)(\alpha)` with a small
amplitude (which typically occurs for large values of :math:`|\alpha|`
due to the rapid decay of the Fourier coefficients of :math:`g_X`). In
order to avoid this issue, we prefer in practice restricting the
bandwidth of the :math:`\widehat{w_X}` filter by replacing this filter
by

  .. math ::
     :label: fpb2d_filter

     \forall \alpha \in I_{N_B}\,,\quad \widehat{w_X}(\alpha) =
     \left\{\begin{array}{cl} \frac{-i \cdot
     \mathrm{sign}(\alpha)}{\mathrm{DFT}{(g_X)}(\alpha)} & \text{if }
     |\alpha| \leq \tau \frac{N_B}{2} \\0 & \text
     {otherwise}\end{array}\right.

in which :math:`\tau \in [0,1]` is called the *frequency cut-off*
parameter and is set by the user.
       
Once the :math:`I_n(\ell)` values are calculated, we can compute
:math:`\widetilde{I_n}(k) \approx
\mathcal{I}_{X,\gamma_n}(\langle-\gamma_n, k\delta\rangle)` for all
values of :math:`k \in I_{N_1} \times I_{N_2}` by interpolating the
values :math:`(I_n(\ell))_{\ell \in I_{N_B}}` associated to the
regularly spaced nodes :math:`\left(r_\ell\right)_{\ell \in I_{N_B}}`
onto the non-regularly spaced nodes :math:`(\rho_k := \langle
-\gamma_n, k \delta \rangle)_{k \in I_{N_1}\times I_{N_2}}`. Finally,
we end-up with the discrete reconstruction formula

 .. math ::
    :label: fbp2d_discrete
	    
    \forall k \in I_{N_1} \times I_{N_2}\,,\quad u_X(k) = \frac{1}{2 N} \sum_{n = 1}^{N} \|\gamma_n\|^2 \cdot
    \widetilde{I_n}(k)\,.

**PyEPRI implementation**: the 2D filtered backprojection
corresponding to :eq:`fbp2d_discrete` is implemented in function
:py:func:`pyepri.processing.eprfbp2d`. This implementation let the
user provides as input the image dimensions :math:`(N_1, N_2)` and the
spatial sampling step :math:`\delta` of the discrete image :math:`u_X`
to reconstruct, the frequency cut-off parameter :math:`\tau` to use in
:eq:`fpb2d_filter` and the 1D interpolation method used to evaluate
the :math:`(\widetilde{I_n}(k))_{k \in I_{N_2} \times I_{N_2}}` from
the :math:`(I_n(\ell))_{\ell \in I_{N_B}}`.

3D setting
..........

The methodology is the same as in the 2D setting. First, we approach
the continuous integral :eq:`fbp3d_continuous_formula` by a Riemann
sum, leading to

 .. math ::
    :label: fbp3d_build1

    \forall k \in I_{N_1} \times I_{N_2} \times I_{N_3}\,,\quad u_X(k) \approx U_X^c(k
    \delta) \approx \frac{1}{4 N} \sum_{n = 1}^{N} \|\gamma_n\|^3
    \cdot \mathcal{J}_{X, \gamma_n}^{c}(\langle -\gamma_n ,
    k\delta\rangle)\,.

Then, the continuous integral :math:`\mathcal{J}_{X,\gamma_n}^{c}(r)` is
evaluated over the regular grid made of the :math:`(r_\ell := \ell
\delta_B)_{\ell \in I_{N_B}}` using 

 .. math ::
    :label: fbp3d_build2
	    
    \mathcal{J}_{X,\gamma_n}^{c}(r_\ell) \approx J_n(\ell) = \frac{1}{\delta_B} \,
    \mathrm{IDFT}\left(\mathrm{DFT}(p_n) \cdot
    \widehat{\kappa_{X,n}}\right)(\ell)

where :math:`\widehat{\kappa_{X,n}} : I_{N_B} \to \mathbb{C}` is
defined by

 .. math ::
    :label: fbp3d_filter
    
    \forall \alpha \in I_{N_B}\,,\quad \widehat{\kappa_{X,n}}(\alpha) = 	    
    \left\{\begin{array}{cl}
    \frac{-2 i \pi \alpha \sin{(\theta_{2,n})}}{N_B \delta_B \mathrm{DFT}(g_X)(\alpha)}&\text{if } |\alpha| \leq \tau \frac{N_B}{2}\\
    0&\text{otherwise}\,,
    \end{array}\right.

in which :math:`\tau \in [0,1]` represents a frequency cut-off
parameter (to be set by the user) and :math:`\theta_{2,n}` corresponds
to the latitudinal angle (modulo :math:`2\pi`) associated to the field
gradient vector :math:`\gamma_n \in \mathbb{R}^3`.

Last, interpolating the values :math:`J_n(\ell) \approx
\mathcal{J}_{X,\gamma_n}^{c}(r_\ell)` associated to the regularly
spaced nodes :math:`(r_\ell)_{\ell \in I_{N_B}}` allows for the
evaluation of :math:`\widetilde{J_n}(k) \approx
\mathcal{J}_{X,\gamma_n}^{c}(\langle -\gamma_n ,
k\delta\rangle)`. Finally we end-up with the discrete reconstruction
formula

 .. math ::
    :label: fbp3d_discrete
	    
    \forall k \in I_{N_1} \times I_{N_2} \times I_{N_3}\,,\quad u_X(k)
    = \frac{1}{4 N} \sum_{n = 1}^{N} \|\gamma_n\|^3 \cdot
    \widetilde{J_n}(k)\,.

**PyEPRI implementation**: the 3D filtered backprojection
corresponding to :eq:`fbp3d_discrete` is implemented in function
:py:func:`pyepri.processing.eprfbp3d`. As in the 2D setting, this implementation let the
user provides as input the image dimensions :math:`(N_1, N_2, N_3)` and the
spatial sampling step :math:`\delta` of the discrete image :math:`u_X`
to reconstruct, the frequency cut-off parameter :math:`\tau` to use in
:eq:`fbp3d_filter` and the 1D interpolation method used to evaluate
the :math:`(\widetilde{J_n}(k))_{k \in I_{N_2} \times I_{N_2} \times I_{N_3}}` from
the :math:`(J_n(\ell))_{\ell \in I_{N_B}}`.

TV-regularized reconstruction
-----------------------------

Direct methods for solving inverse problems are generally not robust
to noise and require a large number of measurements—typically at least
as many as the number of unknowns (i.e., the number of pixels to
reconstruct). In contrast, total variation (TV)-regularized approaches
have proven to be significantly more effective.  They have been
developed for over 20 years in parallel with the development of
efficient optimization algorithms. TV-based methods can be interpreted
as sparse gradient promoting approaches, which enhances robustness to
noise and enables the reconstruction of high-quality signals from far
fewer measurements than required by direct models.

Given a :math:`d`-dimensional discrete image :math:`u: \Omega \to
\mathbb{R}` with discrete image domain :math:`\Omega := I_{N_1} \times
I_{N_2} \times \cdots \times I_{N_d}`, the TV of :math:`u` is defined
by

.. math::
   
   \mathrm{TV}(u) := \sum_{k \in \Omega} \left\|\big(\nabla u\big) (k) \right\|_2

where

.. math::
   :label: discrete-nabla

   \big(\nabla (u)\big)(k) = \bigg(\big(\nabla_1 (u)\big)(k),
   \big(\nabla_2 (u)\big)(k), \dots, \big(\nabla_d (u)\big)(k) \bigg)
   \in \mathbb{R}^d

and

.. math::
   :label: discrete-partial-nabla
   
   \forall i \in \{1, 2, \dots, d\}\,,\quad \big(\nabla_i (u)\big) (k)
   = \left\{\begin{array}{cl}u(k+\delta_i) - u(k)&\text{if } k +
   \delta_i \in \Omega\\0&\text{otherwise,}\end{array}\right.

denoting by :math:`\delta_i` the vector of :math:`\mathbb{R}^d` made
of zero everywhere expect at its :math:`i`-th entry which takes the
value one.

The term :math:`\big(\nabla (u)\big)(k)` described in
:eq:`discrete-nabla` represents a discrete gradient (or finite
differences) of :math:`u` at the pixel position :math:`k \in
\Omega`. The term :math:`\nabla_i (u)` in :eq:`discrete-partial-nabla`
represents the discrete (or finite differences) partial derivative of
:math:`u` along its :math:`i`-th coordinate axis. There exist many
other finite differences schemes (leading to as many variants of TV),
but the definition given above is the most commonly used and is the
one implemented in PyEPRI.

TV-regularized single source EPR imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us place ourselves in the single EPR source framework. As we did
earlier, we shall denote by :math:`X` the name of the EPR species
contained into the sample, and by :math:`h_X` its reference
spectrum. Let :math:`s_{X,\Gamma} = (p_{X,\gamma_1}, p_{X,\gamma_2},
\dots, p_{X,\gamma_N})` be the sequence of measured projections and
:math:`\Gamma := (\gamma_1, \gamma_2, \dots, \gamma_N)` be the
sequence of associated field gradient vectors. We can address the
image reconstruction problem using TV regularized least-squares
:cite:p:`Durand_2017, Abergel_2023` by computing

.. math::
   :label: inverse-problem-monosrc
   
   \DeclareMathOperator*{\argmin}{argmin}
   \widetilde{u}_X \in \argmin_{u_X: \Omega\to \mathbb{R}} \frac{1}{2}
   \left\| A_{X,\Gamma}(u) - s_{X,\Gamma} \right\|_2^2 + \lambda \cdot
   \mathrm{TV}(u_X)\,,
      
where :math:`A_{X,\Gamma}` denotes the single source projection
operator defined in the :ref:`Mathematical definitions Section
<mathematical_definitions_single_source_operators>` (see Equation
:eq:`Agammabold` therein), and :math:`\lambda > 0` is a parameter that
can be set by the user to tune the importance of the data-fidelity
term (the quadratic term) with respect to the regularity term (the TV
term) in the minimization process.

Such a convex and nonsmooth minimization problem can be efficiently
handled using the Condat-Vũ solver :cite:`Condat_2013, Vu_2013`
(generalized by A. Chambolle and T. Pock. in
:cite:`Chambolle_Pock_2016`). This particular scheme involves
computing the gradient of the data fidelity term, and thus of
projection-backprojection term :math:`A_{X,\Gamma}^* \circ
A_{X,\Gamma}` at each iteration, which can be efficiently computed
using Toeplitz kernels as explained :ref:`Mathematical definitions
Section <mathematical_definitions_single_source_projbackproj>`. In
particular, no evalution of the projection :math:`A_{X, \Gamma}` and
backprojection :math:`A_{X,\Gamma}^*` is needed along the scheme
iterations (see the explicit details of the numerical scheme in the
case of EPR imaging in :cite:`Abergel_2025`).

**PyEPRI implementation**: a generic solver for TV-regularized
problems is implemented in
:py:func:`pyepri.optimization.tvsolver_cp2016`. This solver can handle
more general instances than Equation :eq:`inverse-problem-monosrc`,
where the data-fidelity term is replaced by :math:`F(u)`, with
:math:`F` being a Lipschitz-differentiable function. The PyEPRI
package also provides a *higher-level* function,
:py:func:`pyepri.processing.tv_monosrc`, which specifically addresses
the single-source EPR image reconstruction problem defined in Equation
:eq:`inverse-problem-monosrc`. This function enables non-expert users
to perform reconstructions more easily, without needing to manage the
underlying optimization details. Detailed and reproducible 2D and 3D
usage examples for :py:func:`pyepri.processing.tv_monosrc` are
available in the :ref:`tutorial gallery
<example_tv_regularized_imaging>`.

Source separation
~~~~~~~~~~~~~~~~~

Let us now consider the multisource framework, in which the sample
contains more than one EPR species, denoted by :math:`\mathcal{X} =
(X_1, X_2, \dots, X_K)`. We denote by :math:`h_{X_j}` the :math:`j`-th
reference spectrum. In this multisource framework, we aim at
reconstructing a sequence of concentration mapping images
:math:`u_{\mathcal{X}} = (u_{X_1}, u_{X_2}, \dots, u_{X_K})` (one
concentration mapping for each species) rather than a single one. Let
:math:`s_{\mathcal{X},\Gamma} = (p_{\mathcal{X}, \gamma_1},
p_{\mathcal{X}, \gamma_2}, \dots, p_{\mathcal{X}, \gamma_N})` be the
sequence of measured projections with associated field gradient
vectors :math:`\Gamma := (\gamma_1, \gamma_2, \dots, \gamma_N)`. The
multi-image reconstruction was addressed in :cite:`Boussaa_2023` by
computing

.. math::
   :label: inverse-problem-multisrc-monoexp
   
   \widetilde{u}_{\mathcal{X}} \in \argmin_{u_{\mathcal{X}} =
   (u_{X_1}, u_{X_2}, \dots, u_{X_K})} \frac{1}{2} \left\|
   A_{\mathcal{X},\Gamma}(u_\mathcal{X}) - s_{\mathcal{X}, \Gamma}
   \right\|_2^2 + \lambda \sum_{j = 1}^{K} \mathrm{TV}(u_{X_j})\,,

where :math:`A_{\mathcal{X},\Gamma}` denotes the multisource
projection operator defined in the :ref:`Mathematical definitions
Section <mathematical_definitions_multisource_operators>` (see
Equation :eq:`multisrc-sino-def` therein) and :math:`\lambda > 0` is
again a regularity parameter that can be set by the user to tune the
importance of the data-fidelity term (the quadratic term) with respect
to the regularity term (the sum of TV terms) in the minimization
process.

**PyEPRI implementation**: we adopted the same development framework
as in the multisource case; specifically, we provide a generic solver,
:py:func:`pyepri.optimization.tvsolver_cp2016_multisrc`, which
addresses a more general instance of problem
:eq:`inverse-problem-multisrc-monoexp`, where the quadratic
data-fidelity term can be replaced by a more general term
:math:`F(u)`, with :math:`F` being a Lipschitz-differentiable
function. Again, a *higher-level* function,
:py:func:`pyepri.processing.tv_multisrc`, which specifically addresses
the multiple EPR images reconstruction problem (a.k.a. the source
separation problem) defined in Equation
:eq:`inverse-problem-multisrc-monoexp`. This function enables
non-expert users to perform reconstructions more easily, without
needing to manage the underlying optimization details. Detailed and
reproducible 2D and 3D usage examples for
:py:func:`pyepri.processing.tv_multisrc` are available in the
:ref:`tutorial gallery <example_source_separation>`.
