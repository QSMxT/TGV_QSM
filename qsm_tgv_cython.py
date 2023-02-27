from numpy import *
import matplotlib
matplotlib.use('gtkAgg')
import sys
import multiprocessing
import cPickle

import pyximport; pyximport.install()
from qsm_tgv_cython_helper import *

def disp(str):
    sys.stdout.write(str)
    sys.stdout.flush()

def qsm_tgv(laplace_phi0, mask, res, alpha=(0.2, 0.1), iter=1000, vis=False):
    """qsm_tgv(laplace_phi0, mask, res, alpha, iter, vis)
    
    Performs one-step QSM reconstruction with TV/TGV regularization
    by a first-order primal-dual iteration.
    
    Parameters
    ----------

    laplace_phi0 : 3d array
        The laplacian of the wrapped measured phase.
    mask : 3d array
        The 3D pixel mask for the region of interest.
    res : tuple
        The (x,y,z) spatial resolution of the data.
    alpha : tuple, optional
        The regularization parameters for TV/TGV.
        alpha = (alpha1) - TV regularization with alpha1.
        alpha = (alpha0, alpha1) - TGV regularization of second order
                                   with parameters alpha0 and alpha1.
    iter: int, optional
        The number of iterations for the first-order primal-dual
        algorithm.
    vis : bool, optional
        If true, shows basic visualization for some iterates."""

    laplace_phi0 = require(laplace_phi0, float32, 'C')
    mask = require(mask != 0, float32, 'C')
    dtype = laplace_phi0.dtype

    # erode mask
    mask0 = zeros_like(mask)
    erode_mask(mask0, mask)

    # get shapes
    phi_shape = laplace_phi0.shape
    grad_phi_shape = list(phi_shape)
    grad_phi_shape.append(3)
    hess_phi_shape = list(phi_shape)
    hess_phi_shape.append(6)

    # initialize primal variables
    chi = zeros(phi_shape, dtype=dtype, order='C')
    chi_ = zeros(phi_shape, dtype=dtype, order='C')

    w = zeros(grad_phi_shape, dtype=dtype, order='C')
    w_ = zeros(grad_phi_shape, dtype=dtype, order='C')

    phi = zeros(phi_shape, dtype=dtype, order='C')
    phi_ = zeros(phi_shape, dtype=dtype, order='C')

    # initialize dual variables
    eta = zeros(phi_shape, dtype=dtype, order='C')
    p = zeros(grad_phi_shape, dtype=dtype, order='C')
    q = zeros(hess_phi_shape, dtype=dtype, order='C')

    # estimate squared norm
    grad_norm_sqr = 4*(sum(1/(res**2)))
    wave_norm_sqr = (1.0/3.0*(1.0/(res[0]**2 + res[1]**2)) \
                     + 2.0/3.0*(1.0/res[2]**2))**2
    #norm_sqr = 0.5*(wave_norm_sqr + 2*grad_norm_sqr +
                     #sqrt((wave_norm_sqr - 1)**2 + 4*grad_norm_sqr)
                     #+ 1) #TODO
    norm_sqr = 2*grad_norm_sqr**2 + 1

    # set mode and regularization parameters
    if type(alpha) == tuple:
        TGVmode = True
        alpha1 = float32(alpha[1])
        alpha0 = float32(alpha[0])
    else:
        TGVmode = False
        alpha1 = float32(alpha)

    # initialize resolution
    res0 = float32(abs(res[0]))
    res1 = float32(abs(res[1]))
    res2 = float32(abs(res[2]))

    k = 0
    if vis:
        print("Starting QSM reconstruction (%s mode)..."
              % ("TGV" if TGVmode else "TV"))
        visInit = False
    while k < iter:
        print "Iteration %d" % k
        tau = float32(1.0/sqrt(norm_sqr))
        sigma = float32((1.0/norm_sqr)/tau)

        #############
        # dual update

        if vis:
            print "updating eta..."
        tgv_update_eta(eta, phi_, chi_, laplace_phi0,
                       mask0, sigma, res0, res1, res2)

        if vis:
            print "updating p..."
        tgv_update_p(p, chi_, w_, mask, mask0, sigma, alpha1,
                     res0, res1, res2)

        if TGVmode:
            if vis:
                print "updating q..."
            tgv_update_q(q, w_, mask0, sigma, alpha0, res0, res1, res2)

        #######################
        # swap primal variables

        (phi_, phi) = (phi, phi_)
        (chi_, chi) = (chi, chi_)
        if TGVmode:
            (w_  , w  ) = (w  , w_  )

        ###############
        # primal update

        if vis:
            print "updating phi..."
        tgv_update_phi(phi, phi_, eta, mask, mask0, tau,
                       res0, res1, res2)

        if vis:
            print "updating chi..."
        tgv_update_chi(chi, chi_, eta, p, mask0, tau,
                       res0, res1, res2)

        if TGVmode:
            if vis:
                print "updating w..."
            tgv_update_w(w, w_, p, q, mask, mask0, tau, res0, res1, res2)

        ######################
        # extragradient update

        if vis and TGVmode:
            print "updating phi_, chi_, w_..."
        if vis and not TGVmode:
            print "updating phi_, chi_ ..."

        extragradient_update(phi_.ravel(), phi.ravel())
        extragradient_update(chi_.ravel(), chi.ravel())
        if TGVmode:
            extragradient_update(w_.ravel(), w.ravel())

        if (vis and (k % 50 == 0)):
            if visInit:
                figure(1)
                img1.set_data(chi[:,:,chi.shape[2]/2])
                draw()
                figure(2)
                img2.set_data(phi[:,:,phi.shape[2]/2])
                draw()
            else:
                ion()
                figure(1)
                img1 = imshow(chi[:,:,chi.shape[2]/2], cmap=cm.gray, vmin=-pi, vmax=pi)
                draw()
                figure(2)
                img2 = imshow(phi[:,:,phi.shape[2]/2], cmap=cm.gray, vmin=-pi, vmax=pi)
                draw()
                visInit = True

        k += 1

    return (chi)
