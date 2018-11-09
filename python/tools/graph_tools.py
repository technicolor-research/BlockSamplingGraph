# -*- coding: utf-8 -*-
"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved

This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d’auteur) and/or may be \
subject to one or more patent(s).

Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.

Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This module (re-)implement base functions of graph signal processing needed to \
conduct experiments similar to those presented in Section 5.2 of:
    G. Puy and P. Pérez, "Structured sampling and fast reconstruction of \
    smooth graph signals", Information and Inference: A Journal of the IMA, \
    2018.

Author: Gilles Puy

****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
"""

import sys
import numpy as np
import multiprocessing as mp


def graph_filtering(L, signals, coeff, lmax):
    """
    Fast filtering on graphs using Chebychev or Jackson-Chebychev polynomials.
    
    Inputs:
        - `L`: dense of sparse graph Laplacian.
        - `signals`: array of signals to filter (one per column)
        - `coeff`: list/array of polynomial coefficients that approximates the \
        desired graph filter. For bandpass filters, these coefficients can be \
        computed using the function bandpass_chebychev_coefficients.
        - `lmax`: estimation of the maximum eigenvalue of L.
    
    Outputs:
        - Array of filtered signals.
        
    For more details about the method refer to:
        - Hammond et al., `Wavelets on graphs via spectral graph theory`, ACHA, \
        30(2), 129–150, 2011.
    """
    
    # Filtering using recurrence formula of (Jackson-)Chebychev polynomials
    
    # Init. at order 1
    mid = lmax/2.
    cur = (L.dot(signals) - mid*signals)/mid
    res = coeff[0] * signals + coeff[1] * cur
    
    # Loop for order > 1
    for k in range(1, len(coeff)-1):
        new = (2./mid) * (L.dot(cur) - mid*cur) - signals
        res += coeff[k+1] * new
        signals = np.array(cur)
        cur = np.array(new)

    return res


def bandpass_chebychev_coefficients(lmax, bandpass_range, m):
    """
    Approximate an ideal bandpass filter using Chebychev or Jackson-Chebychev 
    polynomials.
    
    Inputs:
        - `lmax`: estimation of the largest eigenvalue of the Laplacian.
        - `bandpass_range = (fmin, fmax)`: bandpass frequency. We should have \
        fmin >= 0 and fmax <= lmax.
        - `m`: order of the polynomial approximation
    
    Outputs:
        - `CH`: array of m+1 Chebychev coefficients approximating the desired \
        filter.
        - `JCH`: array of m+1 Jackson-Chebychev coefficients approximating the \
        desired filter.
        
    For more details about these polynomial approximations please refer to:
        - Napoli et al., `Efficient estimation of eigenvalue counts in an \
        interval`, Numerical Linear Algebra with Applications, 23(4), 674–692, \
        2016.
    """
    
    # Check validity of bandpass range
    assert bandpass_range[0] >= 0., 'Bandpass range should be in [0, lmax]'
    assert bandpass_range[1] <= lmax, 'Bandpass range should be in [0, lmax]'
    
    # Scaling and translation to come back to the classical interval 
    # of Chebychev polynomials [-1,1]
    mid = lmax/2
    a = (bandpass_range[0]-mid)/mid
    b = (bandpass_range[1]-mid)/mid

    # Compute Chebychev coefficients that approximate the bandpass filter
    CH = np.zeros(m+1)    
    CH[0] = (1./np.pi)*(np.arccos(a) - np.arccos(b))
    for j in range(1, m+1):
        CH[j] = (2./(np.pi*j))*(np.sin(j*np.arccos(a)) - np.sin(j*np.arccos(b)))

    # Compute Jackson Chebychev coefficients that approximate the bandpass filter
    gamma = np.zeros(m+1)  
    alpha = np.pi/(m+2)
    for j in range(0, m+1):
        gamma[j] = (1./np.sin(alpha)) * (
             (1. - float(j)/(m+2))*np.sin(alpha)*np.cos(j*alpha) + \
             (1./(m+2))*np.cos(alpha)*np.sin(j*alpha) )
    JCH = np.multiply(CH, gamma);

    return CH, JCH
 
    
def estimate_lk(L, k, lmax, nb_features=0, epsilon=1e-2, order=50, 
                return_features=False, verbose=False):
    """
    Estimation of the k^th eigenvalue of the graph Laplacian

    Inputs:
        - `L`: graph Laplacian.
        - `k`: index of the eigenvalue to estimate.
        - `lmax`: estimation of the largest eigenvalue of L.
        
    Optional inputs:
        - `nb_features`: number of random feature vectors used to estimate \
        lk. Default: 2*ceil(log(L.shape[0])).
        - `epsilon`: stopping criterion.
        - `order`: order of the polynomial approximating the ideal \
        low-pass filter used during the estimation.
        - `return_features`: boolean indicating whether or not to return \
        the filtered feature vectors after estimation.
        - `verbose`: boolean indicating whether or not to display information \
        during estimation.
        
    Outputs:
        - `lk`: the estimated k^th smallest eigenvalue of L.
        - `features`: if `return_features` is True, the `nb_features` random \
        signals low-pass filtered (cut-off at `lk`) obtained after estimation. \
        These vectors can be used, e.g., as feature vectors for spectral \
        clustering, or to estimate the cumulative coherence of order k of `L`.
        
    For more details about the method please refer to:
        - Napoli et al., `Efficient estimation of eigenvalue counts in an \
        interval`, Numerical Linear Algebra with Applications, 23(4), 674–692, \
        2016.
        - Puy et al., `Random sampling of bandlimited signals on graphs`, \
        ACHA, 446-475, 44(2), 2018.
    """

    # Number of random signals used to estimate lambda_k
    if nb_features == 0:
        nb_features = 2*np.ceil(np.log(L.shape[0])).astype('int')
    
    # Generate random signals to estimate lambda_k
    signals = np.random.randn(L.shape[0], nb_features)/np.sqrt(nb_features)
    
    # Init.
    counts = 0 # Count the number of eigenvalues in [lambda_min, lambda_max]
    lambda_min = 0. # Lower bound for lambda_k
    lambda_max = lmax # Upper bound for lambda_k
    
    # Dichotomic search
    while np.abs(counts-k)>1. or (lambda_max-lambda_min)/lambda_max > epsilon:
        # Middle of the search interval
        lambda_mid = (lambda_min+lambda_max)/2
        # Filter the random signals
        JCH = bandpass_chebychev_coefficients(lmax, [0, lambda_mid], order)[1]
        features = graph_filtering(L, signals, JCH, lmax)
        # Count the number of eigenvalues
        counts = np.sum(np.power(features, 2))
        if verbose:
            out = "Estimating lambda_k:"
            out += " lmin = {0:e}, lmax = {1:e},".format(lambda_min, lambda_max)
            out += " k = {0:e}, target k is {1:e}\n".format(counts, k)
            sys.stdout.write(out)
            sys.stdout.flush()
        if counts>k:
            lambda_max = lambda_mid
        elif counts<k:
            lambda_min = lambda_mid
        else: 
            lambda_min = lambda_max = lambda_mid
            break
    
    # Final results
    lambda_k = (lambda_min + lambda_max)/2.
    
    if return_features:
        return(lambda_k, features)
    else:
        return lambda_k


def estimate_cum_coh(L, lambda_k, lmax, nb_features=0, order=50):
    """
    Estimation of the local cumulative coherence of order k.

    Inputs:
        - `L`: graph Laplacian.
        - `lambda_k`: estimation of the k^th smallest eigenvalue of L. It can \
        be estimated using the function estimate_lk.
        - `lmax`: estimation of the largest eigenvalue of L.
    
    Optional inputs:
        - `nb_features`: number of random feature vectors used to estimate \
        the local cumulative coherence. Default: 2*ceil(log(L.shape[0])).
        - `order`: order of the polynomial approximating the ideal \
        low-pass filter used during the estimation.
        
    Outputs:
        - `cumcoh`: estimated local cumulative coherence for each node.
        
    For more details about the local cumulative coherence please refer to:
        - Puy et al., `Random sampling of bandlimited signals on graphs`, \
        ACHA, 446-475, 44(2), 2018.
    """
    
    # Default value for the number of features
    if nb_features == 0:
        nb_features = 2*np.ceil(np.log(L.shape[0])).astype('int')
    
    # Low-pass filter few random signals on the graph
    signals = np.random.randn(L.shape[0], nb_features)/np.sqrt(nb_features)
    JCH = bandpass_chebychev_coefficients(lmax, [0, lambda_k], order)[1]
    features = graph_filtering(L, signals, JCH, lmax)    
    
    # Cumulative local coherence
    cumcoh = np.sum(np.power(features, 2), 1)
    
    return cumcoh


def estimate_group_cum_coh_fro(L, lambda_k, lmax, groups, nb_features=0, 
                               order=50):
    """ 
    Estimation of the local group cumulative coherence of order k defined using
    the Frobenius norm.
                
    Inputs:
        - `L`: graph Laplacian.
        - `lambda_k`: estimation of the k^th smallest eigenvalue of L. It can \
        be estimated using the function estimate_lk.
        - `lmax`: estimation of the largest eigenvalue of L.
        - `groups`: array indicating the group label of each node.
        
    Optional inputs:
        - `nb_features`: number of random feature vectors used to estimate \
        the local cumulative coherence. Default: 2*ceil(log(L.shape[0])).
        - `order`: order of the polynomial approximating the ideal \
        low-pass filter used during the estimation.
        
    Outputs:
        - `cumcoh`: estimated local group cumulative coherence.
        
    For more details about the local group cumulative coherence please refer to:
        - Puy et al., `Structured sampling and fast reconstruction of smooth \
        graph signals`, Information and Inference: A Journal of the IMA, 2018.
    """
    
    # Estimate local cumulative coherence
    local_cumcoh = estimate_cum_coh(L, lambda_k, lmax, nb_features, order)
                             
    # Sum over the groups
    cumcoh = np.zeros(groups.max()+1)
    for idg in range(groups.max()+1):
        cumcoh[idg] = local_cumcoh[groups==idg].sum()
    
    return cumcoh
    
   
def estimate_group_cum_coh_spec(L, lambda_k, lmax, groups, nb_cpu=1,
                                nb_features=0, epsilon=1e-2, max_iter=100, 
                                order=50, verbose=False):
    """
    Estimation of the local group cumulative coherence of order k defined using
    the spectral norm.

    Inputs:
        - `L`: graph Laplacian.
        - `lambda_k`: estimation of the k^th smallest eigenvalue of L. It can \
        be estimated using the function estimate_lk.
        - `lmax`: estimation of the largest eigenvalue of L.
        - `groups`: array indicating the group label of each node.
    
    Optional inputs:
        - `nb_cpu`: number of cpu used for estimation. Use nb_cpu>1 for \
        parallel computing.
        - `nb_features`: number of random feature vectors used to estimate \
        the local cumulative coherence. Default: 2*ceil(log(L.shape[0])).
        - `epsilon`: stopping criterion for one estimation.
        - `max_iter`: maximum number of iterations for one estimation.
        - `order`: order of the polynomial approximating the ideal \
        low-pass filter used during the estimation.
        - `verbose`: boolean indicating whether or not to display information \
        during estimation.
        
    Outputs:
        - `cumcoh`: estimated local group cumulative coherence.
    
    For more details about the local group cumulative coherence please refer to:
        - Puy et al., `Structured sampling and fast reconstruction of smooth \
        graph signals`, Information and Inference: A Journal of the IMA, 2018.
    """
    
    # Compute filter coefficients
    JCH = bandpass_chebychev_coefficients(lmax, [0, lambda_k], order)[1]
    
    # Arguments to pass to  __estimate_group_cum_coh_for_one_group 
    # for each group.
    tasks = [(L, lmax, groups, JCH, max_iter, verbose, epsilon, i)
                for i in range(groups.max()+1)]
    
    # Choose between sequential or parallel processing
    if nb_cpu == 1:
        cumcoh = np.zeros(groups.max()+1)
        for i in range(groups.max()+1):
            cumcoh[i] = __estimate_group_cum_coh_for_one_group(tasks[i])
    else:
        pool = mp.Pool(nb_cpu)
        cumcoh = pool.map(__estimate_group_cum_coh_for_one_group, tasks)
        pool.terminate()
        cumcoh = np.array(cumcoh)
    
    return cumcoh


def constrained_diffusion(L, meas, meas_ind, lmax, init=None, 
                          tol=1e-4, max_iter=int(1e6), verbose=False):
    """
    Solve the following constrained diffusion problem using FISTA:
        
            min_x   Tr(x' L x)   subject to    x[meas_ind] = meas
            
    Inputs:
        - `L`: graph Laplacian.
        - `meas`: constrained value of the solution at the indices `meas_ind`.
        - `meas_ind`: indices at which the solution is constrained.
        - `lmax`: estimation of the largest eigenvalue of L.
    
    Optional inputs:
        - `init`: signal used as initialisation of the optimisation.
        - `tol`: stopping criterion for one estimation.
        - `max_iter`: maximum number of iterations for one estimation.
        - `verbose`: boolean indicating whether or not to display information \
        during estimation.
        
    Outputs:
        - `sol`:estimated solution of the constrained problem.
    
    For more details about FISTA please refer to:
        - Beck et al., `A Fast Iterative Shrinkage-Thresholding Algorithm for \
        Linear Inverse Problems`, SIAM J. Imaging Sci., 2(1), 183-202, 2009.
    """
    
    # Init.
    tk = 1.
    if init is None:
        init = np.zeros(L.shape[0])
    midpoint = np.array(init)
    old_sol = np.array(init)
    
    # Iterate
    for ind_iter in range(max_iter):
        
        # Gradient step
        sol = midpoint - L.dot(midpoint)/lmax
        
        # Projection onto measurement constraints
        sol[meas_ind] = meas
        
        # Inertial term for FISTA
        tk1 = (1.+np.sqrt(1.+tk**2.))/2.
        midpoint = sol + (tk-1)/tk1 * (sol - old_sol)
        
        # Log.
        if verbose and ind_iter%10 == 0:
            obj = np.dot(sol.T, L.dot(sol))
            out = '  Iter {0:d}, obj = {1:e}\n'.format(ind_iter, obj)
            sys.stdout.write(out); sys.stdout.flush()
        
        # Stopping criterion
        rel = np.linalg.norm(sol - old_sol)
        if rel < tol*np.linalg.norm(sol):
            break
        
        # Keep in memory previous estimate
        tk = tk1
        old_sol = np.array(sol)
        
    return sol

    
def __estimate_group_cum_coh_for_one_group(args):
    """
    Private function used to estimate the group cumulative coherence, defined
    using the spectral norm, for the group 'idg'.
    """
    
    # Inputs
    L, lmax, groups, JCH, max_iter, verbose, epsilon, idg = args
    
    # Indicator vector for the group `idg`
    indvector = (groups==idg)
            
    # Draw a random signal
    sig = np.random.randn(L.shape[0], 1)
    sig[np.logical_not(indvector)] = 0.
    sig = sig/np.sqrt(np.power(sig[indvector], 2).sum())

    # Power method
    for iter_num in range(max_iter):
        
        # Store previous estimated eigenvector (to monitor evolution)
        sig_old = np.array(sig)
        
        # Filtering
        X = graph_filtering(L, sig, JCH, lmax)
        
        # Normalise signal per group and estimate group coherence
        cumcoh = np.dot(sig[indvector].T, X[indvector])
        sig[indvector] = X[indvector]/np.sqrt(np.power(X[indvector], 2).sum())
    
        # Control evolution
        err = np.power(sig[indvector] - sig_old[indvector], 2).sum()/ \
            np.power(sig[indvector], 2).sum()
        err = np.sqrt(err)
        if verbose:
            out = "   Estimating group coherence:"
            out += " group = {0:d}, iter = {1:d},".format(idg, iter_num)
            out += " err. = {0:e}\n".format(np.asscalar(err))
            sys.stdout.write(out)
            sys.stdout.flush()
        if err<epsilon:
            break
    
    if verbose:
        out = "Final value:"
        out += " group = {0:d}, ".format(idg)
        out += " total iter = {0:d}, ".format(iter_num)
        out += " coh. = {0:e}\n".format(np.asscalar(cumcoh))
        sys.stdout.write(out)
        sys.stdout.flush()
    
    return np.asscalar(cumcoh)
   
