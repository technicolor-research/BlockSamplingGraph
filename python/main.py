# -*- coding: utf-8 -*-
"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved

This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s). 

Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.

Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This script permits one to conduct samplings and reconstructions similar to
those conducted in Section 5.2  of:
    G. Puy and P. Pérez, "Structured sampling and fast reconstruction of \
    smooth graph signals", Information and Inference: A Journal of the IMA, \
    2018.

Author: Gilles Puy
"""

import cv2
import sys
import time
import numpy as np
import scipy.sparse as sp
import tools.image_graph as ig
import tools.graph_tools as gt
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable


def snr(im_rec, im_ref):
    """
    Compute and return SNR between reconstructed image 'im_rec' and ground
    truth image 'im_ref'.
    """
    
    res = np.linalg.norm(im_rec-im_ref)/np.linalg.norm(im_ref)
    
    return -20*np.log10(res)


def solve_at_superpixel_resolution(L, meas_sp, sampled_sp, segments, lmax):
    """
    Solve the constrained diffusion problem (5.2) in [1] to propagate sampled \
    label to the whole image. The solution is computed at superpixel \
    resolution.
    
    Inputs:
        - `L`: graph Laplacian.
        - `meas_sp`: measurements in each sampled superpixels.
        - `sampled_sp`: indices of the sampled superpixels.
        - `segments`: array indicating the superpixel label for each pixel in \
        the image.
        - `lmax`: estimation of the maximum eigenvalue of L.
    
    Outputs:
        - `sol`: estimated solution after solving the diffusion problem at \
        superpixel resolution.
    
    [1] Puy et al., `Random sampling of bandlimited signals on graphs`, \
        ACHA, 446-475, 44(2), 2018.
    """

    # Nb. of superpixels
    nb_superpix = segments.max()+1

    # Get size of each superpixel
    segments = segments.flatten()
    superpixels_size = np.zeros(nb_superpix)
    for ind in range(nb_superpix):
        superpixels_size[ind] = (segments==ind).sum()

    # "Averaging" matrix in each superpixel
    weights = np.zeros(L.shape[0])
    for ind in range(nb_superpix):
        weights[segments==ind] = 1/np.sqrt(superpixels_size[ind])
    Low = sp.csr_matrix((weights, (np.arange(L.shape[0]), segments)),
                        [L.shape[0], nb_superpix]).T
    
    # Rescale measurements
    meas_sp = np.multiply(meas_sp, np.sqrt(superpixels_size[sampled_sp]))
    
    # Downsample Laplacian
    L = Low.dot(L.dot(Low.T))

    # Solve constrained problem at superpixels resolution
    sol = gt.constrained_diffusion(L, meas_sp, sampled_sp, lmax,
                                   init=np.zeros(nb_superpix))

    # Project to high dimension
    sol = Low.T.dot(sol[:nb_superpix])

    return sol


def solve_at_pixel_resolution(L, meas_sp, sampled_sp, segments, lmax, init):
    """
    Solve the constrained diffusion problem (5.3) in [1] to propagate sampled \
    label to the whole image. The solution is computed at pixel resolution.
    
    Inputs:
        - `L`: graph Laplacian.
        - `meas_sp`: measurements in each sampled superpixels.
        - `sampled_sp`: indices of the sampled superpixels.
        - `segments`: array indicating the superpixel label for each pixel in \
        the image.
        - `lmax`: estimation of the maximum eigenvalue of L.
        - `init`: signal used as initialisation of the optimisation problem.
    
    Outputs:
        - `sol`: estimated solution after solving the diffusion problem at \
        pixel resolution.
    
    [1] Puy et al., `Random sampling of bandlimited signals on graphs`, \
        ACHA, 446-475, 44(2), 2018.
    """

    # Convert measurement from SP level to pixel level
    segments = segments.flatten()
    ref = np.arange(segments.size, dtype='uint')
    chosen_pix = []; meas = []
    for ind in np.arange(sampled_sp.size):
        loc = (segments==sampled_sp[ind])
        chosen_pix = np.concatenate((chosen_pix, ref[loc]))
        meas = np.concatenate((meas, meas_sp[ind]*np.ones(loc.sum())))
    chosen_pix = chosen_pix.astype('uint')

    # Solve constrained problem
    sol = gt.constrained_diffusion(L, meas, chosen_pix, lmax, init=init)

    return sol


if __name__ == '__main__':

    
    # ---
    image_name = '108004'
    image_path = 'data/Semantic dataset100/image/' + image_name + '.jpg'
    seg_path = 'data/Semantic dataset100/ground-truth/' + image_name  + '.png'


    # --- Parameters
    # Bandlimit for estimation of sampling distributions
    bdl = 50
    # Number of simulations
    nb_sim = 100
    # Number of measurements
    nb_meas = 150 
    # Polynomial order for filtering on graph
    order_filter = 150 
    # Number of superpixels
    nb_superpixels = 600 
    # Sampling method. Choose in {'uniform', 'estimated_fro', 'estimated_spec'}.
    sampling_method = sys.argv[1]


    # --- Load image and segmentation map
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow('Image to segment', image); cv2.waitKey(1)
    seg_map = cv2.imread(seg_path)[:, :, 0].astype('float32')/255.
    cv2.imshow('Ground truth segmentation', seg_map); cv2.waitKey(1)
    

    # --- Construct graph
    sys.stdout.write('Constructing graph... '); sys.stdout.flush()
    # Extract RGB patches
    patches = ig.extract_patches(image, 3)
    patches = patches.reshape((patches.shape[0], 27))
    # Add pixels coordinates
    x = np.linspace(0, 255, image.shape[1])
    y = np.linspace(0, 255, image.shape[0])
    x, y = np.meshgrid(x, y)
    x = ig.extract_patches(x, 3)
    y = ig.extract_patches(y, 3)
    patches = np.concatenate((patches, x, y), 1)
    # Get nearest neighbours graph
    L, D, _ = ig.get_nearest_neighbour_graph(patches, nb_neighbours=9)
    lmax = 2*D.diagonal().max()
    sys.stdout.write('Done.\n'); sys.stdout.flush()


    # --- Get superpixels
    sys.stdout.write('Computing superperpixels... '); sys.stdout.flush()
    # Use SLIC to get superpixels
    segments = slic(image[..., ::-1], n_segments=nb_superpixels,
                    compactness=10.0, sigma=1., enforce_connectivity=False)
    # Postprocess list of superpixels to remove empty superpixels
    unique_seg = np.unique(segments)
    nb_superpixels = unique_seg.size
    # Segments with continuous list of indices
    csegments = np.zeros(segments.shape, dtype='uint')
    for ind in range(nb_superpixels):
        # Extract location of current superpixel
        loc = (segments==unique_seg[ind])
        # Construct list of superpixels indices
        csegments[loc] = ind
    segments = csegments
    # Show superpixels
    cv2.imshow('Superpixels', mark_boundaries(image, segments))
    cv2.waitKey(1)
    sys.stdout.write('Done.\n'); sys.stdout.flush()
    

    # --- Sampling distribution
    # Estimation
    sys.stdout.write('Computing probability distribution... ')
    sys.stdout.flush()
    if sampling_method == 'uniform':
        sampling_prob = np.ones(nb_superpixels)
    elif sampling_method == 'estimated_fro':
        sys.stdout.write('\n'); sys.stdout.flush()
        lambda_k = gt.estimate_lk(L, bdl, lmax, order=order_filter, verbose=True)
        sampling_prob = gt.estimate_group_cum_coh_fro(L, lambda_k, lmax,
            segments.flatten(), order=order_filter)
    elif sampling_method == 'estimated_spec':
        sys.stdout.write('\n'); sys.stdout.flush()
        lambda_k = gt.estimate_lk(L, bdl, lmax, order=order_filter, verbose=True)
        sampling_prob = gt.estimate_group_cum_coh_spec(L, lambda_k, lmax,
            segments.flatten(), nb_cpu=6, order=order_filter, verbose=True)
    else:
        raise ValueError('Sampling method not implemented.')    
    # Display sampling distribution
    cumcoh = np.zeros(segments.shape)
    for i in range(nb_superpixels):
        cumcoh[segments==i] = sampling_prob[i]
    plt.figure('Cumulative coherence ' + sampling_method)
    plt.clf()
    ax = plt.gca()
    im = ax.imshow(cumcoh)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    sys.stdout.write('Done.\n\n');
    sys.stdout.write('*** Close the window \"Cumulative coherence\" ' +
                     'to continue.\n\n')
    sys.stdout.flush()
    plt.show()    
    # Normalise
    sampling_prob = sampling_prob/sampling_prob.sum()
    
    
    # --- Perform several samplings and reconstruction
    snr_pix = np.zeros(nb_sim)
    snr_spix = np.zeros(nb_sim)
    time_pix = np.zeros(nb_sim)
    time_spix = np.zeros(nb_sim)
    for ind_sim in range(nb_sim):
        
        #
        log = 'Simulation {0:d} over {1:d}.\n'.format(ind_sim+1, nb_sim)
        sys.stdout.write(log); sys.stdout.flush()
        
        # Draw superpixels at random
        chosen_superpixels = np.random.choice(np.arange(nb_superpixels),
            size=nb_meas, replace=True, p=sampling_prob)
        
        # Get measurements in each chosen superpixel
        meas = np.zeros(chosen_superpixels.size)
        for ind in range(chosen_superpixels.size):
            mean = seg_map[segments==chosen_superpixels[ind]].mean()
            if mean>.5:
                meas[ind] = 1.0
            else:
                meas[ind] = 0.0
        
        # Reconstruction at superpixel level
        start = time.time()
        sol_spix = solve_at_superpixel_resolution(L, meas, chosen_superpixels,
                                                  segments, lmax)
        snr_spix[ind_sim] = snr(sol_spix, seg_map.flatten())
        time_spix[ind_sim] = time.time() - start
        
        # Refine solution at pixel level
        start = time.time()
        sol_pix = solve_at_pixel_resolution(L, meas, chosen_superpixels,
                                            segments, lmax, sol_spix)
        snr_pix[ind_sim]= snr(sol_pix, seg_map.flatten())
        time_pix[ind_sim] = time.time() - start
        
    # Print mean snrs
    sys.stdout.write('Mean SNR when solving at superpixel resolution: ' + 
                     str(snr_spix.mean()) + ' dB.\n')
    sys.stdout.write('Mean SNR when solving at pixel resolution: ' + 
                     str(snr_pix.mean()) + ' dB.\n')
    sys.stdout.flush()
    
    # Print mean reconstruction time
    sys.stdout.write('Mean reconstruction time when solving at superpixel ' +
                     'resolution: ' + str(time_spix.mean()) + ' seconds.\n')
    sys.stdout.write('Mean reconstruction time when solving at pixel ' + 
                     'resolution: ' + str(time_pix.mean()) + ' seconds.\n')
    sys.stdout.flush()