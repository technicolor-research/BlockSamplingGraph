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
This module implement functions to construct image graphs based on similarity \
between image patches, and needed to conduct experiments similar to those \
presented in Section 5.2 of:
    G. Puy and P. Pérez, "Structured sampling and fast reconstruction of \
    smooth graph signals", Information and Inference: A Journal of the IMA, \
    2018.

Author: Gilles Puy
*******************************************************************************
"""


import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors as NN


def extract_patches(image, n):
    """
    Extract an n x n patch around each pixel in an image.
    
    Inputs:
        - `image`: image on which to extract patches.
        - `n`: size of the patches to extract.
    
    Outputs:
        - `patches`: array of size nb_pixels x size_patch.
    """

    # --- Sizes
    imsize = image.shape
    nb_patches = int(imsize[0]*imsize[1])
    around = int(np.floor(n/2))

    # Handle grayscale and color image inthe same way
    if image.ndim == 2:
        image.shape = (imsize[0], imsize[1], 1)
        imsize = image.shape

    # --- Mirror image
    image_large = np.zeros((imsize[0]+n-1, imsize[1]+n-1, imsize[2]))
    # Copy image in the center
    image_large[around:-around, around:-around, :] = image
    # Mirror at the top
    image_large[:around, around:-around, :] = image[around-1::-1,:,:]
    # Mirror at the bottom
    image_large[-around:, around:-around, :] = image[-1:-around-1:-1,:,:]
    # Mirror at the left
    image_large[around:-around, :around, :] = image[:,around-1::-1,:]
    # Mirror at the right
    image_large[around:-around, -around:, :] = image[:,-1:-around-1:-1,:]
    # Mirror top left corner
    image_large[:around, :around, :] = image[around-1::-1,around-1::-1,:]
    # Mirror top right corner
    image_large[:around, -around:, :] = image[around-1::-1,-1:-around-1:-1,:]
    # Mirror bottom left corner
    image_large[-around:, :around, :] = image[-1:-around-1:-1,around-1::-1,:]
    # Mirror bottom right corner
    image_large[-around:, -around:, :] = image[-1:-around-1:-1,-1:-around-1:-1,:]

    # --- Extract all patches
    ind_block = 0
    patches = np.zeros((nb_patches, imsize[2]*(n**2)), dtype='float')
    for i in range(around, imsize[0]+around, 1):
        for j in range(around, imsize[1]+around, 1):
            temp = image_large[i-around:i+around+1, j-around:j+around+1, :]
            patches[ind_block, :] = temp.ravel()
            ind_block += 1

    return patches

def get_nearest_neighbour_graph(patches, nb_neighbours=9):
    """
    Construct a nearest neighbour graph using patches extracted from an image.
    
    Input:
        - `patches`: array of patches of size nb_pixels x size_patch.
        
    Optional input:
        - `nb_neighbors`: number of nearest neighbors to construct the graph.
    
    Outputs:
        - `L`: the combinatorial Laplacian `L` = `D` - `W`.
        - `D`: the diagonal degree matrix.
        - `W`: the adjacency matrix.
    """
    
    # Find nearest neighbours
    modelNN = NN(n_neighbors=nb_neighbours+1, algorithm='auto',
                 n_jobs=6, metric='l2').fit(patches)
    NLdistances, NLindices = modelNN.kneighbors(patches)
        
    # Construct adjacency matrix
    # ... edges
    edges = np.arange(NLindices.shape[0], dtype='uint')
    edges = np.repeat(edges[:, np.newaxis], NLindices.shape[1], 1)
    edges = (edges.flatten(), NLindices.flatten())
    # ... weights
    std = np.percentile(NLdistances[:,1:], 25)
    weights = np.exp(-NLdistances**2/std**2).flatten()
    # ... sparse matrix
    W = sp.csr_matrix((weights, edges), (patches.shape[0], patches.shape[0]))
    # ... force symmetry
    W = (W + W.T)/2.
    W.sum_duplicates()
    
    # Degree matrix 
    D = sp.diags(W.sum(0).A[0], 0)
    
    # Laplacian
    L = D - W
    
    return L, D, W
