%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% *************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ***************
% Copyright (c) 2018 [Thomson Licensing]
% All Rights Reserved
% 
% This program contains proprietary information which is a trade 
% secret/business secret of [Thomson Licensing] and is protected, even if 
% unpublished, under applicable Copyright laws (including French droit 
% d'auteur) and/or may be subject to one or more patent(s).
% 
% Recipient is to retain this program in confidence and is not permitted to
% use or make copies thereof other than as permitted in a written agreement
% with [Thomson Licensing] unless otherwise expressly allowed by applicable
% laws or by [Thomson Licensing] under express agreement.
% 
% Thomson Licensing is a company of the group TECHNICOLOR
% *************************************************************************
% This script permits one to reproduce the experiments in Section 5.1 of:
%   G. Puy and P. Pérez, "Structured sampling and fast reconstruction of
%   smooth graph signals", Information and Inference: A Journal of the IMA,
%   2018.
%
% Author: Gilles Puy
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;
addpath tools\

%% Parameters for simulations
% Choose on graph: bunny or minnesota
G = prepare_minnesota; % or G = prepare_bunny 
% Choose band-limit
band_limit = 5;
% Choose sampling strategy:
%   'uniform' for uniform distribution;
%   'optimal' for the distribution denoted $p^*$ in [1];
%   'estimated_fro' for the distribution denoted \bar{q} in [1];
%   'estimated_spec', for the distribution denoted \bar{p} in [1].
% [1] G. Puy and P. Pérez, "Structured sampling and fast reconstruction of 
% smooth graph signals", Information and Inference: A Journal of the IMA, 
% 2018.
sampling = 'uniform';

%% Required precomputations
nb_groups = max(G.groups);
G = gsp_estimate_lmax(G);
G = gsp_compute_fourier_basis(G);
Uk = G.U(:, 1:band_limit); % Truncated Fourier matrix

%% Compute lowest RIP constant
% Init.
min_eig = zeros(500, nb_groups);
% Perform several random sampling
for ind_sim = 1:500

    %
    fprintf('Simulation %d over 500.\n', ind_sim)
    
    % Estimate sampling distribution
    switch sampling
        case 'uniform'
            sampling_prob = ones(nb_groups, 1);
        case 'optimal'        
            sampling_prob = zeros(nb_groups, 1);
            for ind_gr = 1:nb_groups
                sampling_prob(ind_gr) = norm(Uk(G.groups==ind_gr, :))^2;
            end
        case 'estimated_fro'
            param_est.nb_estimation = 1;
            [~, temp] = gsp_estimate_lk(G, band_limit, param_est);
            sampling_prob = zeros(nb_groups, 1);
            for ind = 1:nb_groups
                sampling_prob(ind) =  sum(temp(G.groups==ind, :));
            end
        case 'estimated_spec'
            param_est.nb_estimation = 1;
            lk = gsp_estimate_lk(G, band_limit, param_est);
            sampling_prob = estimate_groupcoh(G, lk, G.groups, param_est);
    end
    sampling_prob = sampling_prob/sum(sampling_prob(:));
 
    % For each number of measurements
    for nb_meas = 1:nb_groups

        % Random draw of groups of nodes 
        chosen_groups = datasample(1:nb_groups, nb_meas, 'Replace', true, ... 
            'Weights', sampling_prob);
        
        % Extract selected nodes in drwan groups
        diag_P = []; chosen_nodes = [];
        for ind_g = 1:numel(chosen_groups)
            new_nodes = find(G.groups==chosen_groups(ind_g));
            chosen_nodes = [chosen_nodes ; new_nodes];
            diag_P = [diag_P; sampling_prob(chosen_groups(ind_g))*ones(size(new_nodes))];
        end
        
        % Sampling matrix
        M = sparse(1:numel(chosen_nodes), chosen_nodes, 1, ...
            numel(chosen_nodes), G.N);
        P = sparse(1:numel(chosen_nodes), 1:numel(chosen_nodes), ...
            1./sqrt(diag_P), numel(chosen_nodes), numel(chosen_nodes));
        
        % Compute eigenvalues
        A = P*M*Uk;
        A = (A'*A)/nb_meas;
        e = eig(A);
        min_eig(ind_sim, nb_meas) = min(e);
        
    end
end

%% Display results

% Eigenvalues
figure(1); hold all;
prob = sum(min_eig>0.005, 1)/size(min_eig, 1);
plot(1:nb_groups, prob, '-+');
title(['k = ', int2str(band_limit)])

% Sampling distribution
figure(2); clf;
param.show_edges = 0;
gsp_plot_signal(G, sampling_prob(G.groups), param);
colormap(flipud(hot));
axis square
