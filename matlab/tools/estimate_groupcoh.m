function group_coh = estimate_groupcoh(G, lambda_k, groups, param)
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
% This code permits to estimate the square of the group graph coherence
% as explained in Section 3 of [1].
%
% Inputs:
%   - G: Structure array defining the graph as in the GSP toolbox
%   - lambda_k: estimated value for the k^th eigenvalue of the graph
%   Laplacian.
%   - groups: vector indicating the group ID of node of the graph
%   - param: structure defining option for the estimation
%       param.epsilon: tolerance for stopping criterion (default: 1e-2)
%       param.verbose: print log or not during estimation (default: 0)
%       param.max_iter: maximum number of iteration (default: 500)
%       param.jackson: Use Chebyshev polynomial (0) or 
%          Jackson-Chebyshev polynomial (1) for graph filtering (default:
%          1)
%
% Output:
%   - group_coh, the square of the group graph coherence.
% 
% [1] G. Puy and P. Pérez, "Structured sampling and fast reconstruction of 
% smooth graph signals", Information and Inference: A Journal of the IMA, 
% 2018.
%
% Author: Gilles Puy
% *************************************************************************
%

% Compute maximum eigenvalue if not available
if ~isfield(G, 'lmax')
    G = gsp_estimate_lmax(G);
end

% Parameters
if nargin < 3
    param = struct;
end
if ~isfield(param, 'epsilon'),
    param.epsilon = 1e-2;
end
if ~isfield(param, 'verbose'),
    param.verbose = 0;
end
if ~isfield(param, 'jackson'),
    param.jackson = 1;
end
if ~isfield(param, 'max_iter'),
    param.max_iter = 500;
end

% Initialisation
group_coh = zeros(max(groups), 1);

% Prepare filter coefficients
[ch, jch] = gsp_jackson_cheby_coeff(0, lambda_k, [0 G.lmax], 50);

% Perform one estimation for each group
for id = 1:max(groups)
    
    % Check that the group is not empty
    indvector = (groups==id);
    if sum(indvector)==0
        continue;
    end
    
    % While flag is True, do an estimation
    flag = 1;
    while flag
        
        % Random signal living on the current group
        sig = randn(G.N, 1).*double(indvector);
        sig = sig/norm(sig);
        sig_old = sig;
        
        % Power method
        iter = 1;
        while true
            % Filtering
            if param.jackson
                X = gsp_cheby_op(G, jch(:), sig);
            else
                X = gsp_cheby_op(G, ch(:), sig);
            end
            
            % Normalise signal per group and estimate group coherence
            group_coh(id) = sig(indvector)'*X(indvector);
            group_coh(id) = group_coh(id)/(sig(indvector)'*sig(indvector));
            sig(indvector) = X(indvector)/norm(X(indvector));
            
            % Control evolution
            iter = iter + 1;
            err = norm(sig(indvector) - sig_old(indvector))/norm(sig_old(indvector));
            sig_old(indvector) = sig(indvector);
            if param.verbose,
                fprintf(' ... estimating group coherence: %i, %i iter., %e\n', ...
                    id, iter, err);
            end
            if err<param.epsilon
                flag = 0;
                break;
            elseif iter >= param.max_iter
                flag = 0;
                break;
            elseif iter>10 && err>1.9
                flag = 1; % Relaunch an estimation in this case
                break;
            end
        end
        
    end
    
    if param.verbose,
        fprintf('Done.\n');
    end
    
end
