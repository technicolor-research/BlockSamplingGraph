function G = prepare_bunny()
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
% This function load the minnesota graph available in the GSP toolbox 
% and assign a group to each node. 
%
% Result is returned in a structure array (compatible with the GSP 
% toolbox). The group label assigned to each node appears in G.groups
%
% Author: Gilles Puy
%
% *************************************************************************
%

% Graph
G = gsp_bunny;

% Create grid to search spatially local neighbors
nb = 8;
xgrid = linspace(min(G.coords(:, 1)), max(G.coords(:, 1)), nb+1);
xgrid = xgrid(1:nb) + (xgrid(2)-xgrid(1))/2;
ygrid = linspace(min(G.coords(:, 2)), max(G.coords(:, 2)), nb+1);
ygrid = ygrid(1:nb) + (ygrid(2)-ygrid(1))/2;
zgrid = linspace(min(G.coords(:, 3)), max(G.coords(:, 3)), nb+1);
zgrid = zgrid(1:nb) + (zgrid(2)-zgrid(1))/2;
[xgrid, ygrid, zgrid] = meshgrid(xgrid, ygrid, zgrid);

% Assign group to each node
groups = ones(G.N, 1);
for node = 1:G.N
    dist = abs(xgrid-G.coords(node, 1)).^2 + ...
        abs(ygrid-G.coords(node, 2)).^2 + ...
        abs(zgrid-G.coords(node, 3)).^2;
    groups(node) = find(dist==min(dist(:)));
end

% Remove empty groups
[C, ~, ic] = unique(groups);
temp = transpose((1:length(C)));
G.groups = temp(ic);

end
