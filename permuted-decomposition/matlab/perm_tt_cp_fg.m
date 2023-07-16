%%% DISCLAIMER %%%
% This file is a modified version of `tt_cp_fg.m` from Tensor Toolbox:
% Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.1,
% www.tensortoolbox.org, June 2019. https://gitlab.com/tensors/tensor_toolbox.
% It is therefore subject to the following license:

% BSD 2-Clause License

% Copyright (c) 2018, Sandia National Labs
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:

% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



function [f,G] = perm_tt_cp_fg(Z,A,Znormsqr,NDIRS)
%PERM_TT_CP_FG Computes function and gradient of the CP function.
%
%   [F,G] = PERM_TT_CP_FG(Z,A) calculates F = (1/2) ||Z - ktensor(A)||^2 where
%   Z is an N-way tensor and A is a ktensor or a cell array with N
%   factor matrices. It also calculates the gradient of the CP fit
%   function where Z is an N-way tensor and A is a ktensor or a
%   cell array with N factor matrices. The result is also a cell
%   array with N factor matrices corresponding to the gradients; in
%   other words, G{n}(:,r) is the partial derivative of the fit
%   function with respect to A{n}(:,r). 
%
%   [F,G] = PERM_TT_CP_FG(Z,A,NORMZSQR) also passes in the pre-computed
%   norm of Z, which makes the computations faster. 
%
%   [F,G] = PERM_TT_CP_FG(Z,A,NORMZSQR,NDIRS) updates the current components
%   based on the circular-shifted version of each response map in Z that
%   gives the min cost. NDIRS is the number of rows of the original 2-D response.
%
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.


%% Set-up
% if ~isa(Z,'tensor') && ~isa(Z,'sptensor')
%     error('Z must be a tensor or a sptensor');
% end
N = ndims(Z);

if ~iscell(A) && ~isa(A,'ktensor');
    error('A must be a cell array or ktensor');
end

if isa(A,'ktensor')
    A = tocell(A);
end
R = size(A{1},2);




%% Calculation

%F1
if exist('Znormsqr','var')
    f_1 = Znormsqr;
else
    f_1 = norm(Z)^2;
end

%% Upsilon and Gamma
Upsilon = cell(N,1);
for n = 1:N
    Upsilon{n} = A{n}'*A{n};%'
end

Gamma = cell(N,1);
for n = 1:N
    Gamma{n} = ones(R,R);
    for m = [1:n-1,n+1:N]
        Gamma{n} = Gamma{n} .* Upsilon{m};
    end
end

%F3
W = Gamma{1} .* Upsilon{1};
f_3 = sum(W(:));

%% NEW - find argmin_Z ||Z - A|| by circular-shifting slices of Z

ktA = ktensor(A);
doubleA = double(ktA);

%circshift for each stimulus separately
Ncells = size(Z,1);
NSTIMS = size(Z,2);
PSTHLEN = size(Z,3);

shape4d = [Ncells,NSTIMS,NDIRS,PSTHLEN/NDIRS];
shapeDot = [Ncells,PSTHLEN];
tensor4d = reshape(Z.data,shape4d);

stim_shifts = zeros(NSTIMS,Ncells);

for si = 1:NSTIMS
    objs = inf(Ncells,NDIRS);
    stimA = squeeze(doubleA(:,si,:));
    stimZ = squeeze(tensor4d(:,si,:,:));
    for shifti = 1:NDIRS %suffices to compute second summand
        objs(:,shifti) = -sum(stimA .* reshape(circshift(stimZ,shifti,2),shapeDot), 2);
    end
    [~,shifts] = min(objs,[],2);
    stim_shifts(si,:) = shifts;
end

argminZ = zeros(size(tensor4d));

for xi = 1:Ncells
    for si = 1:NSTIMS
        %shift PSTH
        argminZ(xi,si,:,:) = circshift(tensor4d(xi,si,:,:),stim_shifts(si,xi),3);
        
    end
end
Z = tensor(reshape(argminZ,size(Z)));


%% Calculate gradient and F2
G = cell(N,1);
U = mttkrp(Z,A,1);
V = A{1} .* U;
f_2 = sum(V(:));
G{1} = -U + A{1}*Gamma{1};
for n = 2:N
    U = mttkrp(Z,A,n);
    G{n} = -U + A{n}*Gamma{n};
end



%SUM
f = 0.5 * f_1 - f_2 + 0.5 * f_3;
%compare with standard formula
%f_ = .5 * norm(full(Z) - full(ktensor(A)))^2
%f_ = .5*(f_1 + f_3) - innerprod(ktensor(A),Z)





