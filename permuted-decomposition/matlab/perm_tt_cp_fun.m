%%% DISCLAIMER %%%
% This file is a modified version of `tt_cp_fun.m` from Tensor Toolbox:
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



function [f,g,shifts] = perm_tt_cp_fun(x,Z,Znormsqr,doshift)
%TT_CP_FUN Calculate function and gradient for CP fit function.
%
%  [F,G] = TT_CP_FUN(X,Z) where X is a vector containing the entries of the
%  components of the model and Z is the tensor to be fit.
%
%  See also TT_CP_VEC_TO_FAC, TT_FAC_TO_VEC, TT_CP_FG, CP_OPT
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.
if nargin < 4
    doshift = 1;
end
    

%% Convert x to a cell array of matrices
A = tt_cp_vec_to_fac(x,Z);

%% Call cp_fit and cp_gradient using cp_fg
if doshift > 1
    [f,G] = perm_tt_cp_fg(Z,A,Znormsqr,doshift);
else
    [f,G] = tt_cp_fg(Z,A,Znormsqr);
end

%% Convert a cell array to a vector
g = tt_fac_to_vec(G);


