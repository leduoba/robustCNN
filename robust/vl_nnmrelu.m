function Y = vl_nnmrelu(X,dzdY)
%VL_NNSOFTMAX CNN softmax.
%   Y = VL_NNSOFTMAX(X) applies the softmax operator the data X. X
%   has dimension H x W x D x N, packing N arrays of W x H
%   D-dimensional vectors.
%
%   D can be thought of as the number of possible classes and the
%   function computes the softmax along the D dimension. Often W=H=1,
%   but this is not a requirement, as the operator is applied
%   convolutionally at all spatial locations.
%
%   DZDX = VL_NNSOFTMAX(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
Y = max(1 - abs(X), 0);

if nargin < 2, return ; end

% backward
Y = zeros(size(X), 'single');
Y = Y + single(X > -1 & X < 0) - single(X > 0 & X < 1);
Y = Y .* dzdY;
