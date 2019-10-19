%%
% Copyright (c) 2019-present, Mahmoud Afifi
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
% 
% Please, cite the following paper if you use this code:
%
% Mahmoud Afifi and Michael S. Brown. What else can fool deep learning? 
% Addressing color constancy errors on deep neural network performance.
% ICCV, 2019
%
% Email: mafifi@eecs.yorku.ca | m.3afifi@gmail.com
%%

classdef PCAFeature
    properties
        weights
        bias
    end
    methods
        function feature = encode(obj,hist)
		% Generates a compacted feature of a given histogram tensor.
            feature = (reshape(hist,1,[]) - obj.bias') *obj.weights;
        end
    end
end