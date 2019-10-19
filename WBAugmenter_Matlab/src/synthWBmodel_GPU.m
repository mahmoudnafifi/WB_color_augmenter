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

classdef synthWBmodel_GPU
% WB emulator GPU model
    properties
        features % training features
        mappingFuncs9 % training mapping functions 9x3 poly
        K % K value for KNN
        encoder % PCA object
        wb_photo_finishing % WB & photo finishing (PF) styles
    end
    methods
        
        function feature = encode(obj,hist)
		% Generates a compacted feature of a given histogram tensor.
            feature =  obj.encoder.encode(hist);
        end
        
        function hist = RGB_UVhist(obj,I)
		% Generates an RGB-histogram tensor of a given image I.
            I = im2double(I);
            if size(I,1)*size(I,2) > 202500 % if input image > (450*450),
                factor = sqrt(202500/(size(I,1)*size(I,2))); % scale factor
                newH = floor(size(I,1)*factor); % new dimensions
                newW = floor(size(I,2)*factor);
                I = imresize(I,[newH,newW]); % scale it down
            end
            
            h= sqrt(max(size(obj.encoder.weights,1),...
                size(obj.encoder.weights,2))/3); % histogram dimension
            eps= 6.4/h; % threshold 
            I=(reshape(I,[],3)); % reshape input image
            A=gpuArray([-3.2:eps:3.19]); % dummy vector
            hist=gpuArray(zeros(size(A,2),size(A,2),3)); % histogram tensor will be stored here
            i_ind=I(:,1)~=0 & I(:,2)~=0 & I(:,3)~=0; % remove zero pixels
            I=I(i_ind,:);
            Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2); % intensity vector
            for i = 1 : 3 % for each layer in the histogram
                r = setdiff([1,2,3],i); % extract the current color channel
                Iu=log(abs((I(:,i))./(I(:,r(1))))); % Iu vector
                Iv=log(abs((I(:,i))./(I(:,r(2))))); % Iv vector
                diff_u=abs(Iu-A); % differences in u space
                diff_v=abs(Iv-A); % differences in v space
                diff_u=(reshape((reshape(diff_u,[],1)<=eps/2),...
                    [],size(A,2))); % set 1's for all pixels below the threshold
                diff_v=(reshape((reshape(diff_v,[],1)<=eps/2),...
                    [],size(A,2))); % the same in the v space
                hist(:,:,i)=... % hist = Iy .* diff_u' * diff_v (.* element-wise mult)
                    (Iy.*double(diff_u))'*double(diff_v);
                hist(:,:,i)=... %final hist is sqrt(hist/sum(hist))
                    sqrt(hist(:,:,i)/sum(sum(hist(:,:,i))));
            end
            hist = imresize(hist,[h h]);
        end
        
        function [synthWBimages,wb_pf] = generate_wb_srgb (obj,I, ...
                outNum, feature, sigma)
		% Generates outNum new images from a given image I, where outNum should be <=10
            I = im2double(I); % convert to double
            if nargin == 2
                outNum = length(obj.wb_photo_finishing); % use all WB & PF styles (default)
                feature = obj.encode(obj.RGB_UVhist(I)); % encode histogram of I
                sigma = 0.25; % fall-off factor
            elseif nargin == 3
                feature = obj.encode(obj.RGB_UVhist(I));
                sigma = 0.25;
            elseif nargin == 4
                sigma = 0.25;
            end
            
            if outNum > length(obj.wb_photo_finishing) % if selected styles > the available WB & PF styles
                error('Error: number of new images should be <= %d',...
                    length(obj.wb_photo_finishing));
            end
            
            if outNum ~= length(obj.wb_photo_finishing) % if selected styles < the available WB & PF styles
                inds = randperm(length(obj.wb_photo_finishing)); % randomize from the available WB & PF styles
                wb_pf = obj.wb_photo_finishing(...
                    inds(1:outNum));
            else
                wb_pf = obj.wb_photo_finishing; % otherwise, use all available WB & PF styles
                inds = [1:length(wb_pf)];
            end
            synthWBimages = gpuArray(zeros(size(I,1),size(I,2),size(I,3),...
                length(wb_pf))); % new images will be stored here as a GPU array
            
            [dH,idH] = pdist2(obj.features,feature,...
                'euclidean','Smallest',obj.K); % K nearest neighbor
            weightsH = exp(-((dH).^2)/(2*sigma^2)); % compute blending weights
            weightsH = weightsH/sum(weightsH); % normalize weights
            count = 1;
            for i = inds(1:outNum) % for each WB & PF style, do
                mf = sum(weightsH .* ... % compute mapping funciton
                    obj.mappingFuncs9((idH-1)*10 + i,:),1);
                mf = reshape(mf,[9,3]); % reshape it to be 9*3
                synthWBimages(:,:,:,count) = obj.change_wb(I,mf); % apply mf
                count = count + 1;
            end
            
        end
        
        function out = change_wb(obj,input, m)
		% Applies a given mapping function m to input image input.
            sz=size(input);
            input=reshape(input,[],3); % reshape image to be n*3 (n total number of pixels)
            %higher degree (N-D)
            input=obj.kernelP9(input); % raise it to a higher-dim space   
            out=input*m; % apply m
            out = obj.out_of_gamut_clipping(out); % clip out-of-gamut pixels
            out=reshape(out,[sz(1),sz(2),sz(3)]); % reshape the image back to its original shape
        end
        
        function O=kernelP9(obj,I)
		% Kernel function: \phi(r,g,b) -> (r,g,b,r2,g2,b2,rg,rb,gb)
            O=[I,... %r,g,b
                I.*I,... %r2,g2,b2
                I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
                ];
        end
        
        
        function I = out_of_gamut_clipping(obj,I)
		% Clips out-of-gamut pixels.
            I(I>1)=1;
            I(I<0)=0;
        end
        
    end
end
