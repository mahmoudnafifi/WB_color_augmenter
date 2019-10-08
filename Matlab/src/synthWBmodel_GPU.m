%%
% Copyright (c) 2019-present, Mahmoud Afifi
% All rights reserved.
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
    properties
        features %training features
        mappingFuncs9 %training mapping functions 9x3 poly
        K %K value for KNN
        encoder %PCA object
        wb_photo_finishing %modes
    end
    methods
        
        function feature = encode(obj,hist) %encode RGB-histogram feature
            feature =  obj.encoder.encode(hist);
        end
        
        function hist = RGB_UVhist(obj,I)  %generate RGB-histogram
            I = im2double(I);
            if size(I,1)*size(I,2) > 202500 %(450*450)
                factor = sqrt(202500/(size(I,1)*size(I,2)));
                newH = floor(size(I,1)*factor);
                newW = floor(size(I,2)*factor);
                I = imresize(I,[newH,newW]);
            end
            
            h= sqrt(max(size(obj.encoder.weights,1),...
                size(obj.encoder.weights,2))/3);
            eps= 6.4/h;
            I=(reshape(I,[],3));
            A=gpuArray([-3.2:eps:3.19]);
            hist=gpuArray(zeros(size(A,2),size(A,2),3));
            i_ind=I(:,1)~=0 & I(:,2)~=0 & I(:,3)~=0;
            I=I(i_ind,:);
            Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2);
            for i = 1 : 3
                r = setdiff([1,2,3],i);
                Iu=log(abs((I(:,i))./(I(:,r(1)))));
                Iv=log(abs((I(:,i))./(I(:,r(2)))));
                diff_u=abs(Iu-A);
                diff_v=abs(Iv-A);
                diff_u=(reshape((reshape(diff_u,[],1)<=eps/2),[],size(A,2)));
                diff_v=(reshape((reshape(diff_v,[],1)<=eps/2),[],size(A,2)));
                hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v);
                hist(:,:,i)=sqrt(hist(:,:,i)/sum(sum(hist(:,:,i))));
            end
            hist = imresize(hist,[h h]);
        end
        
        function [synthWBimages,wb_pf] = generate_wb_srgb (obj,I, ...
                outNum, feature, sigma) %WB emulation
            I = im2double(I);
            if nargin == 2
                outNum = length(obj.wb_photo_finishing);
                feature = obj.encode(obj.RGB_UVhist(I));
                sigma = 0.25;
            elseif nargin == 3
                feature = obj.encode(obj.RGB_UVhist(I));
                sigma = 0.25;
            elseif nargin == 4
                sigma = 0.25;
            end
            
            if outNum > length(obj.wb_photo_finishing)
                error('Error: number of new images should be <= %d',...
                    length(obj.wb_photo_finishing));
            end
            
            if outNum ~= length(obj.wb_photo_finishing)
                inds = randperm(length(obj.wb_photo_finishing));
                wb_pf = obj.wb_photo_finishing(...
                    inds(1:outNum));
            else
                wb_pf = obj.wb_photo_finishing;
                inds = [1:length(wb_pf)];
            end
            synthWBimages = gpuArray(zeros(size(I,1),size(I,2),size(I,3),...
                length(wb_pf)));
            
            [dH,idH] = pdist2(obj.features,feature,...
                'euclidean','Smallest',obj.K);
            weightsH = exp(-((dH).^2)/(2*sigma^2));
            weightsH = weightsH/sum(weightsH);
            count = 1;
            for i = inds(1:outNum)
                mf = sum(weightsH .* obj.mappingFuncs9((idH-1)*10 + i,:),1);
                mf = reshape(mf,[9,3]);
                [synthWBimages(:,:,:,count),~] = obj.change_wb(I,mf,9,1);
                count = count + 1;
            end
            
        end
        
        function [out,map] = change_wb(obj,input, m, f, clipping)
            if nargin == 3
                clipping = 0 ;
                map = [];
            end
            sz=size(input);
            input=reshape(input,[],3);
            %higher degree (N-D)
            switch f
                case 3
                    input=obj.kernel3(input);
                case 9
                    input=obj.kernelP9(input);
                case 11
                    input=obj.kernelP11(input);
            end
            
            out=input*m;
            out=reshape(out,[sz(1),sz(2),sz(3)]);
            if clipping == 1
                [out,map] = obj.out_of_gamut_clipping(out);
            end
        end
        
        function O=kernel3(obj,I)
            %kernel(r,g,b)=[r,g,b];
            O=I;
        end
        
        function O=kernelP9(obj,I)
            %kernel(r,g,b)=[r,g,b,r2,g2,b2,rg,rb,gb];
            O=[I,... %r,g,b
                I.*I,... %r2,g2,b2
                I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
                ];
        end
        
        
        function O=kernelP11(obj,I)
            %kernel(R,G,B)=[R,G,B,RG,RB,GB,R2,G2,B2,RGB,1];
            O=[I,... %r,g,b
                I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
                I.*I,... %r2,g2,b2
                I(:,1).*I(:,2).*I(:,3),... %rgb
                ones(size(I,1),1)]; %1
        end
        
        
        function [I,map] = out_of_gamut_clipping(obj,I)
            sz = size(I);
            I = reshape(I,[],3);
            map = ones(size(I,1),1);
            map(I(:,1)>1 | I(:,2)>1 | I(:,3)>1 | I(:,1)<0 | I(:,2)<0 | I(:,3)<0)=0;
            map = reshape(map,[sz(1),sz(2)]);
            I(I>1)=1;
            I(I<0)=0;
            I = reshape(I,[sz(1),sz(2),sz(3)]);
        end
        
    end
end
