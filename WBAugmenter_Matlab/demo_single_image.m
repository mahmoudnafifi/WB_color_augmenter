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

%% 
% This demo generates ten images from the given image. Each generated file
% will be in the following format: originalName_WB_CS.ext, where
% originalName is the original filename, WB refers to white balance (WB)
% settings, and CS is the camera style. 
% WB settings are: 
%   - T: Tungsten WB [2850 Kelvin (K)]
%   - F: Fluorescent WB [3800K]
%   - D: Daylight WB [5500K] 
%   - C: Cloudy WB [6500K]
%   - S: Shade WB [7500K]
% Camera styes are:
%   - AS: Adobe Standard
%   - CS: Camera Standard
% The generated images and a copy of the input image (optional) will be 
% saved in the output directory (output_dir)

%%
clear; 
clc

imagename = fullfile('..','images','image1.jpg'); % image filename
output_dir = fullfile('..','results'); % output directory to save the 
% generated images and a copy of input image
useGPU = false; %to use GPU
NumOfImgs = 10; % should be less than or equal 10
saveOrig = true; %to save a copy of the original image in output_dir

if NumOfImgs > 10
    error('Cannot generate more than 10 images for each input image');
end

if exist(output_dir,'dir') == 0
    mkdir(output_dir);
end

if useGPU
    load('synthWBmodel_GPU.mat'); % load WB_emulator GPU model
else
    load('synthWBmodel.mat'); % load WB_emulator CPU model
end

I_in = imread(imagename); % read the image
[~,name,ext] = fileparts(imagename);
if saveOrig == true % save a copy of the original image
    imwrite(I_in,fullfile(output_dir,sprintf('%s%s%s',name,'_original',ext)));
end

%%
disp('processing...'); 
tic
% generate images with synthetic WB effects
out = WB_emulator.generate_wb_srgb(I_in, NumOfImgs); 
toc
if useGPU
    out = gather(out); % if GPU is used, convert images to a double tensor
end
disp('done!'); 
disp('saving...'); 
for i =1 : size(out,4)
    % save generated images
    imwrite(out(:,:,:,i),fullfile(output_dir,sprintf('%s%s%s',...
        name,WB_emulator.wb_photo_finishing{i},ext)));
end
