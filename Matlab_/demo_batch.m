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
% This demo generates ten images from each image in the  given path. 
% Each generated file will be in the following format: 
% originalName_WB_CS.ext, where originalName is the original filename, WB 
% refers to white balance (WB) settings, and CS is the camera style. 
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
clc;

use_GPU = 0; % to use GPU
datasetbase = fullfile('..','images'); % path of images directory
output_dir = fullfile('..','results'); % output directory (will contain 
% generated images and copy of original images
NumOfImgs = 10; % should be less than or equal 10
saveOrig = true; % to save a copy of original images in output_dir

if NumOfImgs > 10
    error('Cannot generate more than 10 images for each input image');
end

imds = imageDatastore(datasetbase,'IncludeSubfolders' ,1);
images = {imds.Files{:}}'; % get all input filenames

if use_GPU==1
    load('synthWBmodel_GPU.mat'); % load WB_emulator GPU model
else
    load('synthWBmodel.mat'); % load WB_emulator CPU model
end

for i = 1 : length (images) % for each input image, do
    fprintf('processing image: %s...\n',images{i});
    imgin = images{i};
    I_in = imread(imgin); % read input image
    [~,name,ext] = fileparts(imgin);
    if saveOrig == true
        imwrite(I_in,fullfile(output_dir,... % save a copy of it in output dir
            sprintf('%s%s.%s',name,'_original',ext)));
    end
    try
        % generate images with synthetic WB effects
        out = WB_emulator.generate_wb_srgb(I_in, NumOfImgs); 
        if use_GPU==1 % if GPU is used, 
            out = gather(out); % convert gpuArray to double tensor
        end
        for j =1 : size(out,4) % save generated images
            imwrite(out(:,:,:,j),fullfile(output_dir,...
                sprintf('%s%s%s',name,...
                WB_emulator.wb_photo_finishing{j},ext)));
        end
    catch
        fprintf('Error in image %s!\n', imgin);
    end
end
