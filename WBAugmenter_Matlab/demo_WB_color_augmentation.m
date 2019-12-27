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
% This demo shows how to use our data augmenter to generate extra X images
% with different WB and camera styles for data augmentation. You can
% generate up to ten images for each of training images (i.e., X<=10)
% The code will automatically generate copies of the corresponding ground
% truth files for each original image. The copied ground truth files will
% be saved with the same name of generated images (of course without 
% changing the original file extension of ground truth data).
% Each of the generated images will be saved in the following format: 
% originalName_WB_CS.ext, where originalName is the original filename, 
% WB refers to white balance (WB) settings, and CS is the camera style. 
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

clc
clear 
close all
warning off

training_set_dir = fullfile('..','example','training_set'); % training data path

ground_truth_dir = fullfile('..','example','ground_truth'); % ground truth path

training_ext = {'.jpg','.png'}; % file extension of training images (you can add more)

ground_truth_ext = {'.png'};  % file extension of ground truth data 
% be sure that your ground truth data is saved in the same names of 
% corresponding training images

output_ext = '.jpeg'; % output extension of our generated images

output_dir = fullfile('..','new_training_set'); % directory to save 
% generated images and copy of original training images

useGPU = 0; % to use GPU

usePar = 0; % to use parallel computing (requires Matlab parallel computing toolbox)

outNum = 2; % number of generated images for each original training image. 
%If less than 10, it will randomize between the allowed 10 versions

% directory to save ground truth files for the augmented training images
ground_truth_new_dir = fullfile('..','new_ground_truth'); 

saveOrig = true; % to save a copy of the original images in output_dir

% start color augmentation
disp('Starting color augmentation...');
success = WBAug(training_set_dir, training_ext, output_dir, output_ext, ...
    ground_truth_dir, ground_truth_ext, ground_truth_new_dir,...
    useGPU, usePar, outNum, saveOrig);
disp('Done!');
if success < 0 
    fprintf('There are %d files could not be processed!\n',abs(success));
elseif success == 0
    disp('Sorry, we cannot process any image!');
end