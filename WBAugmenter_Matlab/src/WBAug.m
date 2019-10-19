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

function success = WBAug(input_dir, input_ext, output_dir, output_ext, ...
    groundTruth_dir, groundTruth_ext, groundTruth_output_dir,...
    useGPU, usePar, outNum, saveOrig)

% WBAug function: Generates outNum images with different white balance 
% settings and camera styles, where outNum<=10.
% Input:
%   -input_dir: directory of training images
%   -input_ext: file extension(s) of training images
%   -output_dir: new directory to save original and augmented training 
%   images 
%   -output_ext: file extension of generated images
%   -groundTruth_dir: directory of ground truth files
%   -groundTruth_ext: file extension(s) of ground truth files
%   -groundTruth_output_dir: new directory of ground truth files (after
%   augmentation)
%   -useGPU: true to use GPU
%   -usePar: true to use parallel computation 
%   -outNum: number of new images per each input image (should be <= 10)
%   -saveOrig: true to save copy of each input image in the output_dir

% Output:
%   -success: If success is 1, there was no error. If success = 0, the 
%   function coud not generate any image (there is a major error). If 
%   success < 0, the absolute value of success refers to the number of
%    images that could not be processed.


L = 10; % maximum number of images we can generate per training image
success = 0; % initialization
if useGPU == 1 % use GPU?
    try
        reset(gpuDevice); % reset GPU device
        SynthWB = load('synthWBmodel_GPU.mat'); % load GPU model
    catch
        if exist('synthWBmodel_GPU.mat','file') ~=0 % error in GPU device
            fprintf('Error: cannot access GPU device\n'); % error
            return
        else % could not find the model
            fprintf('Error: cannot find the model(s)\n');
            return
        end
    end
    
else
    SynthWB = load('synthWBmodel.mat'); % load CPU model
end

if outNum > L % if selected number of generated images > max number of allowed iamges
    fprintf('Error: Number of images should be <= %d\n',...
        L);
    return
end
files = []; % training images
for i = 1 : length(input_ext)
    ext = input_ext{i};
    temp = dir(fullfile(input_dir,['*' ext]));
    files = [files; {temp(:).name}'];
end

GTfiles = []; % ground truth files
for i = 1 : length(groundTruth_ext)
    ext = groundTruth_ext{i};
    temp = dir(fullfile(groundTruth_dir,['*' ext]));
    GTfiles = [GTfiles; {temp(:).name}'];
end

if length(files) ~= length(GTfiles) % if not equal, error!
    fprintf(['Error: Number of training files and ground truth files '...
        'must be the same\n']);
    return
end

if isempty(files) == 1 % if no files were found
    fprintf('Error: No training files are found\n');
    return
end

if exist(output_dir,'dir') == 0 % create out dir if not exist
    mkdir(output_dir);
end

if exist(groundTruth_output_dir,'dir') == 0 % same for ground truth
    mkdir(groundTruth_output_dir);
end

temp_names = cell(length(files),1);
GTnames = cell(length(files),1);
GT_ext = cell(length(files),1);
for i = 1 : length(files)
    [~,temp_names{i},~] = fileparts(files{i});
    [~,GTnames{i},GT_ext{i}] = fileparts(GTfiles{i});
end

[~,~,indices] = intersect(lower(temp_names),lower(GTnames));

fprintf('\n\nProcessing...\n\n');

if usePar == 1 % if use parallel computing 
    parfor i = 1 : length(files)
        I = im2double(imread(fullfile(input_dir,files{i}))); % read image
        try % apply our WB augmenter
        [out,pf] = SynthWB.WB_emulator.generate_wb_srgb(I,outNum);
        if useGPU==1 % if that was on GPU, convert to double tensor
            out = gather(out);
        end
        catch
            fprintf('Error in image %s!\n', fullfile(input_dir,files{i}));
            success = success - 1; % couldn't? decrease success
            continue;
        end
        for j =1 : size(out,4)
            imagename = files{i}; % save generated images and copy corresponding ground truth
            imwrite(out(:,:,:,j),fullfile(output_dir,sprintf('%s%s%s',...
                imagename(1:end-4),pf{j},output_ext)));
            copyfile(fullfile(groundTruth_dir,GTfiles{...
                indices(i)}),...
                fullfile(groundTruth_output_dir,[GTnames{indices(i)}...
                '_' pf{j} GT_ext{i}]));
            
            if saveOrig == true % save original image and corresponding ground truth
                imwrite(I,fullfile(output_dir,sprintf('%s%s%s',...
                    imagename(1:end-4),'_original',output_ext)));
                copyfile(fullfile(groundTruth_dir,GTfiles{...
                    indices(i)}),...
                    fullfile(groundTruth_output_dir,[GTnames{indices(i)}...
                    '_original' GT_ext{i}]));
            end
        end
    end
else % no parallel
    for i = 1 : length(files)
        I = im2double(imread(fullfile(input_dir,files{i}))); %read image
        try % apply our WB augmenter
            [out,pf] = SynthWB.WB_emulator.generate_wb_srgb(I,outNum);
            if useGPU==1 % if GPU is used, convert to double tensor
                out = gather(out);
            end
        catch
            fprintf('Error in image %s!\n', fullfile(input_dir,files{i}));
            success = success - 1; % coudln't? decrease success
            continue;
        end
        for j =1 : size(out,4) % save generated images and corresponding ground truth
            imagename = files{i};
            imwrite(out(:,:,:,j),fullfile(output_dir,sprintf('%s%s%s',...
                imagename(1:end-4),pf{j},output_ext)));
            copyfile(fullfile(groundTruth_dir,GTfiles{...
                indices(i)}),...
                fullfile(groundTruth_output_dir,[GTnames{indices(i)}...
                '_' pf{j} GT_ext{i}]));
            
            if saveOrig == true % save original image and corresponding ground truth
                imwrite(I,fullfile(output_dir,sprintf('%s%s%s',...
                    imagename(1:end-4),'_original',output_ext)));
                copyfile(fullfile(groundTruth_dir,GTfiles{...
                    indices(i)}),...
                    fullfile(groundTruth_output_dir,[GTnames{indices(i)}...
                    '_original' GT_ext{i}]));
            end
        end
    end
end
if success == 0 % if there wasn't any failure cases, return success = 1
    success = 1;
end
end