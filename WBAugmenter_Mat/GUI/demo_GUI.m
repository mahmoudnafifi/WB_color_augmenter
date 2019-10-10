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

function varargout = demo_GUI(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @demo_GUI_OpeningFcn, ...
    'gui_OutputFcn',  @demo_GUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end



function demo_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
warning off
addpath('classes');

global WB_emulator
load('synthWBmodel.mat');
handles.save.Enable = 'off';


function varargout = demo_GUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on button press in browse.
function browse_Callback(hObject, eventdata, handles)
% hObject    handle to browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global Path_Name
global File_Name
%global I_temp
global I
global outImages
global WB_emulator
global feature
Old_fileName = File_Name;
Old_pathName = Path_Name;
[File_Name, Path_Name] = uigetfile({'*.jpg';'*.png';'*.jpeg'},'Select input image',fullfile('..','..','images'));

if File_Name == 0
    File_Name = Old_fileName;
    Path_Name = Old_pathName;
    if sum(I(:)) ~= 0
        handles.save.Enable = 'on';
    else
        handles.save.Enable = 'off';
    end
else
    I = imread(fullfile(Path_Name,File_Name));
    axes(handles.image);
    handles.image.Visible = 'On';
    handles.input_text.Visible = 'On';
    imshow(I);
    pause(0.001);
    handles.status.String = 'Processing...';pause(0.001);
    k = handles.k.Value;
    sigma = handles.sigma.Value;
    WB_emulator.K = round(k);
    
%     if handles.wb_checkbox.Value==1
%         WBmodel = load('WBmodel.mat');
%         I_temp = WBmodel.WBmodel.correctImage(I);
%     else
%         I_temp = I ;
%     end
    hist = WB_emulator.RGB_UVhist(im2double(I));
    feature = WB_emulator.encode(hist);
    outImages = WB_emulator.generate_wb_srgb(I,10,feature,sigma);
    for i = 1 : size(outImages,4)
        tempstr= WB_emulator.wb_photo_finishing{i};
        eval(sprintf('axes(handles.%s);',tempstr(2:end)));
        eval(sprintf('handles.%s.Visible=''On'';',tempstr(2:end)));
        eval(sprintf('handles.%s_text.Visible=''On'';',tempstr(2:end)));
        imshow(outImages(:,:,:,i));
    end
    handles.save.Enable = 'on';
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Path_Name
global File_Name
global outImages
global WB_emulator
global I
[~,name,ext] = fileparts(File_Name);
outFile_Name = [name 'SynthWB' ext];
[file,path,~] = uiputfile({'*.jpg';'*.png'},'Save Image',fullfile(Path_Name,outFile_Name));
if file ~=0
    [~,name,ext] = fileparts(file);
    handles.status.String = 'Processing...';pause(0.001);
    imwrite(I,fullfile(path,sprintf('%s%s%s',name,'_original',ext)));
    for j =1 : size(outImages,4)
        imwrite(outImages(:,:,:,j),fullfile(path,...
            sprintf('%s%s%s',name,...
            WB_emulator.wb_photo_finishing{j},ext)));
    end
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end

% --- Executes on slider movement.
function k_Callback(hObject, eventdata, handles)
% hObject    handle to k (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global WB_emulator
global I
global feature
global outImages
axes(handles.image);
handles.status.String = 'Processing...';pause(0.001);
k = handles.k.Value;
sigma = handles.sigma.Value;
WB_emulator.K = round(k);
outImages = WB_emulator.generate_wb_srgb(I,10,feature,sigma);
for i = 1 : size(outImages,4)
    tempstr= WB_emulator.wb_photo_finishing{i};
    eval(sprintf('axes(handles.%s);',tempstr(2:end)));
    eval(sprintf('handles.%s.Visible=''On'';',tempstr(2:end)));
    eval(sprintf('handles.%s_text.Visible=''On'';',tempstr(2:end)));
    imshow(outImages(:,:,:,i));
end
handles.status.String = 'Done!';
pause(0.01); handles.status.String = '';



% --- Executes during object creation, after setting all properties.
function k_CreateFcn(hObject, eventdata, handles)
% hObject    handle to k (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function sigma_Callback(hObject, eventdata, handles)
% hObject    handle to sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global WB_emulator
global I
global feature
global outImages
axes(handles.image);
handles.status.String = 'Processing...';pause(0.001);
k = handles.k.Value;
sigma = handles.sigma.Value;
WB_emulator.K = round(k);
outImages = WB_emulator.generate_wb_srgb(I,10,feature,sigma);
for i = 1 : size(outImages,4)
    tempstr= WB_emulator.wb_photo_finishing{i};
    eval(sprintf('axes(handles.%s);',tempstr(2:end)));
    eval(sprintf('handles.%s.Visible=''On'';',tempstr(2:end)));
    eval(sprintf('handles.%s_text.Visible=''On'';',tempstr(2:end)));
    imshow(outImages(:,:,:,i));
end
handles.status.String = 'Done!';
pause(0.01); handles.status.String = '';



% --- Executes during object creation, after setting all properties.
function sigma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% 
% % --- Executes on button press in wb_checkbox.
% function wb_checkbox_Callback(hObject, eventdata, handles)
% % hObject    handle to wb_checkbox (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hint: get(hObject,'Value') returns toggle state of wb_checkbox
% 
% %global I_temp
% global outImages
% global WB_emulator
% global feature
% global I
% pause(0.001);
% handles.status.String = 'Processing...';pause(0.001);
% k = handles.k.Value;
% sigma = handles.sigma.Value;
% WB_emulator.K = round(k);
% % 
% % if handles.wb_checkbox.Value==1
% %     WBmodel = load('WBmodel.mat');
% %         I_temp = WBmodel.WBmodel.correctImage(I);
% % else
% %     I_temp = I ;
% % end
% hist = WB_emulator.RGB_UVhist(im2double(I));
% feature = WB_emulator.encode(hist);
% outImages = WB_emulator.generate_wb_srgb(I,10,feature,sigma);
% for i = 1 : size(outImages,4)
%     tempstr= WB_emulator.wb_photo_finishing{i};
%     eval(sprintf('axes(handles.%s);',tempstr(2:end)));
%     eval(sprintf('handles.%s.Visible=''On'';',tempstr(2:end)));
%     eval(sprintf('handles.%s_text.Visible=''On'';',tempstr(2:end)));
%     imshow(outImages(:,:,:,i));
% end
% handles.status.String = 'Done!';
% pause(0.01); handles.status.String = '';
