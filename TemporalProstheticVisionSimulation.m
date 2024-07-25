%
% The following simulation enables studying the temporal aspects of
% retinal prosthetic vision including electrode activation rate,
% persistence duration, and perceptual fading (onset and duration)
% with various electrode arrays (sizes and structures) and the two
% image polarities. The zip folder includes videos that could be used
% as an input to the simulation, but any video (grayscale or colored)
% could be used as an input. Please define all the folder paths below
% before running the simulation.
%
%
% Copyright(c) 2021 David Avraham, Jae-Hyun Jung, Yitzhak Yitzhaky and Eli Peli
% All Rights Reserved.
%
%----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% Please refer to the following paper
% Avraham, D., Jung, J. H., Yitzhaky, Y., & Peli, E. (2021). Retinal prosthetic vision simulation: temporal aspects. Journal of Neural Engineering, 18(4), 0460d7
%
% For questions, please address David Avraham in davidavr92@gmail.com.

function varargout = TemporalProstheticVisionSimulation(varargin)
tic
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @TemporalProstheticVisionSimulation_OpeningFcn, ...
    'gui_OutputFcn',  @TemporalProstheticVisionSimulation_OutputFcn, ...
    'gui_LayoutFcn',  [], ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


%% Set parameters in the GUI
function TemporalProstheticVisionSimulation_OpeningFcn(hObject, ~, handles, varargin)
set(handles.SigmaX,'String','.13'); % phosphene dimension in x axis
set(handles.SigmaY,'String','.13'); % phosphene dimension in y axis
set(handles.X0,'String','7'); % distance between two adjecent phosphenes in x axis
set(handles.Y0,'String','7');% distance between two adjecent phosphenes in y axis
set(handles.DFR,'String','60'); % display frame rate (usually 30 or 60 FPS)
set(handles.EAR,'String','10');% electrode activation rate (usually 6 or 10 Hz)
set(handles.PER,'String','2'); % persistence duration (usually between 0 and 8 seconds)
set(handles.PFO,'String','10'); % perceptual fading onset (usually between 0 and 10 seconds)
set(handles.PFD,'String','10'); % perceptual fading onset (usually between 0.5 and 60 seconds)
set(handles.ElectrodeArraySize,'val',3); % number of electrodes (60(val=1), 256(2), 961(3), or 2916(4) electrodes)
set(handles.ElectrodeArrayStructure,'val',1); % electrode array structure (square(val = 1) or hexagonal(2)).
% Currently, the hexagonal structure is pixelized when combined with temporal effects
set(handles.Polarity,'val',1); % image polarity (original(val=1) or reversed(2))
handles.output = hObject;
guidata(hObject, handles);

function UploadInputVideo_Callback(hObject,~, varargin)
chosenfile = uigetfile('*.*', 'Select a video'); % the input video must be in the Matlab current folder
flag = isfile(chosenfile);
if flag == 0
    msgbox('The input video must be in the Matlab current folder','Error')
else
    msgbox('An input video has been loaded successfully')
end
folderName = datestr(datetime(now, 'ConvertFrom','datenum'),'mmmm dd yyyy HH-MM-SS');
mkdir(folderName)
folderName = strcat(folderName,'\');
handles = guidata(hObject);
handles.folderName = folderName;
handles.chosenfile = chosenfile;
guidata(hObject, handles);


function varargout = TemporalProstheticVisionSimulation_OutputFcn(~, ~, handles)
varargout{1} = handles.output;


function CreatePhospheneImage_Callback(~, ~, handles)
folderName = handles.folderName; chosenfile = handles.chosenfile;
SigmaX = str2double(get(handles.SigmaX,'String'));SigmaY = str2double(get(handles.SigmaY,'String'));
X0 = str2double(get(handles.X0,'String'));Y0 = str2double(get(handles.Y0,'String'));
DFR = str2double(get(handles.DFR,'String')); EAR = str2double(get(handles.EAR,'String'));
PER = str2double(get(handles.PER,'String'));
PER_Factor = 3.9/(PER/(1/DFR));% 3.9 was set for 2 precent threshold,
% i.e. the phosphene falls to 2 precent of its original intensity;
PFO = str2double(get(handles.PFO,'String')); PF_Factor = PFO/(1/EAR);
PolarityVal = get(handles.Polarity,'val');
if PolarityVal == 1 %%%%% BH %%%%%
    va = 0; vb = 0.333; vc = 0.667; vd = 1;
elseif PolarityVal == 2 %%%%% BH %%%%%
    va = 1; vb = 0.667; vc = 0.333; vd = 0;
end

sizeval = get(handles.ElectrodeArraySize,'val'); % define number of electrodes
% MMr - number of rows; MMc - number of columns;
% R = number of pixel rows; C = number of pixel columns
f = waitbar(0,'Please wait...');
pause(.5)
if sizeval == 1 %%%%% 60 electrodes %%%%%
    MMr = 6; MMc = 10; C = 240; R = 144; % In the GUI, choose: phosphene size 0.4; X0 and Y0 = 24
elseif sizeval == 2 %%%%% 256 electrodes %%%%%
    MMr = 16; MMc = 16; C = 224; R = 224; % In the GUI, choose: phosphene size 0.22; X0 and Y0 = 14
elseif sizeval == 3 %%%%% 961 electrodes %%%%%
    MMr = 31; MMc = 31; C = 217; R = 217; % In the GUI, choose: phosphene size 0.13; X0 and Y0 = 7
elseif sizeval == 4 %%%%% 2916 electrodes %%%%%
    MMr = 54; MMc = 54; C = 216; R = 216; % In the GUI, choose: phosphene size 0.08; X0 and Y0 = 4
end

val = get(handles.ElectrodeArrayStructure,'val'); % define rectangular or hexagonal electrode array
v2 = 0;
if val == 1
    v3 = 0;
elseif val == 2
    v3 = 1;
end
count = 0;
x = 1:C;%6X24+2X12
y = 1:R;%10X24+2X12
[X,Y] = meshgrid(x,y);

% initialization
z = zeros(R,C); Frame = zeros(R,C); FormerFrame = zeros(R,C); PFC = zeros(MMr,MMc);
Theta = zeros(MMr,MMc);a = zeros(MMr,MMc); b = zeros(MMr,MMc); c = zeros(MMr,MMc);
sigma_x = zeros(MMr,MMc); sigma_y = zeros(MMr,MMc);

%% Spatial Models
for s = 1:MMr
    for t = 1:MMc
        Theta(s,t) = 0; sigma_x(s,t) = SigmaX; sigma_y(s,t) = SigmaY;
        a(s,t) = ((cosd(Theta(s,t))^2) / (2*sigma_x(s,t)^2)) + ((sind(Theta(s,t))^2) / (2*sigma_y(s,t)^2));
        b(s,t) = -((sind(2*Theta(s,t))) / (4*sigma_x(s,t)^2)) + ((sind(2*Theta(s,t))) / (4*sigma_y(s,t)^2));
        c(s,t) = ((sind(Theta(s,t))^2) / (2*sigma_x(s,t)^2)) + ((cosd(Theta(s,t))^2) / (2*sigma_y(s,t)^2));
    end
end
%%

V = VideoReader(chosenfile); % loading a video %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sample = round(V.FrameRate/EAR); % sampling the video according to the electrode activation rate
% I = imread('Lena.png'); % loading an image instead of a video - this command enforce other changes in the code.

waitbar(.5,f,'Processing your data');
pause(1)
for frame = 1:Sample:V.NumFrames % running on the sampled video frames
    I = read(V,frame);
    % Only for color
    %I = rgb2gray(I);
    % Only for non-binary image
    %I = imadjust(I,stretchlim(I),[]);
    I = double(I);
    % Normalization
    %I = I/max(max(I));
    %I = imresize(I,[R C]);
    MeanMatrix = zeros (MMr,MMc);
    s = 1; t = 1; v = 0;
    for m = floor(rem(size(I,1),MMr)/2)+floor(size(I,1)/MMr)*MMr...
            :-floor(size(I,1)/MMr):1+floor(rem(size(I,1),MMr)/2)
        for n = 1:floor(size(I,2)/MMc):floor(size(I,2)/MMc)*MMc
            for j = m:-1:m-floor(size(I,1)/MMr)+1
                for k = n:n+floor(size(I,2)/MMc)-1
                    v = v+I(j,k);
                end
            end
            v = v/(floor(size(I,1)/MMr)*floor(size(I,2)/MMc));
            if v >= 0 && v <= 0.25
                v = va;
            elseif v > 0.25 && v <= 0.5
                v = vb;
            elseif v > 0.5 && v <= 0.75
                v = vc;
            else
                v = vd;
            end
            MeanMatrix(s,t) = v;
            v = 0;
            t = t+1;
        end
        t = 1;
        s = s+1;
    end
    
    for SubFrame = 1:DFR/EAR
        for s = 1:MMr
            for t = 1:MMc
                z = z + MeanMatrix(s,t).*exp(...
                    -0.015 * (1/(3*MeanMatrix(s,t)+1))*...
                    (a(s,t)*(X-X0*t-v2+floor(X0/2)).^2+...
                    2*b(s,t)*(X-X0*t-v2+floor(X0/2)).*(Y-Y0*s+floor(Y0/2))+...
                    c(s,t)*(Y-Y0*s+floor(Y0/2)).^2));
                % Hexagonal: replace X0/2 with X0/4
            end
            v2 = v2 - (X0/2)*v3;
            v3 = v3*-1;
        end
        for s = R:-X0:1
            for t = 1:Y0:C
                if SubFrame == 1
                    if ( (mean(mean(z(s-X0+1:s,t:t+Y0-1))) == 0 && mean(mean(FormerFrame(s-X0+1:s,t:t+Y0-1))) ==0)...
                            || mean(mean(z(s-X0+1:s,t:t+Y0-1))) < mean(mean(FormerFrame(s-X0+1:s,t:t+Y0-1))) ...
                            || MeanMatrix(s/X0,floor(t/Y0)+1) == 0  )...
                            &&  PFC(s/X0,floor(t/Y0)+1) < PF_Factor%*(1+(2/5)*(rand-0.5))...
                        Frame(s-X0+1:s,t:t+Y0-1) = exp(-PER_Factor)*FormerFrame(s-X0+1:s,t:t+Y0-1);
                    elseif mean(mean(z(s-X0+1:s,t:t+Y0-1))) >= mean(mean(FormerFrame(s-X0+1:s,t:t+Y0-1)))...
                            && PFC(s/X0,floor(t/Y0)+1) < PF_Factor%*(1+(2/5)*(rand-0.5))
                        Frame(s-X0+1:s,t:t+Y0-1) = z(s-X0+1:s,t:t+Y0-1) ;
                        PFC(s/X0,floor(t/Y0)+1) = PFC(s/X0,floor(t/Y0)+1) + 1;
                    elseif PFC(s/X0,floor(t/Y0)+1) >= PF_Factor%*(1+(2/5)*(rand-0.5))
                        Frame(s-X0+1:s,t:t+Y0-1) = exp(-3*PER_Factor)*FormerFrame(s-X0+1:s,t:t+Y0-1);
                    end
                else
                    Frame(s-X0+1:s,t:t+Y0-1) = exp(-PER_Factor)*Frame(s-X0+1:s,t:t+Y0-1);
                end
            end
        end
        
        z = zeros(R,C);
        count = count + 1;
        if count>2*V.NumFrames
            break
        end
        FileName = fullfile(folderName,sprintf('%03d.png', count));
        imwrite(flipud(Frame), FileName);
        FormerFrame = Frame;
    end
end
%% Prosthetic vision video
%images = cell (round(V.NumFrames*(DFR/V.FrameRate)),1); % for video
% images = cell (1,1); % for image
video = VideoWriter('ProstheticVideo','MPEG-4');% create new video
video.FrameRate = DFR;
open(video);% open video
%cmap = gray(256);% determine color map
% srcFiles = dir('C:\Users\...................................\*.png');% define the folder path
srcFiles = dir(strcat(folderName,'*.png'));% define the folder path

for p =1:min(count,V.NumFrames*(DFR/V.FrameRate)) % for video
    % for p =1:1  % for image
    framename = strcat(folderName,srcFiles(p).name);% define the folder path
    img = imread(framename); % insert it to 'images'
    %frame = immovie(images{p}, cmap);% insert each frame to the video
    writeVideo(video, img);% insert each frame to the video
end
close(video);% close video
waitbar(1,f,'Finishing');
pause(1)
toc
close all;

function X0_Callback(~, ~, ~)

function Y0_Callback(~, ~, ~)

function SigmaX_Callback(~, ~, ~)

function SigmaY_Callback(~, ~, ~)

function DFR_Callback(~, ~, ~)

function EAR_Callback(~, ~, ~)

function PER_Callback(~, ~, ~)

function PFO_Callback(~, ~, ~)

function RecoveryTime_Callback(~, ~, ~)

function Tilt_Callback(~, ~, ~)

function Polarity_Callback(hObject, ~, ~)
contents = cellstr(get(hObject,'String'));
popChoise = contents{get(hObject,'Value')};
if(strcmp(popChoise,'WH'))
elseif(strcmp(popChoise,'BH'))
end

function ElectrodeArrayStructure_Callback(hObject, ~, ~)
contents = cellstr(get(hObject,'String'));
popChoise = contents{get(hObject,'Value')};
if(strcmp(popChoise,'Square'))
elseif(strcmp(popChoise,'Hexagonal'))
end

function ElectrodeArraySize_Callback(hObject, ~, ~)
contents = cellstr(get(hObject,'String'));
popChoise = contents{get(hObject,'Value')};
if(strcmp(popChoise,'60'))
elseif(strcmp(popChoise,'324'))
elseif(strcmp(popChoise,'961'))
elseif(strcmp(popChoise,'2916'))
end

function PFD_Callback(~, ~, ~)



function PFD_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function EAR_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PER_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PFO_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function DFR_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Polarity_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function SigmaX_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function SigmaY_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function X0_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Y0_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ElectrodeArraySize_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ElectrodeArrayStructure_CreateFcn(hObject, ~, ~)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end