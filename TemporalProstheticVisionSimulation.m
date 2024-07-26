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
chosenfile = uigetfile('*.*', 'Select a video');% the input video must be in the Matlab current folder
handles.chosenfile = chosenfile;
guidata(hObject, handles);


function varargout = TemporalProstheticVisionSimulation_OutputFcn(~, ~, handles)
varargout{1} = handles.output;


function CreatePhospheneImage_Callback(~, ~, handles)
chosenfile = handles.chosenfile;
SigmaX = str2double(get(handles.SigmaX,'String'));SigmaY = str2double(get(handles.SigmaY,'String'));
X0 = str2double(get(handles.X0,'String'));Y0 = str2double(get(handles.Y0,'String'));
DFR = str2double(get(handles.DFR,'String')); EAR = str2double(get(handles.EAR,'String'));
PER = str2double(get(handles.PER,'String'));

PFO = str2double(get(handles.PFO,'String'));
PolarityVal = get(handles.Polarity,'val');
sizeval = get(handles.ElectrodeArraySize,'val');
val = get(handles.ElectrodeArrayStructure,'val');

main_function(chosenfile,SigmaX,SigmaY,X0,Y0,DFR,EAR,PER,PFO, PFD,PolarityVal,sizeval,val)

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