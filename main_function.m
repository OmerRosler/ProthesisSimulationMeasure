function [video_file_name] = main_function(chosenfile, SigmaX,SigmaY, X0, Y0,DFR,EAR,PER,PFO, PFD,Polarity,ElectrodeArraySize, ElectrodeArrayStructure)
folderName = generate_name_for_video_file(chosenfile,SigmaX,SigmaY, X0, Y0,DFR,EAR,PER,PFO, PFD,Polarity,ElectrodeArraySize, ElectrodeArrayStructure);
PER_Factor = 3.9/(PER/(1/DFR));% 3.9 was set for 2 precent threshold,
% i.e. the phosphene falls to 2 precent of its original intensity;
PF_Factor = PFO/(1/EAR);
PolarityVal = Polarity;
sizeval = ElectrodeArraySize;
val = ElectrodeArrayStructure;
if PolarityVal == 1 %%%%% BH %%%%%
    va = 0; vb = 0.333; vc = 0.667; vd = 1;
elseif PolarityVal == 2 %%%%% BH %%%%%
    va = 1; vb = 0.667; vc = 0.333; vd = 0;
end
% MMr - number of rows; MMc - number of columns;
% R = number of pixel rows; C = number of pixel columns

% define number of electrodes
if sizeval == 1 %%%%% 60 electrodes %%%%%
    MMr = 6; MMc = 10; C = 240; R = 144; % In the GUI, choose: phosphene size 0.4; X0 and Y0 = 24
elseif sizeval == 2 %%%%% 256 electrodes %%%%%
    MMr = 16; MMc = 16; C = 224; R = 224; % In the GUI, choose: phosphene size 0.22; X0 and Y0 = 14
elseif sizeval == 3 %%%%% 961 electrodes %%%%%
    MMr = 31; MMc = 31; C = 217; R = 217; % In the GUI, choose: phosphene size 0.13; X0 and Y0 = 7
elseif sizeval == 4 %%%%% 2916 electrodes %%%%%
    MMr = 54; MMc = 54; C = 216; R = 216; % In the GUI, choose: phosphene size 0.08; X0 and Y0 = 4
end

 % define rectangular or hexagonal electrode array
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
video_file_name = strcat(folderName,'/ProstheticVideo.mp4');
warning(video_file_name);
video = VideoWriter(video_file_name,'MPEG-4');% create new video
video.FrameRate = DFR;
open(video);% open video
% srcFiles = dir('C:\Users\...................................\*.png');% define the folder path
srcFiles = dir(strcat(folderName,'/*.png'));% define the folder path

for p =1:min(count,V.NumFrames*(DFR/V.FrameRate)) % for video
    % for p =1:1  % for image
    framename = strcat(folderName,'/',srcFiles(p).name);% define the folder path
    img = imread(framename); % insert it to 'images'
    writeVideo(video, img);% insert each frame to the video
end
close(video);

end
