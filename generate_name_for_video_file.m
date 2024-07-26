function [folderName] = generate_name_for_video_file(chosenfile,SigmaX,SigmaY, X0, Y0,DFR,EAR,PER,PFO, PFD,Polarity,ElectrodeArraySize, ElectrodeArrayStructure)
    % Check if user selected a file
    if isfile(chosenfile)
        % Extract file parts
        [~, fileName, ext] = fileparts(chosenfile);
    else
        ME = MException('MyComponent:inputError', 'The input video must be in the Matlab current folder');
        throw(ME);
    end
    strs = {fileName, tostr3(SigmaX), tostr3(SigmaY), tostr3(X0), tostr3(Y0), tostr3(DFR), tostr3(EAR), tostr3(PER), tostr3(PFO), tostr3(PFD), tostr3(Polarity), tostr3(ElectrodeArraySize), tostr3(ElectrodeArrayStructure)};
    folderName = strjoin(strs,'_');
    mkdir(folderName);
end
function [str] = tostr3(num)
    str = sprintf('%.3f', num);
end