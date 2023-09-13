function [scalingFactor, cropROI] = binVideos(scalingFactor, cropROI);
%[scalingFactor, cropROI] = binVideos(scalingFactor, cropROI);
%
% Tool to crop a rectangular ROI on a series of videos and to apply saptial
% down-sampling. cropROI is not a mandatory input.
%
%----------------------------------------------
%% 
%scalingFactor = 0.5; %The final video proportions
[fileNames, filePath] = uigetfile('*.avi','Multiselect','on');
%Select files in same folder
if ~isdir(fullfile(fileparts(filePath(1:end-1)),'caiman')) %Remove the path separator at the end that identifies the path as a directory
mkdir(fileparts(filePath(1:end-1)),'caiman')
end

%% Let the user select a rectangular ROI to crop the video if required
if ~exist('cropROI') || isempty(cropROI)
    vidObj = VideoReader(fullfile(filePath, fileNames{1}));
    temp = read(vidObj, 1); %Read the first frame
    firstFrame = squeeze(temp(:,:,1));
    
    figure;
    imshow(firstFrame);
    cropping = images.roi.Rectangle(gca, 'Position', [1,1,size(firstFrame, 2), size(firstFrame,1)]);
    
    display('Please select a rectangular ROI excluding black background and then press enter')
    pause()
    
    cropROI = round(cropping.Position);
%     %Account for the subtraction in the lower left introduced for display
%     if cropROI(3) == size(firstFrame, 2)-1
%         cropROI(3) = size(firstFrame, 2)
%     end
%     if cropROI(4) == size(firstFrame, 1)-1
%         cropROI(4) = size(firstFrame, 1)
%     end
    
    % Define a flag for cropping
    if ~isequal(cropROI, [1,1,size(firstFrame, 2), size(firstFrame,1)])
        applyCrop = 1;
    else
        applyCrop = 0;
    end
    
else
    applyCrop = 1;
end

%% Load an process movies
for k = 1:length(fileNames)
    vidObj = VideoReader(fullfile(filePath, fileNames{k}));
    
    while hasFrame(vidObj)
        temp = read(vidObj);
    end
    video = squeeze(temp(:,:,1,:));
    
    if applyCrop
        cropped_video = zeros([cropROI(4) , cropROI(3) , size(video,3)],'uint8');
        %Add one to the video size because counting starts at one
        for n = 1:size(video,3)
            cropped_video(:,:,n) = imcrop(video(:,:,n), cropROI);
        end
    else
        cropROI = [1,1,size(video,2), size(video,1)];
        cropped_video = video;
    end
    
    binned_video = imresize(cropped_video,scalingFactor);
    
    vidWriter = VideoWriter(fullfile(fileparts(filePath(1:end-1)),'caiman',['binned_' fileNames{k}]));
    open(vidWriter)
    framesOut(1:size(binned_video,3)) = struct('cdata',[],'colormap',[]);
    for n = 1:size(binned_video,3)
        framesOut(n).cdata = cat(3,binned_video(:,:,n),binned_video(:,:,n),binned_video(:,:,n));
        writeVideo(vidWriter,framesOut(n));
    end
    close(vidWriter);
    clear video; clear cropped_video; clear binned_video;
    display(sprintf('Processed movie no. %d', round(k)))
end

save(fullfile(fileparts(filePath(1:end-1)),'caiman','cropping_binning'),'scalingFactor','cropROI');
display('------------------------')
end