% Demos the use of project_depth_map.m

% The location of the RAW dataset.
CLIPS_DIR = '~/NYU_v2/nyu2';

% The path to the new saved dataset.
SAVE_IMAGE_PATH = '~/NYU_v2/Processed/Image';
SAVE_DEPTH_PATH = '~/NYU_v2/Processed/Depth';

total_dirs = dir(fullfile(CLIPS_DIR));
Total_count = 0;
for i=0:length(total_dirs)
    % The name of the scene to demo.
    sceneName = char(total_dirs(i).name);
    % The absolute directory of the scene.
    sceneDir = sprintf('%s/%s', CLIPS_DIR, sceneName);
    % The absolute directory of the saved path.
    saved_image_Dir = sprintf('%s/%s', SAVE_IMAGE_PATH, sceneName); 
    saved_depth_Dir = sprintf('%s/%s', SAVE_DEPTH_PATH, sceneName); 
    % Reads the list of frames.
    frameList = get_synched_frames(sceneDir);

    fprintf('Processing scene %s, %d pairs.\n', sceneName, length(frameList));
    if exist(saved_image_Dir,'dir')==0
        mkdir(saved_image_Dir);
    else
        fprintf('images have been saved\n');
        continue
    end
    if exist(saved_depth_Dir,'dir')==0
        mkdir(saved_depth_Dir);
    else
        fprintf('depths have been saved\n');
        continue
    end
    
    k = 1;
    for j=1 : 10 : numel(frameList) %% sampling frequency of video sequence is 10
        %% Load a pair of frames and align them.
        imgRgb = imread(sprintf('%s/%s', sceneDir, frameList(j).rawRgbFilename));
        imgDepth = imread(sprintf('%s/%s', sceneDir, frameList(j).rawDepthFilename));
        imgDepth = swapbytes(imgDepth);
        [imgDepth2, imgRgb2] = project_depth_map(imgDepth, imgRgb);
        imgDepth3 = fill_depth_colorization(im2double(imgRgb2), imgDepth2);
        imgDepth3 = uint16(imgDepth3 * 256);
        %% Saving the image and depth.
        save_image_file = sprintf('%s/%04d.jpg', saved_image_Dir, k);
        save_depth_file = sprintf('%s/%04d.png', saved_depth_Dir, k);
        
        imwrite(imgRgb2, save_image_file);
        imwrite(imgDepth3, save_depth_file);
        k = k+1;
        Total_count = Total_count+1;
        fprintf('Saving the %04dth pairs, %04dth frame, total count %d.\n', k, j, Total_count);
    end
end
fprintf('Acqure the %d image-depth pairs.\n', Total_count);
