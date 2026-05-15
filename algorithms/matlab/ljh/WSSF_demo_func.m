function [matchedPoints1, matchedPoints2, matches] = WSSF_demo_func(im1, im2, matchesPath, outDir)
close all;
beep off;
warning('off');
addpath(genpath('PSATF'));
addpath(genpath('Others'));
%addpath(genpath('TV'));
if nargin < 3 || isempty(matchesPath)
    matchesPath = '.\matches.txt';
end
if nargin < 4 || isempty(outDir)
    outDir = fileparts(matchesPath);
    if isempty(outDir)
        outDir = pwd;
    end
end
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
matchedPoints1 = zeros(0,2);
matchedPoints2 = zeros(0,2);
matches = zeros(0,4);
transformModel = 'fsc-affine';
rmse = NaN;
H1 = eye(3);
nativeMatchesVisPath = fullfile(outDir, 'matches_vis.jpg');
nativeCheckerboardPath = fullfile(outDir, 'checkerboard.jpg');
nativeFusionPath = fullfile(outDir, 'fusion.jpg');
nativeHPath = fullfile(outDir, 'H_fsc_affine_3x3.txt');
nativeMetaPath = fullfile(outDir, 'native_result.txt');
try
    if ischar(im1) || isstring(im1)
        image_3 = uint8(imread(im1));
    else
        image_3 = uint8(im1);
    end
    if ischar(im2) || isstring(im2)
        image_4 = uint8(imread(im2));
    else
        image_4 = uint8(im2);
    end
    if size(image_3,3)==1
        image_3 = cat(3, image_3,image_3,image_3);
    end
    if size(image_4,3)==1
        image_4 = cat(3, image_4,image_4,image_4);
    end
    image1 = image_3;
    image2 = image_4;
    image_3 = adapthisteq(rgb2gray(image_3));
    image_4 = adapthisteq(rgb2gray(image_4));
    image_3 = cat(3, image_3,image_3,image_3);
    image_4 = cat(3, image_4,image_4,image_4);
    image_1 = im2double(image_3);
    image_2 = im2double(image_4);
    Path_Block=48;
    sigma_1=1.6;
    ratio=2^(1/3);
    ScaleValue = 1.6;
    nOctaves = 3;
    filter = 5;
    Scale ='YES';
    [nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1] = Create_Image_space(image_1,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2] = Create_Image_space(image_2,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [Bolb_KeyPts_1,Corner_KeyPts_1,Bolb_gradient_1,Corner_gradient_1,Bolb_angle_1,Corner_angle_1] = WSSF_features(nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1,sigma_1,ratio,Scale,nOctaves);
    [Bolb_KeyPts_2,Corner_KeyPts_2,Bolb_gradient_2,Corner_gradient_2,Bolb_angle_2,Corner_angle_2] = WSSF_features(nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2,sigma_1,ratio,Scale,nOctaves);
    Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio,sigma_1);
    Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio,sigma_1);
    Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio,sigma_1);
    Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio,sigma_1);

    rawPoints1 = zeros(0, 2);
    rawPoints2 = zeros(0, 2);

    if ~isempty(Bolb_descriptors_1.des) && ~isempty(Bolb_descriptors_2.des)
        [indexPairs,~] = matchFeatures(Bolb_descriptors_1.des,Bolb_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
        if ~isempty(indexPairs)
            [matchedPoints_1_1,matchedPoints_1_2] = BackProjection(Bolb_descriptors_1.locs(indexPairs(:, 1), :),Bolb_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);
            rawPoints1 = [rawPoints1; matchedPoints_1_1];
            rawPoints2 = [rawPoints2; matchedPoints_1_2];
        end
    end

    if ~isempty(Corner_descriptors_1.des) && ~isempty(Corner_descriptors_2.des)
        [indexPairs,~] = matchFeatures(Corner_descriptors_1.des,Corner_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
        if ~isempty(indexPairs)
            [matchedPoints_2_1,matchedPoints_2_2] = BackProjection(Corner_descriptors_1.locs(indexPairs(:, 1), :),Corner_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);
            rawPoints1 = [rawPoints1; matchedPoints_2_1];
            rawPoints2 = [rawPoints2; matchedPoints_2_2];
        end
    end

    if isempty(rawPoints1) || size(rawPoints1,1) < 3
        matchedPoints1 = rawPoints1;
        matchedPoints2 = rawPoints2;
    else
        [H1,rmse] = FSC(rawPoints1,rawPoints2,'affine',3);
        Y_ = H1 * [rawPoints1(:,[1,2])'; ones(1, size(rawPoints1,1))];
        Y_(1,:) = Y_(1,:) ./ Y_(3,:);
        Y_(2,:) = Y_(2,:) ./ Y_(3,:);
        E = sqrt(sum((Y_(1:2,:) - rawPoints2(:,[1,2])').^2));
        inliersIndex = E < 3;
        matchedPoints1 = rawPoints1(inliersIndex, :);
        matchedPoints2 = rawPoints2(inliersIndex, :);

        [matchedPoints2, IA] = unique(matchedPoints2,'rows');
        matchedPoints1 = matchedPoints1(IA,:);
    end

    matches = [matchedPoints1 matchedPoints2];
    dlmwrite(matchesPath, matches, 'delimiter', ' ');
    dlmwrite(nativeHPath, H1, 'delimiter', ' ', 'precision', '%.10f');
    cp_showMatch(image1, image2, matchedPoints1, matchedPoints2, [], 'matches_vis.jpg', outDir);
    image_fusion(image2, image1, double(H1), outDir);

    tempBoardPath = fullfile(outDir, 'Fused image of the board.jpg');
    tempFusionPath = fullfile(outDir, 'fusion image.jpg');
    if exist(tempBoardPath, 'file')
        if exist(nativeCheckerboardPath, 'file')
            delete(nativeCheckerboardPath);
        end
        movefile(tempBoardPath, nativeCheckerboardPath, 'f');
    end
    if exist(tempFusionPath, 'file')
        if exist(nativeFusionPath, 'file')
            delete(nativeFusionPath);
        end
        movefile(tempFusionPath, nativeFusionPath, 'f');
    end
catch
end
dlmwrite(matchesPath, matches, 'delimiter', ' ');
dlmwrite(nativeHPath, H1, 'delimiter', ' ', 'precision', '%.10f');
fid = fopen(nativeMetaPath, 'w');
if fid ~= -1
    fprintf(fid, 'transform_model=%s\n', transformModel);
    fprintf(fid, 'matches_path=%s\n', matchesPath);
    fprintf(fid, 'matches_vis_path=%s\n', nativeMatchesVisPath);
    fprintf(fid, 'checkerboard_path=%s\n', nativeCheckerboardPath);
    fprintf(fid, 'fusion_path=%s\n', nativeFusionPath);
    fprintf(fid, 'H_path=%s\n', nativeHPath);
    fprintf(fid, 'rmse=%.10f\n', rmse);
    fprintf(fid, 'matches_count=%d\n', size(matches, 1));
    fprintf(fid, 'inliers_count=%d\n', size(matches, 1));
    fclose(fid);
end
