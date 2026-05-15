function [matchedPoints1, matchedPoints2, matches] = WSSF_demo_func_relaxed(im1, im2, matchesPath)
%WSSF_DEMO_FUNC_RELAXED Relaxed wrapper around WSSF_demo_func.
% Keeps blob/corner matches even if one branch is empty.

close all;
beep off;
warning('off');

if nargin < 3 || isempty(matchesPath)
    matchesPath = '.\matches.txt';
end

outDir = fileparts(matchesPath);
if ~isempty(outDir) && exist(outDir, 'dir') == 0
    mkdir(outDir);
end

matchedPoints1 = zeros(0, 2);
matchedPoints2 = zeros(0, 2);
matches = zeros(0, 4);

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

    if size(image_3, 3) == 1
        image_3 = cat(3, image_3, image_3, image_3);
    end
    if size(image_4, 3) == 1
        image_4 = cat(3, image_4, image_4, image_4);
    end

    image_3 = adapthisteq(rgb2gray(image_3));
    image_4 = adapthisteq(rgb2gray(image_4));
    image_3 = cat(3, image_3, image_3, image_3);
    image_4 = cat(3, image_4, image_4, image_4);
    image_1 = im2double(image_3);
    image_2 = im2double(image_4);

    Path_Block = 48;
    sigma_1 = 1.6;
    ratio = 2^(1/3);
    ScaleValue = 1.6;
    nOctaves = 3;
    filter = 5;
    Scale = 'YES';

    [nonelinear_space_1, E_space_1, Max_space_1, Min_space_1, Phase_space_1] = ...
        Create_Image_space(image_1, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);
    [nonelinear_space_2, E_space_2, Max_space_2, Min_space_2, Phase_space_2] = ...
        Create_Image_space(image_2, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);

    [Bolb_KeyPts_1, Corner_KeyPts_1, Bolb_gradient_1, Corner_gradient_1, Bolb_angle_1, Corner_angle_1] = ...
        WSSF_features(nonelinear_space_1, E_space_1, Max_space_1, Min_space_1, Phase_space_1, sigma_1, ratio, Scale, nOctaves);
    [Bolb_KeyPts_2, Corner_KeyPts_2, Bolb_gradient_2, Corner_gradient_2, Bolb_angle_2, Corner_angle_2] = ...
        WSSF_features(nonelinear_space_2, E_space_2, Max_space_2, Min_space_2, Phase_space_2, sigma_1, ratio, Scale, nOctaves);

    Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio, sigma_1);
    Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio, sigma_1);
    Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio, sigma_1);
    Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio, sigma_1);

    blobPts1 = zeros(0, 2);
    blobPts2 = zeros(0, 2);
    cornerPts1 = zeros(0, 2);
    cornerPts2 = zeros(0, 2);

    if ~isempty(Bolb_descriptors_1.des) && ~isempty(Bolb_descriptors_2.des)
        [indexPairsBlob, ~] = matchFeatures( ...
            Bolb_descriptors_1.des, Bolb_descriptors_2.des, ...
            'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
        if ~isempty(indexPairsBlob)
            [blobPts1, blobPts2] = BackProjection( ...
                Bolb_descriptors_1.locs(indexPairsBlob(:, 1), :), ...
                Bolb_descriptors_2.locs(indexPairsBlob(:, 2), :), ...
                ScaleValue);
        end
    end

    if ~isempty(Corner_descriptors_1.des) && ~isempty(Corner_descriptors_2.des)
        [indexPairsCorner, ~] = matchFeatures( ...
            Corner_descriptors_1.des, Corner_descriptors_2.des, ...
            'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
        if ~isempty(indexPairsCorner)
            [cornerPts1, cornerPts2] = BackProjection( ...
                Corner_descriptors_1.locs(indexPairsCorner(:, 1), :), ...
                Corner_descriptors_2.locs(indexPairsCorner(:, 2), :), ...
                ScaleValue);
        end
    end

    matchedPoints1 = [blobPts1; cornerPts1];
    matchedPoints2 = [blobPts2; cornerPts2];

    if ~isempty(matchedPoints1)
        pairMat = round([matchedPoints1 matchedPoints2] * 1000) / 1000;
        [~, ia] = unique(pairMat, 'rows', 'stable');
        matchedPoints1 = matchedPoints1(ia, :);
        matchedPoints2 = matchedPoints2(ia, :);
        matches = [matchedPoints1 matchedPoints2];
    end
catch
    matchedPoints1 = zeros(0, 2);
    matchedPoints2 = zeros(0, 2);
    matches = zeros(0, 4);
end

dlmwrite(matchesPath, matches, 'delimiter', ' ');
