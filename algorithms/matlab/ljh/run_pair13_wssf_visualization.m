close all;
clc;
beep off;
warning('off');

base_dir = fileparts(mfilename('fullpath'));
orig_dir = pwd;
cleanup_obj = onCleanup(@() cd(orig_dir)); %#ok<NASGU>
cd(base_dir);

addpath(genpath(fullfile(base_dir, 'PSATF')));
addpath(genpath(fullfile(base_dir, 'Others')));
addpath(genpath(fullfile(base_dir, 'TV')));

image_path_1 = fullfile(base_dir, 'ht_eval_for_own_affine', 'pair13_1.jpg');
image_path_2 = fullfile(base_dir, 'ht_eval_for_own_affine', 'pair13_2.jpg');
save_dir = fullfile(base_dir, 'save_image_pair13');

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

image_3 = imread(image_path_1);
image_4 = imread(image_path_2);
image1 = image_3;
image2 = image_4;

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

disp('Start WSSF processing for pair13...');

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

matchedPoints_1 = zeros(0, 2);
matchedPoints_2 = zeros(0, 2);

[indexPairsBlob, ~] = matchFeatures(Bolb_descriptors_1.des, Bolb_descriptors_2.des, ...
    'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
if ~isempty(indexPairsBlob)
    [matchedPoints_1_1, matchedPoints_1_2] = BackProjection( ...
        Bolb_descriptors_1.locs(indexPairsBlob(:, 1), :), ...
        Bolb_descriptors_2.locs(indexPairsBlob(:, 2), :), ScaleValue);
    matchedPoints_1 = [matchedPoints_1; matchedPoints_1_1];
    matchedPoints_2 = [matchedPoints_2; matchedPoints_1_2];
end

[indexPairsCorner, ~] = matchFeatures(Corner_descriptors_1.des, Corner_descriptors_2.des, ...
    'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
if ~isempty(indexPairsCorner)
    [matchedPoints_2_1, matchedPoints_2_2] = BackProjection( ...
        Corner_descriptors_1.locs(indexPairsCorner(:, 1), :), ...
        Corner_descriptors_2.locs(indexPairsCorner(:, 2), :), ScaleValue);
    matchedPoints_1 = [matchedPoints_1; matchedPoints_2_1];
    matchedPoints_2 = [matchedPoints_2; matchedPoints_2_2];
end

if size(matchedPoints_1, 1) < 3
    error('WSSF did not produce enough matches for pair13.');
end

[H1, rmse] = FSC(matchedPoints_1, matchedPoints_2, 'affine', 3);
Y_ = H1 * [matchedPoints_1(:, [1, 2])'; ones(1, size(matchedPoints_1, 1))];
Y_(1, :) = Y_(1, :) ./ Y_(3, :);
Y_(2, :) = Y_(2, :) ./ Y_(3, :);
E = sqrt(sum((Y_(1:2, :) - matchedPoints_2(:, [1, 2])').^2));
inliersIndex = E < 3;

clearedPoints1 = matchedPoints_1(inliersIndex, :);
clearedPoints2 = matchedPoints_2(inliersIndex, :);
[clearedPoints2, IA] = unique(clearedPoints2, 'rows');
clearedPoints1 = clearedPoints1(IA, :);

cp_showMatch(image1, image2, clearedPoints1, clearedPoints2, [], 'pair13_match.jpg', save_dir);
image_fusion(image2, image1, double(H1), save_dir);

temp_board = fullfile(save_dir, 'Fused image of the board.jpg');
temp_fusion = fullfile(save_dir, 'fusion image.jpg');
if exist(temp_board, 'file')
    movefile(temp_board, fullfile(save_dir, 'pair13_checkerboard.jpg'), 'f');
end
if exist(temp_fusion, 'file')
    movefile(temp_fusion, fullfile(save_dir, 'pair13_fusion.jpg'), 'f');
end

disp(['Inlier matches: ' num2str(size(clearedPoints1, 1))]);
disp(['RMSE: ' num2str(rmse)]);
disp(['Match figure: ' fullfile(save_dir, 'pair13_match.jpg')]);
disp(['Checkerboard figure: ' fullfile(save_dir, 'pair13_checkerboard.jpg')]);
disp(['Fusion figure: ' fullfile(save_dir, 'pair13_fusion.jpg')]);
