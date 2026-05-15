function metrics = WSSF_eval_pair_metrics_logtv(im1, im2, tv_lambda, tv_iterations)
% Evaluate one image pair with the WSSF + logTV pipeline and return core metrics.

warning('off');

if nargin < 3 || isempty(tv_lambda)
    tv_lambda = 1.0;
end
if nargin < 4 || isempty(tv_iterations)
    tv_iterations = 50;
end

this_file = which(mfilename);
if isempty(this_file)
    base_dir = pwd;
else
    base_dir = fileparts(this_file);
end
addpath(genpath(fullfile(base_dir, 'PSATF')));
addpath(genpath(fullfile(base_dir, 'Others')));
addpath(genpath(fullfile(base_dir, 'TV')));

metrics = struct( ...
    'rmse', NaN, ...
    'match_count', 0, ...
    'success', false, ...
    'status', 'init');

try
    image_3 = read_input_image(im1);
    image_4 = read_input_image(im2);

    if size(image_3, 3) > 1
        image_3 = rgb2gray(image_3);
    end
    if size(image_4, 3) > 1
        image_4 = rgb2gray(image_4);
    end

    image_3 = double(image_3);
    image_4 = double(image_4);
    image_4 = log_total_variation(image_4, tv_iterations, tv_lambda);
    image_3 = uint8(255 * mat2gray(image_3));
    image_4 = uint8(255 * mat2gray(image_4));

    if size(image_3, 3) == 1
        image_3 = cat(3, image_3, image_3, image_3);
    end
    if size(image_4, 3) == 1
        image_4 = cat(3, image_4, image_4, image_4);
    end

    image_3 = adapthisteq(mat2gray(image_3(:, :, 1)));
    image_4 = adapthisteq(mat2gray(image_4(:, :, 1)));
    image_3 = cat(3, image_3, image_3, image_3);
    image_4 = cat(3, image_4, image_4, image_4);
    image_1 = im2double(image_3);
    image_2 = im2double(image_4);

    Path_Block = 48;
    sigma_1 = 1.6;
    ratio = 2^(1 / 3);
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

    if ~isfield(Bolb_descriptors_1, 'des') || ~isfield(Bolb_descriptors_2, 'des') || ...
            isempty(Bolb_descriptors_1.des) || isempty(Bolb_descriptors_2.des)
        metrics.status = 'no_blob_descriptors';
        return;
    end

    if ~isfield(Corner_descriptors_1, 'des') || ~isfield(Corner_descriptors_2, 'des') || ...
            isempty(Corner_descriptors_1.des) || isempty(Corner_descriptors_2.des)
        metrics.status = 'no_corner_descriptors';
        return;
    end

    [indexPairsBlob, ~] = matchFeatures(Bolb_descriptors_1.des, Bolb_descriptors_2.des, ...
        'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
    if isempty(indexPairsBlob)
        metrics.status = 'no_blob_matches';
        return;
    end

    [matchedPoints_1_1, matchedPoints_1_2] = BackProjection( ...
        Bolb_descriptors_1.locs(indexPairsBlob(:, 1), :), ...
        Bolb_descriptors_2.locs(indexPairsBlob(:, 2), :), ...
        ScaleValue);

    [indexPairsCorner, ~] = matchFeatures(Corner_descriptors_1.des, Corner_descriptors_2.des, ...
        'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
    if isempty(indexPairsCorner)
        metrics.status = 'no_corner_matches';
        return;
    end

    [matchedPoints_2_1, matchedPoints_2_2] = BackProjection( ...
        Corner_descriptors_1.locs(indexPairsCorner(:, 1), :), ...
        Corner_descriptors_2.locs(indexPairsCorner(:, 2), :), ...
        ScaleValue);

    matchedPoints_1 = [matchedPoints_1_1; matchedPoints_2_1];
    matchedPoints_2 = [matchedPoints_1_2; matchedPoints_2_2];

    if size(matchedPoints_1, 1) < 3
        metrics.status = 'insufficient_initial_matches';
        return;
    end

    [H1, rmse] = FSC(matchedPoints_1, matchedPoints_2, 'affine', 3);
    Y_ = H1 * [matchedPoints_1(:, [1, 2])'; ones(1, size(matchedPoints_1, 1))];
    Y_(1, :) = Y_(1, :) ./ Y_(3, :);
    Y_(2, :) = Y_(2, :) ./ Y_(3, :);
    E = sqrt(sum((Y_(1:2, :) - matchedPoints_2(:, [1, 2])').^2));
    inliersIndex = E < 3;
    clearedPoints1 = matchedPoints_1(inliersIndex, :);
    clearedPoints2 = matchedPoints_2(inliersIndex, :);

    if isempty(clearedPoints1)
        metrics.rmse = rmse;
        metrics.status = 'no_inliers_after_fsc';
        return;
    end

    [clearedPoints2, IA] = unique(clearedPoints2, 'rows');
    clearedPoints1 = clearedPoints1(IA, :);

    metrics.rmse = rmse;
    metrics.match_count = size(clearedPoints1, 1);
    metrics.success = metrics.match_count > 4;
    metrics.status = 'ok';
catch ME
    metrics.status = ['error: ' ME.identifier];
end

end

function image_data = read_input_image(im)
if ischar(im) || isstring(im)
    image_data = imread(im);
else
    image_data = im;
end

if ~isa(image_data, 'uint8')
    image_data = im2uint8(mat2gray(image_data));
end
end
