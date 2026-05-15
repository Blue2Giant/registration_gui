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

image_path_1 = fullfile(base_dir, 'ht_eval_for_own_affine', 'pair13_1.jpg');
image_path_2 = fullfile(base_dir, 'ht_eval_for_own_affine', 'pair13_2.jpg');
save_dir = fullfile(base_dir, 'save_image_pair13_keypoints');

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

assert(exist(image_path_1, 'file') == 2, 'Image not found: %s', image_path_1);
assert(exist(image_path_2, 'file') == 2, 'Image not found: %s', image_path_2);

image_show_1 = imread(image_path_1);
image_show_2 = imread(image_path_2);

image_proc_1 = preprocess_like_demo(image_show_1);
image_proc_2 = preprocess_like_demo(image_show_2);

sigma_1 = 1.6;
ratio = 2^(1/3);
ScaleValue = 1.6;
nOctaves = 3;
filter = 5;
Scale = 'YES';
max_points_to_plot = 20;
sample_seed = 13;

disp('Start plotting blob/corner keypoints for pair13...');

[nonelinear_space_1, E_space_1, Max_space_1, Min_space_1, Phase_space_1] = ...
    Create_Image_space(image_proc_1, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);
[nonelinear_space_2, E_space_2, Max_space_2, Min_space_2, Phase_space_2] = ...
    Create_Image_space(image_proc_2, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);

[blob_xy_1, corner_xy_1] = extract_nms_keypoints( ...
    nonelinear_space_1, E_space_1, Max_space_1, Min_space_1, Phase_space_1, sigma_1, ratio, Scale, nOctaves);
[blob_xy_2, corner_xy_2] = extract_nms_keypoints( ...
    nonelinear_space_2, E_space_2, Max_space_2, Min_space_2, Phase_space_2, sigma_1, ratio, Scale, nOctaves);

rng(sample_seed, 'twister');
blob_xy_1_plot = sample_keypoints_for_plot(blob_xy_1, max_points_to_plot);
corner_xy_1_plot = sample_keypoints_for_plot(corner_xy_1, max_points_to_plot);
blob_xy_2_plot = sample_keypoints_for_plot(blob_xy_2, max_points_to_plot);
corner_xy_2_plot = sample_keypoints_for_plot(corner_xy_2, max_points_to_plot);

save_keypoint_figure(image_show_1, blob_xy_1_plot, size(blob_xy_1, 1), 'pair13_1 Blob Keypoints', ...
    fullfile(save_dir, 'pair13_1_blob_keypoints.png'), [0 1 0]);
save_keypoint_figure(image_show_1, corner_xy_1_plot, size(corner_xy_1, 1), 'pair13_1 Corner Keypoints', ...
    fullfile(save_dir, 'pair13_1_corner_keypoints.png'), [1 0 0]);
save_keypoint_figure(image_show_2, blob_xy_2_plot, size(blob_xy_2, 1), 'pair13_2 Blob Keypoints', ...
    fullfile(save_dir, 'pair13_2_blob_keypoints.png'), [0 1 0]);
save_keypoint_figure(image_show_2, corner_xy_2_plot, size(corner_xy_2, 1), 'pair13_2 Corner Keypoints', ...
    fullfile(save_dir, 'pair13_2_corner_keypoints.png'), [1 0 0]);

save_summary_figure(image_show_1, blob_xy_1_plot, size(blob_xy_1, 1), corner_xy_1_plot, size(corner_xy_1, 1), ...
    image_show_2, blob_xy_2_plot, size(blob_xy_2, 1), corner_xy_2_plot, size(corner_xy_2, 1), ...
    fullfile(save_dir, 'pair13_blob_corner_summary.png'));

disp(['pair13_1 blob count: ' num2str(size(blob_xy_1, 1))]);
disp(['pair13_1 corner count: ' num2str(size(corner_xy_1, 1))]);
disp(['pair13_2 blob count: ' num2str(size(blob_xy_2, 1))]);
disp(['pair13_2 corner count: ' num2str(size(corner_xy_2, 1))]);
disp(['Randomly plotted points per figure: up to ' num2str(max_points_to_plot)]);
disp(['Saved figures to: ' save_dir]);

function image_out = preprocess_like_demo(image_in)
if size(image_in, 3) == 1
    image_in = cat(3, image_in, image_in, image_in);
end
gray_image = adapthisteq(rgb2gray(image_in));
gray_image = cat(3, gray_image, gray_image, gray_image);
image_out = im2double(gray_image);
end

function [blob_xy, corner_xy] = extract_nms_keypoints(nonelinear_space, E_space, Max_space, Min_space, Phase_space, sigma_1, ratio, Scale, nOctaves)
[blob_space, corner_space] = WSSF_gradient_feature( ...
    nonelinear_space, E_space, Max_space, Min_space, Phase_space, Scale, nOctaves);

points_layer1 = 5000;
points_layer2 = 5000;
[blob_key_point_array, corner_key_point_array] = FeatureDetection( ...
    blob_space, corner_space, nOctaves, points_layer1, points_layer2, sigma_1, ratio);

blob_key_point_array = unique(blob_key_point_array, 'rows', 'stable');
corner_key_point_array = unique(corner_key_point_array, 'rows', 'stable');

window = 5;
[blob_keypoints, ~] = WSSF_selectMax_NMS(blob_key_point_array, window);
[corner_keypoints, ~] = WSSF_selectMax_NMS(corner_key_point_array, window);

blob_xy = unique(round(blob_keypoints.kpts(:, 1:2)), 'rows', 'stable');
corner_xy = unique(round(corner_keypoints.kpts(:, 1:2)), 'rows', 'stable');
end

function save_keypoint_figure(image_in, xy_plot, total_count, title_text, out_path, color)
f = figure('Color', 'w', 'Visible', 'off');
imshow(image_in, []);
hold on;

if ~isempty(xy_plot)
    plot(xy_plot(:, 1), xy_plot(:, 2), 'o', ...
        'Color', color, ...
        'MarkerSize', 4, ...
        'LineWidth', 0.8, ...
        'MarkerFaceColor', 'none');
end

title(sprintf('%s (plot %d / total %d)', title_text, size(xy_plot, 1), total_count));
set(f, 'PaperPositionMode', 'auto');
print(f, out_path, '-dpng', '-r300');
close(f);
end

function save_summary_figure(image_1, blob_xy_1_plot, blob_count_1, corner_xy_1_plot, corner_count_1, ...
    image_2, blob_xy_2_plot, blob_count_2, corner_xy_2_plot, corner_count_2, out_path)
f = figure('Color', 'w', 'Visible', 'off');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imshow(image_1, []);
hold on;
plot_points(blob_xy_1_plot, [0 1 0]);
title(sprintf('pair13\\_1 Blob (plot %d / total %d)', size(blob_xy_1_plot, 1), blob_count_1));

nexttile;
imshow(image_1, []);
hold on;
plot_points(corner_xy_1_plot, [1 0 0]);
title(sprintf('pair13\\_1 Corner (plot %d / total %d)', size(corner_xy_1_plot, 1), corner_count_1));

nexttile;
imshow(image_2, []);
hold on;
plot_points(blob_xy_2_plot, [0 1 0]);
title(sprintf('pair13\\_2 Blob (plot %d / total %d)', size(blob_xy_2_plot, 1), blob_count_2));

nexttile;
imshow(image_2, []);
hold on;
plot_points(corner_xy_2_plot, [1 0 0]);
title(sprintf('pair13\\_2 Corner (plot %d / total %d)', size(corner_xy_2_plot, 1), corner_count_2));

sgtitle('WSSF Blob / Corner Keypoints for pair13');
set(f, 'PaperPositionMode', 'auto');
print(f, out_path, '-dpng', '-r300');
close(f);
end

function plot_points(xy, color)
if isempty(xy)
    return;
end
plot(xy(:, 1), xy(:, 2), 'o', ...
    'Color', color, ...
    'MarkerSize', 4, ...
    'LineWidth', 0.8, ...
    'MarkerFaceColor', 'none');
end

function xy_plot = sample_keypoints_for_plot(xy, max_points_to_plot)
num_points = size(xy, 1);
if num_points <= max_points_to_plot
    xy_plot = xy;
    return;
end
sample_idx = randperm(num_points, max_points_to_plot);
xy_plot = xy(sample_idx, :);
end
