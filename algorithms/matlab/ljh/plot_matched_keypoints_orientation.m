plot_matched_keypoints_orientation_main();

function plot_matched_keypoints_orientation_main()
close all;

repoRoot = fileparts(mfilename('fullpath'));
outDir = repoRoot;
dpi = 2400;
figWIn = 12;
figHIn = 6;
maxPlot = 200;
baseLen = 12;

addpath(genpath(fullfile(repoRoot, 'PSATF')));
addpath(genpath(fullfile(repoRoot, 'Others')));
addpath(genpath(fullfile(repoRoot, 'TV')));
addpath(genpath(fullfile(repoRoot, 'edges-master')));

assert(exist('Create_Image_space', 'file') == 2, '缺少函数 Create_Image_space。');
assert(exist('WSSF_features', 'file') == 2, '缺少函数 WSSF_features。');
assert(exist('GLOH_descriptors', 'file') == 2, '缺少函数 GLOH_descriptors。');
assert(exist('BackProjection', 'file') == 2, '缺少函数 BackProjection。');
assert(exist('matchFeatures', 'file') == 2, '缺少函数 matchFeatures（需要 Computer Vision Toolbox）。');
assert(exist('detectKAZEFeatures', 'file') == 2, '缺少函数 detectKAZEFeatures（需要 Computer Vision Toolbox）。');
assert(exist('adapthisteq', 'file') == 2, '缺少函数 adapthisteq（需要 Image Processing Toolbox）。');
assert(exist('rgb2gray', 'file') == 2, '缺少函数 rgb2gray（需要 Image Processing Toolbox）。');

modelFile = fullfile(repoRoot, 'edges-master', 'models', 'forest', 'modelBsds.mat');
assert(exist(modelFile, 'file') == 2, '缺少模型文件：%s', modelFile);

[optPath, sarPath] = resolve_pair(repoRoot);
optRaw = imread(optPath);
sarRaw = imread(sarPath);

tv_lambda = 1.0;
tv_iterations = 50;

Path_Block = 48;
sigma_1 = 1.6;
ratio = 2^(1/3);
ScaleValue = 1.6;
nOctaves = 3;
filter = 5;
Scale = 'YES';

optShow = optRaw;
sarShow = sarRaw;

[image_1] = preprocess_optical_like_demo(optRaw);
[image_2, sar_raw_db] = preprocess_sar_like_demo(sarRaw, tv_iterations, tv_lambda);

origDir = pwd;
cdCleanup = onCleanup(@() cd(origDir));
cd(repoRoot);
[nonelinear_space_1, E_space_1, Max_space_1, Min_space_1, Phase_space_1] = Create_Image_space(image_1, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);
[nonelinear_space_2, E_space_2, Max_space_2, Min_space_2, Phase_space_2] = Create_Image_space(image_2, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);

[Bolb_KeyPts_1, Corner_KeyPts_1, Bolb_gradient_1, Corner_gradient_1, Bolb_angle_1, Corner_angle_1] = WSSF_features(nonelinear_space_1, E_space_1, Max_space_1, Min_space_1, Phase_space_1, sigma_1, ratio, Scale, nOctaves);
[Bolb_KeyPts_2, Corner_KeyPts_2, Bolb_gradient_2, Corner_gradient_2, Bolb_angle_2, Corner_angle_2] = WSSF_features(nonelinear_space_2, E_space_2, Max_space_2, Min_space_2, Phase_space_2, sigma_1, ratio, Scale, nOctaves);

Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio, sigma_1);
Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio, sigma_1);
Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio, sigma_1);
Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio, sigma_1);

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
plot_match_group('Blob', Bolb_descriptors_1, Bolb_descriptors_2, optShow, sarShow, ScaleValue, maxPlot, baseLen, dpi, figWIn, figHIn, outDir, timestamp);
plot_match_group('Corner', Corner_descriptors_1, Corner_descriptors_2, optShow, sarShow, ScaleValue, maxPlot, baseLen, dpi, figWIn, figHIn, outDir, timestamp);
end

function plot_match_group(tag, desc1, desc2, optShow, sarShow, ScaleValue, maxPlot, baseLen, dpi, figWIn, figHIn, outDir, timestamp)
[indexPairs, matchMetric] = matchFeatures(desc1.des, desc2.des, 'MaxRatio', 1, 'MatchThreshold', 50, 'Unique', true);
if isempty(indexPairs)
    fprintf('%s: no matches\n', tag);
    return
end

if ~isempty(matchMetric)
    [~, ord] = sort(matchMetric, 'ascend');
    ord = ord(1:min(numel(ord), maxPlot));
    indexPairs = indexPairs(ord, :);
else
    indexPairs = indexPairs(1:min(size(indexPairs, 1), maxPlot), :);
end

locs1 = desc1.locs(indexPairs(:, 1), :);
locs2 = desc2.locs(indexPairs(:, 2), :);

[pts1, pts2] = BackProjection(locs1(:, 1:3), locs2(:, 1:3), ScaleValue);
a1 = normalize_angle_for_plot(locs1(:, 4));
a2 = normalize_angle_for_plot(locs2(:, 4));
len1 = baseLen .* (ScaleValue .^ (double(locs1(:, 3)) - 1));
len2 = baseLen .* (ScaleValue .^ (double(locs2(:, 3)) - 1));

f = create_figure(figWIn, figHIn);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
show_image(optShow);
hold on;
draw_points_and_dirs(pts1, a1, len1, [0 1 0]);
title(sprintf('光学 %s 匹配点 + 主方向 (%d)', tag, size(pts1, 1)));

nexttile;
show_sar(sarShow);
hold on;
draw_points_and_dirs(pts2, a2, len2, [1 0 0]);
title(sprintf('SAR %s 匹配点 + 主方向 (%d)', tag, size(pts2, 1)));

sgtitle(sprintf('%s 匹配点主方向可视化 | 空心圆=关键点 直线=主方向', tag));

outFile = fullfile(outDir, sprintf('matched_orient_%s_%s.png', tag, timestamp));
drawnow;
print(f, outFile, '-dpng', sprintf('-r%d', dpi));
fprintf('Saved: %s\n', outFile);
close(f);
end

function show_image(img)
if ndims(img) == 3 && size(img, 3) == 3
    imshow(img);
else
    imshow(img, []);
end
end

function show_sar(sarShow)
sar = sar_to_gray_double01(sarShow);
imshow(sar, []);
end

function draw_points_and_dirs(pts, angDeg, len, color)
x = pts(:, 1);
y = pts(:, 2);
plot(x, y, 'o', 'Color', color, 'MarkerSize', 5, 'LineWidth', 1, 'MarkerFaceColor', 'none');

x2 = x + len(:) .* cosd(angDeg(:));
y2 = y - len(:) .* sind(angDeg(:));

X = [x'; x2'; nan(1, numel(x))];
Y = [y'; y2'; nan(1, numel(y))];
line(X(:), Y(:), 'Color', color, 'LineWidth', 1);
end

function ang = normalize_angle_for_plot(angIn)
ang = double(angIn(:));
isBin = ang >= 0 & ang <= 12 & abs(ang - round(ang)) < 1e-9;
ang(isBin) = ang(isBin) * 30;
ang = mod(ang, 360);
end

function [optPath, sarPath] = resolve_pair(repoRoot)
optPath = fullfile(repoRoot, 'ht_eval_for_own_origin', 'pair1_1.jpg');
sarPath = fullfile(repoRoot, 'ht_eval_for_own_origin', 'pair1_2.jpg');
if exist(optPath, 'file') == 2 && exist(sarPath, 'file') == 2
    return
end
assert(exist('peppers.png', 'file') == 2, '未找到默认光学示例图 peppers.png。');
assert(exist('cameraman.tif', 'file') == 2, '未找到默认 SAR 示例图 cameraman.tif。');
optPath = 'peppers.png';
sarPath = 'cameraman.tif';
end

function [image_1] = preprocess_optical_like_demo(opt_raw)
image_3 = uint8(opt_raw);
if size(image_3, 3) > 1
    image_3 = rgb2gray(image_3);
end
image_3 = double(image_3);
image_3 = uint8(255 * mat2gray(image_3));
if size(image_3, 3) == 1
    image_3 = cat(3, image_3, image_3, image_3);
end
image_3 = adapthisteq(mat2gray(image_3(:, :, 1)));
image_3 = cat(3, image_3, image_3, image_3);
image_1 = im2double(image_3);
end

function [image_2, sar_raw_db] = preprocess_sar_like_demo(sar_raw, tv_iterations, tv_lambda)
image_4 = uint8(sar_raw);
if size(image_4, 3) > 1
    image_4 = rgb2gray(image_4);
end
sar_raw_double = double(image_4);
sar_raw_db = 10 * log10(max(sar_raw_double, eps));
image_4 = log_total_variation(sar_raw_double, tv_iterations, tv_lambda);
image_4 = uint8(255 * mat2gray(image_4));
if size(image_4, 3) == 1
    image_4 = cat(3, image_4, image_4, image_4);
end
image_4 = adapthisteq(mat2gray(image_4(:, :, 1)));
image_4 = cat(3, image_4, image_4, image_4);
image_2 = im2double(image_4);
end

function f = create_figure(wIn, hIn)
f = figure('Color', 'w', 'Visible', 'off');
set(f, 'Units', 'inches');
pos = get(f, 'Position');
set(f, 'Position', [pos(1) pos(2) wIn hIn]);
set(f, 'PaperPositionMode', 'auto');
end

function sar = sar_to_gray_double01(sarShow)
sar = sarShow;
if ndims(sar) == 3
    sar = sar(:, :, 1);
end
if ~isfloat(sar)
    sar = double(sar) / double(intmax(class(sar)));
else
    sar = double(sar);
    mn = min(sar(:));
    mx = max(sar(:));
    if isfinite(mn) && isfinite(mx) && mx > mn
        sar = (sar - mn) / (mx - mn);
    end
end
sar = max(min(sar, 1), 0);
end

function cl = robust_clim(img)
v = img(isfinite(img));
if isempty(v)
    cl = [-30 0];
    return
end
v = sort(v(:));
n = numel(v);
i1 = max(1, round(0.01 * n));
i2 = max(i1, round(0.99 * n));
cl = [v(i1) v(i2)];
if ~isfinite(cl(1)) || ~isfinite(cl(2)) || cl(1) == cl(2)
    cl = [v(1) v(end)];
end
end
