clc;
clear;
close all;
beep off;
warning('off');
addpath(genpath('PSATF'));
addpath(genpath('Others'));
addpath(genpath('TV'));

input_dir = './ht_eval_for_own_affine';
save_dir = fullfile(pwd, 'save_image_eval');
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

Path_Block = 48;
sigma_1 = 1.6;
ratio = 2^(1/3);
ScaleValue = 1.6;
nOctaves = 3;
filter = 5;
Scale = 'YES';

tv_lambda = 1.0;
tv_iterations = 50;

pair_id = 32;
name_1 = ['pair' num2str(pair_id) '_1.jpg'];
name_2 = ['pair' num2str(pair_id) '_2.jpg'];
gt_name = ['pair' num2str(pair_id) '.txt'];
base_name = ['pair' num2str(pair_id)];
path_1 = fullfile(input_dir, name_1);
path_2 = fullfile(input_dir, name_2);
gt_path = fullfile(input_dir, gt_name);

if ~exist(path_1, 'file') || ~exist(path_2, 'file') || ~exist(gt_path, 'file')
    error('pair32 文件缺失');
end

image_3 = imread(path_1);
image_4 = imread(path_2);
image1 = image_3;

if size(image_3,3) > 1
    image_3 = rgb2gray(image_3);
end
if size(image_4,3) > 1
    image_4 = rgb2gray(image_4);
end
image_3 = double(image_3);
image_4 = double(image_4);

scales = 0.5:0.1:1.5;
results = zeros(numel(scales), 3);

gt = load(gt_path);
if numel(gt) == 6
    H_gt = [reshape(gt,2,3); 0 0 1];
elseif numel(gt) == 9
    H_gt = reshape(gt,3,3);
elseif size(gt,1) == 2 && size(gt,2) == 3
    H_gt = [gt; 0 0 1];
elseif size(gt,1) == 3 && size(gt,2) == 3
    H_gt = gt;
else
    error('gt 形状无效');
end

for k = 1:numel(scales)
    scale = scales(k);
    image_4_scaled = imresize(image_4, scale, 'bilinear');
    image_4_scaled = fit_to_size(image_4_scaled, size(image_4));
    image_4_scaled = double(image_4_scaled);

    image_3_tv = total_variation(image_3, tv_iterations, tv_lambda);
    image_4_tv = log_total_variation(image_4_scaled, tv_iterations, tv_lambda);
    image_3_tv = uint8(255 * mat2gray(image_3_tv));
    image_4_tv = uint8(255 * mat2gray(image_4_tv));

    if size(image_3_tv,3)==1
        image_3_tv = cat(3, image_3_tv, image_3_tv, image_3_tv);
    end
    if size(image_4_tv,3)==1
        image_4_tv = cat(3, image_4_tv, image_4_tv, image_4_tv);
    end

    image_3_eq = adapthisteq(mat2gray(image_3_tv(:,:,1)));
    image_4_eq = adapthisteq(mat2gray(image_4_tv(:,:,1)));
    image_3_eq = cat(3, image_3_eq, image_3_eq, image_3_eq);
    image_4_eq = cat(3, image_4_eq, image_4_eq, image_4_eq);
    image_1 = im2double(image_3_eq);
    image_2 = im2double(image_4_eq);

    [nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1] = Create_Image_space(image_1,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2] = Create_Image_space(image_2,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);

    [Bolb_KeyPts_1,Corner_KeyPts_1,Bolb_gradient_1,Corner_gradient_1,Bolb_angle_1,Corner_angle_1] = WSSF_features(nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1,sigma_1,ratio,Scale,nOctaves);
    [Bolb_KeyPts_2,Corner_KeyPts_2,Bolb_gradient_2,Corner_gradient_2,Bolb_angle_2,Corner_angle_2] = WSSF_features(nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2,sigma_1,ratio,Scale,nOctaves);

    Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio,sigma_1);
    Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio,sigma_1);
    Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio,sigma_1);
    Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio,sigma_1);

    [indexPairs,~] = matchFeatures(Bolb_descriptors_1.des,Bolb_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
    if isempty(indexPairs)
        results(k,:) = [scale, 0, NaN];
        continue
    end
    [matchedPoints_1_1,matchedPoints_1_2] = BackProjection(Bolb_descriptors_1.locs(indexPairs(:, 1), :),Bolb_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);
    [indexPairs,~] = matchFeatures(Corner_descriptors_1.des,Corner_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
    if isempty(indexPairs)
        results(k,:) = [scale, 0, NaN];
        continue
    end
    [matchedPoints_2_1,matchedPoints_2_2] = BackProjection(Corner_descriptors_1.locs(indexPairs(:, 1), :),Corner_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);

    matchedPoints_1 = [matchedPoints_1_1;matchedPoints_2_1];
    matchedPoints_2 = [matchedPoints_1_2;matchedPoints_2_2];
    if size(matchedPoints_1,1) < 3
        results(k,:) = [scale, 0, NaN];
        continue
    end

    [H1,~] = FSC(matchedPoints_1,matchedPoints_2,'affine',3);
    Y_ = H1*[matchedPoints_1(:,[1,2])';ones(1,size(matchedPoints_1,1))];
    Y_(1,:) = Y_(1,:)./Y_(3,:);
    Y_(2,:) = Y_(2,:)./Y_(3,:);
    E = sqrt(sum((Y_(1:2,:)-matchedPoints_2(:,[1,2])').^2));
    inliersIndex = E < 3;
    clearedPoints1 = matchedPoints_1(inliersIndex, :);
    clearedPoints2 = matchedPoints_2(inliersIndex, :);

    [clearedPoints2,IA] = unique(clearedPoints2,'rows');
    clearedPoints1 = clearedPoints1(IA,:);

    pts1 = clearedPoints1(:,[1,2])';
    pts1 = [pts1; ones(1,size(pts1,2))];
    Y_gt = H_gt * pts1;
    Y_gt(1,:) = Y_gt(1,:)./Y_gt(3,:);
    Y_gt(2,:) = Y_gt(2,:)./Y_gt(3,:);
    diff_gt = Y_gt(1:2,:) - clearedPoints2(:,[1,2])';
    E_gt = sqrt(sum(diff_gt.^2,1));
    if length(E_gt) <= 4
        rmse = 20;
    else
        rmse = sum(E_gt)/size(E_gt,2);
    end
    rmse = rmse / 100;

    results(k,:) = [scale, size(clearedPoints1,1), rmse];
end

save(fullfile(pwd, 'pair32_scale_results.mat'), 'results');

figure('Color', 'w', 'Position', [100, 100, 1000, 400]);
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(results(:,1), results(:,2), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Scale');
ylabel('Matches');
title([ ' 匹配点数量']);
grid on;

nexttile;
plot(results(:,1), results(:,3), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Scale');
ylabel('RMSE');
title([ ' 特征点的坐标误差']);
grid on;

function img_out = fit_to_size(img_in, target_size)
    h = target_size(1);
    w = target_size(2);
    img_out = img_in;
    [h_in, w_in] = size(img_out);
    if h_in > h
        start_r = floor((h_in - h)/2) + 1;
        img_out = img_out(start_r:start_r+h-1, :);
    elseif h_in < h
        pad_top = floor((h - h_in)/2);
        pad_bottom = h - h_in - pad_top;
        img_out = padarray(img_out, [pad_top, 0], 0, 'pre');
        img_out = padarray(img_out, [pad_bottom, 0], 0, 'post');
    end
    [h_in, w_in] = size(img_out);
    if w_in > w
        start_c = floor((w_in - w)/2) + 1;
        img_out = img_out(:, start_c:start_c+w-1);
    elseif w_in < w
        pad_left = floor((w - w_in)/2);
        pad_right = w - w_in - pad_left;
        img_out = padarray(img_out, [0, pad_left], 0, 'pre');
        img_out = padarray(img_out, [0, pad_right], 0, 'post');
    end
end
