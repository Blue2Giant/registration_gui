sar_path = 'D:\Edge_download\ht_eval_pair_affine\pairs_affine\pair1_2.jpg';
iter_max = 50;
lambda_list = [0.1, 0.5, 1, 5,10];

img = imread(sar_path);
if size(img,3) > 1
    img = rgb2gray(img);
end
img = double(img);

results = cell(1, numel(lambda_list));
for i = 1:numel(lambda_list)
    results{i} = log_total_variation(img, iter_max, lambda_list(i));
end

figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
imshow(uint8(img));
title('input');

for i = 1:numel(lambda_list)
    nexttile;
    imshow(uint8(results{i}));
    title(['lambda=' num2str(lambda_list(i))]);
end
