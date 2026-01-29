sar_path = 'D:\Edge_download\ht_eval_pair_affine\pairs_affine\pair1_2.jpg';
iter_max = 50;
lambda_list = [0.5, 0.7, 0.8, 0.9, 1.0];

img = imread(sar_path);
if size(img,3) > 1
    img = rgb2gray(img);
end
img = double(img);

log_results = cell(1, numel(lambda_list));
tv_results = cell(1, numel(lambda_list));

for i = 1:numel(lambda_list)
    log_results{i} = log_total_variation(img, iter_max, lambda_list(i));
    tv_results{i} = total_variation(img, iter_max, lambda_list(i));
end

figure;
tiledlayout(2, numel(lambda_list)+1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imshow(uint8(img));
title('log\_tv input');

for i = 1:numel(lambda_list)
    nexttile;
    imshow(uint8(log_results{i}));
    title(['log\_tv ' num2str(lambda_list(i))]);
end

nexttile;
imshow(uint8(img));
title('tv input');

for i = 1:numel(lambda_list)
    nexttile;
    imshow(uint8(tv_results{i}));
    title(['tv ' num2str(lambda_list(i))]);
end
