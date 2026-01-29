sar_path = 'D:\Edge_download\ht_eval_pair\pairs\pair1_2.jpg';
opt_path = 'D:\Edge_download\ht_eval_pair\pairs\pair1_1.jpg';
iter_max = 50;
lambda_list = [ 1.0, 1.5 ,2.0, 3.0];

sar_img = imread(sar_path);
if size(sar_img,3) > 1
    sar_img = rgb2gray(sar_img);
end
sar_img = double(sar_img);

opt_img = imread(opt_path);
if size(opt_img,3) > 1
    opt_img = rgb2gray(opt_img);
end
opt_img = double(opt_img);

log_results = cell(1, numel(lambda_list));
tv_results = cell(1, numel(lambda_list));

for i = 1:numel(lambda_list)
    log_results{i} = log_total_variation(sar_img, iter_max, lambda_list(i));
    tv_results{i} = total_variation(opt_img, iter_max, lambda_list(i));
end

figure;
tiledlayout(2, numel(lambda_list)+1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imshow(uint8(sar_img));
title('log\_tv SAR input');

for i = 1:numel(lambda_list)
    nexttile;
    imshow(uint8(log_results{i}));
    title(['log\_tv ' num2str(lambda_list(i))]);
end

nexttile;
imshow(uint8(opt_img));
title('tv OPT input');

for i = 1:numel(lambda_list)
    nexttile;
    imshow(uint8(tv_results{i}));
    title(['tv ' num2str(lambda_list(i))]);
end
