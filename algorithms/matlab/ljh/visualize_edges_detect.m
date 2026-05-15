close all;
beep off;
warning('off');
addpath(genpath('edges-master'));

input_dir = 'D:\Edge_download\ht_eval_pair\pairs';
opt_path = fullfile(input_dir, 'pair1_1.jpg');
sar_path = fullfile(input_dir, 'pair1_2.jpg');
save_dir = fullfile(pwd, 'edges_visualization');
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

model = load('./edges-master/models/forest/modelBsds.mat');

opt_img = imread(opt_path);
sar_img = imread(sar_path);

if size(opt_img,3) == 1
    opt_img = cat(3, opt_img, opt_img, opt_img);
end
if size(sar_img,3) == 1
    sar_img = cat(3, sar_img, sar_img, sar_img);
end

opt_edge = edgesDetect(opt_img, model.model);
sar_edge = edgesDetect(sar_img, model.model);

imwrite(opt_edge, fullfile(save_dir, 'opt_edges.png'));
imwrite(sar_edge, fullfile(save_dir, 'sar_edges.png'));
