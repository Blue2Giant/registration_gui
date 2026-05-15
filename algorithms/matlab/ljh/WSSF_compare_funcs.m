close all;
beep off;
warning('off');
addpath(genpath('PSATF'));
addpath(genpath('Others'));
addpath(genpath('TV'));
save_dir = fullfile(pwd, 'save_image_compare');
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end
file_image = 'D:\Edge_download\ht_eval_pair\pairs';
im1 = fullfile(file_image, 'pair1_1.jpg');
im2 = fullfile(file_image, 'pair1_2.jpg');
image1 = uint8(imread(im1));
image2 = uint8(imread(im2));
if size(image1,3)==1
    image1 = cat(3, image1,image1,image1);
end
if size(image2,3)==1
    image2 = cat(3, image2,image2,image2);
end
[mp1_tv, mp2_tv, ~] = WSSF_demo_tv_logtv_func(im1, im2, fullfile(save_dir,'matches_tv.txt'));
[mp1_ws, mp2_ws, ~] = WSSF_demo_func(im1, im2, fullfile(save_dir,'matches_wssf.txt'));
results = struct();
if size(mp1_tv,1) >= 3
    [H_tv, rmse_tv] = FSC(mp1_tv, mp2_tv, 'affine', 3);
    Y_ = H_tv*[mp1_tv(:,[1,2])';ones(1,size(mp1_tv,1))];
    Y_(1,:) = Y_(1,:)./Y_(3,:);
    Y_(2,:) = Y_(2,:)./Y_(3,:);
    E = sqrt(sum((Y_(1:2,:)-mp2_tv(:,[1,2])').^2));
    inliersIndex = E < 3;
    clearedPoints1 = mp1_tv(inliersIndex, :);
    clearedPoints2 = mp2_tv(inliersIndex, :);
    [clearedPoints2,IA] = unique(clearedPoints2,'rows');
    clearedPoints1 = clearedPoints1(IA,:);
    RCM_tv = size(clearedPoints1,1)/size(mp1_tv,1);
    out_dir_tv = fullfile(save_dir,'tv');
    if ~exist(out_dir_tv,'dir'), mkdir(out_dir_tv); end
    cp_showMatch(image1, image2, clearedPoints1, clearedPoints2, [], 'matches_tv.jpg', out_dir_tv);
    image_fusion(image2,image1,double(H_tv),out_dir_tv);
    results.rmse_tv = rmse_tv;
    results.rcm_tv = RCM_tv;
    results.matches_tv = size(clearedPoints1,1);
else
    results.rmse_tv = NaN;
    results.rcm_tv = 0;
    results.matches_tv = 0;
end
if size(mp1_ws,1) >= 3
    [H_ws, rmse_ws] = FSC(mp1_ws, mp2_ws, 'affine', 3);
    Y_ = H_ws*[mp1_ws(:,[1,2])';ones(1,size(mp1_ws,1))];
    Y_(1,:) = Y_(1,:)./Y_(3,:);
    Y_(2,:) = Y_(2,:)./Y_(3,:);
    E = sqrt(sum((Y_(1:2,:)-mp2_ws(:,[1,2])').^2));
    inliersIndex = E < 3;
    clearedPoints1 = mp1_ws(inliersIndex, :);
    clearedPoints2 = mp2_ws(inliersIndex, :);
    [clearedPoints2,IA] = unique(clearedPoints2,'rows');
    clearedPoints1 = clearedPoints1(IA,:);
    RCM_ws = size(clearedPoints1,1)/size(mp1_ws,1);
    out_dir_ws = fullfile(save_dir,'wssf');
    if ~exist(out_dir_ws,'dir'), mkdir(out_dir_ws); end
    cp_showMatch(image1, image2, clearedPoints1, clearedPoints2, [], 'matches_wssf.jpg', out_dir_ws);
    image_fusion(image2,image1,double(H_ws),out_dir_ws);
    results.rmse_wssf = rmse_ws;
    results.rcm_wssf = RCM_ws;
    results.matches_wssf = size(clearedPoints1,1);
else
    results.rmse_wssf = NaN;
    results.rcm_wssf = 0;
    results.matches_wssf = 0;
end
