clc;
clear;
close all;
beep off;
warning('off');
addpath(genpath('PSATF'));
addpath(genpath('Others'));
addpath(genpath('TV'));

%input_dir = 'D:\hand_craft_registration\SRIF-master\dataset\HT';
input_dir = './pairs_random_affine_match';
%input_dir = 'D:\hand_craft_registration\SRIF-master\dataset\Optical-SAR';
use_log_tv = true;
if use_log_tv
    result_tag = 'logtv';
    save_dir = fullfile(pwd, 'save_image_eval_logtv_raw');
else
    result_tag = 'notv';
    save_dir = fullfile(pwd, 'save_image_eval_notv_raw');
end
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

Path_Block=48;
sigma_1=1.6;
ratio=2^(1/3);
ScaleValue = 1.6;
nOctaves = 3;
filter = 5;
Scale ='YES';

tv_lambda = 1.0;
tv_iterations = 50;

max_pairs = 100;
RES = [];
detail_txt = fullfile(pwd, ['RES_rift_' result_tag '_detail.txt']);
fid = fopen(detail_txt, 'w');

for i = 1:max_pairs
    t_start = tic;
    name_1 = ['pair' num2str(i) '_1.jpg'];
    name_2 = ['pair' num2str(i) '_2.jpg'];
    gt_name = ['pair' num2str(i) '.txt'];
    base_name = ['pair' num2str(i)];
    disp(['Processing ', base_name]);
    path_1 = fullfile(input_dir, name_1);
    path_2 = fullfile(input_dir, name_2);
    gt_path = fullfile(input_dir, gt_name);
    if ~exist(path_1, 'file') || ~exist(path_2, 'file')
        disp(['Skip ', base_name, ' (pair missing)']);
        continue
    end
    if ~exist(gt_path, 'file')
        disp(['Skip ', base_name, ' (gt missing)']);
        continue
    end
    image_3 = imread(path_1);
    image_4 = imread(path_2);
    image1 = image_3; 
    image2 = image_4;

    if size(image_3,3) > 1
        image_3 = rgb2gray(image_3);
    end
    if size(image_4,3) > 1
        image_4 = rgb2gray(image_4);
    end
    image_3 = double(image_3);
    image_4 = double(image_4);
    if use_log_tv
        image_4 = log_total_variation(image_4, tv_iterations, tv_lambda);
    end
    image_3 = uint8(255 * mat2gray(image_3));
    image_4 = uint8(255 * mat2gray(image_4));

    if size(image_3,3)==1
        image_3 = cat(3, image_3,image_3,image_3);
    end
    if size(image_4,3)==1
        image_4 = cat(3, image_4,image_4,image_4);
    end

    image_3 = adapthisteq(mat2gray(image_3(:,:,1)));
    image_4 = adapthisteq(mat2gray(image_4(:,:,1)));
    image_3 = cat(3, image_3,image_3,image_3);
    image_4 = cat(3, image_4,image_4,image_4);
    image_1 = im2double(image_3);
    image_2 = im2double(image_4);

    [nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1]=Create_Image_space(image_1,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2]=Create_Image_space(image_2,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);

    [Bolb_KeyPts_1,Corner_KeyPts_1,Bolb_gradient_1,Corner_gradient_1,Bolb_angle_1,Corner_angle_1]  =  WSSF_features(nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1,sigma_1,ratio,Scale,nOctaves);
    [Bolb_KeyPts_2,Corner_KeyPts_2,Bolb_gradient_2,Corner_gradient_2,Bolb_angle_2,Corner_angle_2]  =  WSSF_features(nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2,sigma_1,ratio,Scale,nOctaves);

    Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio,sigma_1);
    Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio,sigma_1);
    Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio,sigma_1);
    Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio,sigma_1);

    [indexPairs,~]= matchFeatures(Bolb_descriptors_1.des,Bolb_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
    if isempty(indexPairs)
        disp(['Skip ', base_name, ' (no blob matches)']);
        continue
    end
    [matchedPoints_1_1,matchedPoints_1_2] = BackProjection(Bolb_descriptors_1.locs(indexPairs(:, 1), :),Bolb_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);
    [indexPairs,~]= matchFeatures(Corner_descriptors_1.des,Corner_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
    if isempty(indexPairs)
        disp(['Skip ', base_name, ' (no corner matches)']);
        continue
    end
    [matchedPoints_2_1,matchedPoints_2_2] = BackProjection(Corner_descriptors_1.locs(indexPairs(:, 1), :),Corner_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);

    matchedPoints_1 = [matchedPoints_1_1;matchedPoints_2_1];
    matchedPoints_2 = [matchedPoints_1_2;matchedPoints_2_2];
    if size(matchedPoints_1,1) < 3
        disp(['Skip ', base_name, ' (insufficient matches)']);
        continue
    end

    [H1,~]=FSC(matchedPoints_1,matchedPoints_2,'affine',3);
    Y_=H1*[matchedPoints_1(:,[1,2])';ones(1,size(matchedPoints_1,1))];
    Y_(1,:)=Y_(1,:)./Y_(3,:);
    Y_(2,:)=Y_(2,:)./Y_(3,:);
    E=sqrt(sum((Y_(1:2,:)-matchedPoints_2(:,[1,2])').^2));
    inliersIndex=E < 3;
    clearedPoints1 = matchedPoints_1(inliersIndex, :);
    clearedPoints2 = matchedPoints_2(inliersIndex, :);

    [clearedPoints2,IA]=unique(clearedPoints2,'rows');
    clearedPoints1=clearedPoints1(IA,:);

    match_name = [base_name '_match.jpg'];
    chess_name = [base_name '_chess.jpg'];
    cp_showMatch(image1, image2, clearedPoints1,clearedPoints2,[],match_name,save_dir);
    image_fusion(image2,image1,double(H1),save_dir);
    temp_chess = fullfile(save_dir, 'Fused image of the board.jpg');
    temp_fusion = fullfile(save_dir, 'fusion image.jpg');
    if exist(temp_chess,'file')
        movefile(temp_chess, fullfile(save_dir, chess_name), 'f');
    end
    if exist(temp_fusion,'file')
        delete(temp_fusion);
    end

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
        disp(['Skip ', base_name, ' (gt shape invalid)']);
        continue
    end

    pts1 = clearedPoints1(:,[1,2])';
    pts1 = [pts1; ones(1,size(pts1,2))];
    Y_gt = H_gt * pts1;
    Y_gt(1,:) = Y_gt(1,:)./Y_gt(3,:);
    Y_gt(2,:) = Y_gt(2,:)./Y_gt(3,:);
    diff_gt = Y_gt(1:2,:) - clearedPoints2(:,[1,2])';
    E_gt = sqrt(sum(diff_gt.^2,1));
    %inliers_gt = E_gt < 25;
    %if sum(inliers_gt) <= 4
    %    rmse = 20;
    %else
    %    rmse = sqrt(sum(E_gt(inliers_gt).^2)/sum(inliers_gt));
    %end
    if length(E_gt)<=4
        rmse = 20;
    else
        rmse = sum(E_gt)/size(E_gt,2);
        rmse = rmse/100;
    end
    time = toc(t_start);
    RES = [RES; time rmse size(clearedPoints1,1)];
    fprintf(fid, '%s %.6f %.6f %d\n', base_name, time, rmse, size(clearedPoints1,1));
    disp(['Done ', base_name, ' | matches: ', num2str(size(clearedPoints1,1)), ' | rmse: ', num2str(rmse), ' | time: ', num2str(time)]);
end
fclose(fid);

if isempty(RES)
    avg_time = NaN;
    avg_rmse = NaN;
    avg_matches = NaN;
else
    avg_time = mean(RES(:,1));
    avg_rmse = mean(RES(:,2));
    avg_matches = mean(RES(:,3));
end

disp(['Average matches: ', num2str(avg_matches)]);
disp(['Average RMSE: ', num2str(avg_rmse)]);
disp(['Average time: ', num2str(avg_time)]);

save(fullfile(pwd, ['RES_wssf_' result_tag '.mat']), 'RES')
