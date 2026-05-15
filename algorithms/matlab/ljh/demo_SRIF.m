clc;clear;close all; warning('off')
%addpath dataset\Optical-SAR\
%addpath dataset\HT\
addpath 'D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_affine'
addpath algorithms\SRIF\;
addpath algorithms\common\
addpath utils\
RES=[];
dataDir = 'D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_affine';

for i=1:100
    i
    str1 = fullfile(dataDir, ['pair' num2str(i) '_1.jpg']);
    str2 = fullfile(dataDir, ['pair' num2str(i) '_2.jpg']);
    gtstr = fullfile(dataDir, ['pair' num2str(i) '.txt']);

    if exist(str1,'file')==0
        continue;
    end
    gt=load(gtstr);
    im1 = uint8(imread(str1));
    im2 = uint8(imread(str2));
    
    %% 进行TV变换
    %IterMax=30;lambda=0.03; 
    %if (max(im2(:))>2)
    %    im2 = im2/255;
    %end
    %im1=im2double(im1);
    %im2=im2double(im2);
    %im1=total_variation(im1,IterMax,lambda);
    %SAR图像得先做log
    %c = 1;
    %im2 = c * log(1 + im2);
    %im2=total_variation(im2,IterMax,lambda);
    %im2 = exp(im2)-1;
    %归一化到255
    %im2 = (im2 - min(im2(:))) / (max(im2(:)) - min(im2(:)));
    %im2 = im2uint8(im2);
    %im1 = (im1 - min(im1(:))) / (max(im1(:)) - min(im1(:)));
    %im1 = im2uint8(im1);
    %% 
    imwrite(im1,'.\algorithms\SRIF\1.png');
    imwrite(im2,'.\algorithms\SRIF\2.png');

    exe='.\algorithms\SRIF\SRIF.exe';
    cmd = [exe ' ' '.\algorithms\SRIF\1.png' ' ' '.\algorithms\SRIF\2.png' ' ' '128' ' ' '4' ' ' '8' ' ' '5000' ' ' '1' ' ' '1' ' ' '.\algorithms\SRIF\matches.txt'];
    t1=clock();
    system(cmd);
    t2=clock();
    time=etime(t2,t1);
    matches = load('.\algorithms\SRIF\matches.txt');

    matchedPoints1 = matches(:,1:2);
    matchedPoints2 = matches(:,3:4);

    H=[gt;0 0 1];
    Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
    Y_(1,:)=Y_(1,:)./Y_(3,:);
    Y_(2,:)=Y_(2,:)./Y_(3,:);
    E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
    inliersIndex=E<3;
    save_dir = fullfile(pwd, 'save_image', 'srif', ['pair' num2str(i)]);
    if ~exist(save_dir,'dir')
        mkdir(save_dir);
    end
    cp_showMatch(im1, im2, matchedPoints1, matchedPoints2, find(inliersIndex), 'matches.jpg', save_dir);
    image_fusion(im1, im2, H, save_dir);
    cleanedPoints1 = matchedPoints1(inliersIndex, :);
    cleanedPoints2 = matchedPoints2(inliersIndex, :);
    [cleanedPoints2,IA] = unique(cleanedPoints2,'rows');
    cleanedPoints1 = cleanedPoints1(IA,:);
    cleanedPoints=[cleanedPoints1 cleanedPoints2];
    cleanedPoints = double(cleanedPoints);
    Y_=H*[cleanedPoints(:,1:2)';ones(1,size(cleanedPoints,1))];
    Y_(1,:)=Y_(1,:)./Y_(3,:);
    Y_(2,:)=Y_(2,:)./Y_(3,:);
    E=sqrt(sum((Y_(1:2,:)-cleanedPoints(:,3:4)').^2));
    if length(E)<4
        rmse = 20;
    else
        rmse = sqrt(sum(E.^2)/size(E,2));
    end
    length(E)
    timeres = double([time rmse size(cleanedPoints,1)]);
    RES = [RES;timeres];

end

save RES_srif_tv.mat RES
