%  ******* Code: Multi-Modal Remote Sensing Image Matching Based on Weighted Structure Saliency Feature
%  author: Created  in 2024/1/9.
%  note:��һ����������ʹ��GPU���ٻ�����������Ҫ5�������ҡ�
%  ********************************************************* 
close all;
beep off;
warning('off');
addpath(genpath('PSATF'));
addpath(genpath('Others'));
%% 1 Import and display reference and image to be registered
save_dir = fullfile(pwd, 'save_image_origin');
scalespace_opt_dir = fullfile(pwd, 'scalespace_opt');
scalespace_sar_dir = fullfile(pwd, 'scalespace_sar');
grad_opt_dir = fullfile(pwd, 'gradspace_opt');
grad_sar_dir = fullfile(pwd, 'gradspace_sar');
%file_image= 'D:\hand_craft_registration\SRIF-master\dataset\Optical-SAR';
file_image = 'D:\Edge_download\ht_eval_pair\pairs';
image_path_1 = fullfile(file_image, 'pair1_1.jpg');
image_path_2 = fullfile(file_image, 'pair1_2.jpg');
image_3 = imread(image_path_1);
image_4 = imread(image_path_2);

%tv_lambda = 30;
%tv_iterations = 100;
%image_3 = tv_denoise_image(image_3, tv_lambda, tv_iterations);
%image_4 = tv_denoise_image(image_4, tv_lambda, tv_iterations);

image1 = image_3; image2 = image_4;

if size(image_3,3)==1
    image_3 = cat(3, image_3,image_3,image_3);
end
if size(image_4,3)==1
    image_4 = cat(3, image_4,image_4,image_4);
end

image_3 = adapthisteq(rgb2gray(image_3));image_4 = adapthisteq(rgb2gray(image_4));
image_3 = cat(3, image_3,image_3,image_3);image_4 = cat(3, image_4,image_4,image_4);
image_1 = im2double(image_3);image_2 = im2double(image_4);



%% 2  Setting of initial parameters 
%Key parameters:
Path_Block=48;                   
sigma_1=1.6;   
ratio=2^(1/3);                     
ScaleValue = 1.6;
nOctaves = 3;
filter = 5;
Scale ='YES';


%% 3 Ӱ��ռ�
t1=clock;
disp('Start WSSF algorithm processing, please waiting...');
tic;
[nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1]=Create_Image_space(image_1,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter,scalespace_opt_dir);
[nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2]=Create_Image_space(image_2,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter,scalespace_sar_dir);
disp(['����Ӱ��߶ȿռ仨��ʱ�䣺',num2str(toc),' ��']);

%% 4 ������ȡ
tic;
[Bolb_KeyPts_1,Corner_KeyPts_1,Bolb_gradient_1,Corner_gradient_1,Bolb_angle_1,Corner_angle_1]  =  WSSF_features(nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1,sigma_1,ratio,Scale,nOctaves,grad_opt_dir);
[Bolb_KeyPts_2,Corner_KeyPts_2,Bolb_gradient_2,Corner_gradient_2,Bolb_angle_2,Corner_angle_2]  =  WSSF_features(nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2,sigma_1,ratio,Scale,nOctaves,grad_sar_dir);
disp(['��������ȡ����ʱ��:  ',num2str(toc),' ��']);
%% 5 GLOH Descriptor 
tic;

Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio,sigma_1);
Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio,sigma_1);
Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio,sigma_1);
Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio,sigma_1);

disp(['���������ӻ���ʱ��:  ',num2str(toc),' ��']); 


%% 6 Matching by FSC
[indexPairs,~]= matchFeatures(Bolb_descriptors_1.des,Bolb_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true ); 
[matchedPoints_1_1,matchedPoints_1_2] = BackProjection(Bolb_descriptors_1.locs(indexPairs(:, 1), :),Bolb_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue); 
[indexPairs,~]= matchFeatures(Corner_descriptors_1.des,Corner_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true ); 
[matchedPoints_2_1,matchedPoints_2_2] = BackProjection(Corner_descriptors_1.locs(indexPairs(:, 1), :),Corner_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue); 

matchedPoints_1 = [matchedPoints_1_1;matchedPoints_2_1];
matchedPoints_2 = [matchedPoints_1_2;matchedPoints_2_2];
allNCM = size(matchedPoints_1,1);


[H1,rmse]=FSC(matchedPoints_1,matchedPoints_2,'affine',3);
Y_=H1*[matchedPoints_1(:,[1,2])';ones(1,size(matchedPoints_1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints_2(:,[1,2])').^2));
inliersIndex=E < 3;
clearedPoints1 = matchedPoints_1(inliersIndex, :);
clearedPoints2 = matchedPoints_2(inliersIndex, :);

[clearedPoints2,IA]=unique(clearedPoints2,'rows');
clearedPoints1=clearedPoints1(IA,:);
 


disp(['����ƥ�仨��ʱ��:  ',num2str(toc),' ��']); 
tic; 

cp_showMatch(image1, image2, clearedPoints1,clearedPoints2,[],'matches.jpg',save_dir);
RCM = size(clearedPoints1,1)/allNCM;
image_fusion(image2,image1,double(H1),save_dir);


fprintf('\n');
disp(['��ȷƥ��������',num2str(size(clearedPoints1,1))]);
disp(['ƥ��ĳɹ���RCM��',num2str(RCM*100),'%']);
disp(['RMSE of Matching results: ',num2str(rmse),'  ����']); 
t2=clock;
disp(['����׼����ʱ��:  ',num2str(etime(t2,t1)),' ��']); 
tic;
