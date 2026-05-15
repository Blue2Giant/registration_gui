close all;
beep off;
warning('off');
addpath(genpath('PSATF'));
addpath(genpath('Others'));

input_dir = 'D:\hand_craft_registration\SRIF-master\dataset\HT';
save_dir = fullfile(pwd, 'save_image_ht_pair');
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

pair_list = dir(fullfile(input_dir, 'pair*_1.*'));
total_time = 0;
done_count = 0;

for k = 1:numel(pair_list)
    t_start = tic;
    name_1 = pair_list(k).name;
    base_name = regexprep(name_1, '_1\.[^.]+$', '');
    pair2_list = dir(fullfile(input_dir, [base_name '_2.*']));
    if isempty(pair2_list)
        continue
    end
    name_2 = pair2_list(1).name;
    image_3 = imread(fullfile(input_dir, name_1));
    image_4 = imread(fullfile(input_dir, name_2));

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

    [nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1]=Create_Image_space(image_1,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2]=Create_Image_space(image_2,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);

    [Bolb_KeyPts_1,Corner_KeyPts_1,Bolb_gradient_1,Corner_gradient_1,Bolb_angle_1,Corner_angle_1]  =  WSSF_features(nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1,sigma_1,ratio,Scale,nOctaves);
    [Bolb_KeyPts_2,Corner_KeyPts_2,Bolb_gradient_2,Corner_gradient_2,Bolb_angle_2,Corner_angle_2]  =  WSSF_features(nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2,sigma_1,ratio,Scale,nOctaves);

    Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio,sigma_1);
    Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio,sigma_1);
    Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio,sigma_1);
    Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio,sigma_1);

    [indexPairs,~]= matchFeatures(Bolb_descriptors_1.des,Bolb_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
    [matchedPoints_1_1,matchedPoints_1_2] = BackProjection(Bolb_descriptors_1.locs(indexPairs(:, 1), :),Bolb_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);
    [indexPairs,~]= matchFeatures(Corner_descriptors_1.des,Corner_descriptors_2.des,'MaxRatio',1,'MatchThreshold', 50,'Unique',true );
    [matchedPoints_2_1,matchedPoints_2_2] = BackProjection(Corner_descriptors_1.locs(indexPairs(:, 1), :),Corner_descriptors_2.locs(indexPairs(:, 2), :),ScaleValue);

    matchedPoints_1 = [matchedPoints_1_1;matchedPoints_2_1];
    matchedPoints_2 = [matchedPoints_1_2;matchedPoints_2_2];

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
    elapsed = toc(t_start);
    done_count = done_count + 1;
    total_time = total_time + elapsed;
    avg_time = total_time / done_count;
    disp(['完成: ' base_name '  用时: ' num2str(elapsed, '%.2f') 's  平均: ' num2str(avg_time, '%.2f') 's']);
end
