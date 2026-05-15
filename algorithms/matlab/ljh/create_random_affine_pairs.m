close all;
beep off;
warning('off');

input_dir = 'D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_origin_match';
output_dir = fullfile(pwd, 'pairs_random_affine_match');
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end
pair_list = dir(fullfile(input_dir, 'pair*_1.*'));

rng('shuffle');

for k = 1:numel(pair_list)
    name_1 = pair_list(k).name;
    base_name = regexprep(name_1, '_1\.[^.]+$', '');
    name_2 = [base_name '_2.jpg'];
    path_2 = fullfile(input_dir, name_2);
    out_path_1 = fullfile(output_dir, name_1);
    out_path_2 = fullfile(output_dir, name_2);
    gt_path = fullfile(output_dir, [base_name '.txt']);
    if ~exist(path_2, 'file')
        continue
    end

    image_2 = imread(path_2);
    [h, w, ~] = size(image_2);
    cx = (w + 1) / 2;
    cy = (h + 1) / 2;

    theta = (-30 + 60 * rand) * pi / 180;
    s = 0.8 + 0.4 * rand;

    T_shift1 = [1 0 0; 0 1 0; -cx -cy 1];
    T_rs = [s*cos(theta) s*sin(theta) 0; -s*sin(theta) s*cos(theta) 0; 0 0 1];
    T_shift2 = [1 0 0; 0 1 0; cx cy 1];
    T = T_shift1 * T_rs * T_shift2;

    tform = affine2d(T);
    ref = imref2d([h w]);
    image_2_warp = imwarp(image_2, tform, 'OutputView', ref, 'FillValues', 0);
    copyfile(fullfile(input_dir, name_1), out_path_1);
    imwrite(image_2_warp, out_path_2);

    gt = T(1:2, 1:3);
    fid = fopen(gt_path, 'w');
    fprintf(fid, '%.15f %.15f %.15f\n', gt(1,1), gt(1,2), gt(1,3));
    fprintf(fid, '%.15f %.15f %.15f\n', gt(2,1), gt(2,2), gt(2,3));
    fclose(fid);
end
