function WSSF_debug_probe(im1, im2, reportPath)
%WSSF_DEBUG_PROBE Probe WSSF pipeline and save intermediate counts.

if nargin < 3 || isempty(reportPath)
    reportPath = '.\wssf_debug_report.txt';
end

outDir = fileparts(reportPath);
if ~isempty(outDir) && exist(outDir, 'dir') == 0
    mkdir(outDir);
end

fid = fopen(reportPath, 'w');
if fid < 0
    error('Cannot open report: %s', reportPath);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'WSSF debug probe\n');
fprintf(fid, 'im1=%s\n', string(im1));
fprintf(fid, 'im2=%s\n', string(im2));

try
    image_3 = uint8(imread(im1));
    image_4 = uint8(imread(im2));
    fprintf(fid, 'raw size1=%s size2=%s\n', mat2str(size(image_3)), mat2str(size(image_4)));

    if size(image_3,3)==1
        image_3 = cat(3, image_3,image_3,image_3);
    end
    if size(image_4,3)==1
        image_4 = cat(3, image_4,image_4,image_4);
    end

    image_3 = adapthisteq(rgb2gray(image_3));
    image_4 = adapthisteq(rgb2gray(image_4));
    image_3 = cat(3, image_3,image_3,image_3);
    image_4 = cat(3, image_4,image_4,image_4);
    image_1 = im2double(image_3);
    image_2 = im2double(image_4);

    Path_Block=48;
    sigma_1=1.6;
    ratio=2^(1/3);
    ScaleValue = 1.6;
    nOctaves = 3;
    filter = 5;
    Scale ='YES';

    [nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1] = Create_Image_space(image_1,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2] = Create_Image_space(image_2,nOctaves,Scale, ScaleValue, ratio,sigma_1,filter);
    [Bolb_KeyPts_1,Corner_KeyPts_1,Bolb_gradient_1,Corner_gradient_1,Bolb_angle_1,Corner_angle_1] = WSSF_features(nonelinear_space_1,E_space_1,Max_space_1,Min_space_1,Phase_space_1,sigma_1,ratio,Scale,nOctaves);
    [Bolb_KeyPts_2,Corner_KeyPts_2,Bolb_gradient_2,Corner_gradient_2,Bolb_angle_2,Corner_angle_2] = WSSF_features(nonelinear_space_2,E_space_2,Max_space_2,Min_space_2,Phase_space_2,sigma_1,ratio,Scale,nOctaves);

    fprintf(fid, 'blob kpts: %d vs %d\n', size(Bolb_KeyPts_1,1), size(Bolb_KeyPts_2,1));
    fprintf(fid, 'corner kpts: %d vs %d\n', size(Corner_KeyPts_1,1), size(Corner_KeyPts_2,1));

    Bolb_descriptors_1 = GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Path_Block, ratio,sigma_1);
    Corner_descriptors_1 = GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Path_Block, ratio,sigma_1);
    Bolb_descriptors_2 = GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Path_Block, ratio,sigma_1);
    Corner_descriptors_2 = GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Path_Block, ratio,sigma_1);

    fprintf(fid, 'blob des: %s vs %s\n', mat2str(size(Bolb_descriptors_1.des)), mat2str(size(Bolb_descriptors_2.des)));
    fprintf(fid, 'corner des: %s vs %s\n', mat2str(size(Corner_descriptors_1.des)), mat2str(size(Corner_descriptors_2.des)));

    thresholds = [50 80 120 200];
    for t = thresholds
        try
            [pairsB, ~] = matchFeatures(Bolb_descriptors_1.des, Bolb_descriptors_2.des, 'MaxRatio', 1, 'MatchThreshold', t, 'Unique', true);
            fprintf(fid, 'blob matches threshold %d: %d\n', t, size(pairsB,1));
        catch ME1
            fprintf(fid, 'blob threshold %d error: %s\n', t, ME1.message);
        end
        try
            [pairsC, ~] = matchFeatures(Corner_descriptors_1.des, Corner_descriptors_2.des, 'MaxRatio', 1, 'MatchThreshold', t, 'Unique', true);
            fprintf(fid, 'corner matches threshold %d: %d\n', t, size(pairsC,1));
        catch ME2
            fprintf(fid, 'corner threshold %d error: %s\n', t, ME2.message);
        end
    end
catch ME
    fprintf(fid, 'probe error: %s\n', ME.message);
end
