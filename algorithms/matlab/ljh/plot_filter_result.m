% 提取的滤波代码位置（含封装调用）：
% - WSSF_demo_tv_logtv.m: L23-L35（SAR: Log-TV 去噪 + 灰度化/类型变换/归一化）
% - WSSF_demo_tv_logtv.m: L44-L47（SAR/光学: adapthisteq 局部增强/CLAHE）
% - WSSF_demo_tv_logtv.m: L59-L60（SAR/光学: Create_Image_space 多尺度空间构建）
% - TV/log_total_variation.m: L1-L117（Log-TV 去噪实现，IterMax=50，lambda=1.0）
% - Create_Image_space.m: L12-L70（多尺度：phasecong(4,6)、edgesDetect、Gaussian(imfilter)、imresize(bilinear)、SAF）
%
% 依赖变量与维度假设：
% - SAR 图像：二维灰度或三通道图像；显示时按强度 I 做 dB = 10*log10(I+eps)。
% - 光学图像：二维灰度或三通道 RGB；原始显示按输入维度自动选择 imshow。
% - Create_Image_space 输出的 Nonelinear_Scalespace 为 cell(1,nOctaves)，每个元素为二维 double 强度图（不同尺度分辨率可能不同）。
%
% 工具箱/第三方依赖检查：
% - Image Processing Toolbox：rgb2gray, adapthisteq, mat2gray, im2double, imresize, imfilter, fspecial
% - 第三方/工程内：TV/log_total_variation.m, Create_Image_space.m, Others/phasecong.m, edges-master/edgesDetect.m, PSATF/SAF.m
% - edges-master 模型文件：edges-master/models/forest/modelBsds.mat

plot_filter_result_main();

function plot_filter_result_main()
close all;

outDir = pwd;
repoRoot = fileparts(mfilename('fullpath'));
dpi = 1200;

addpath(genpath(fullfile(repoRoot, 'PSATF')));
addpath(genpath(fullfile(repoRoot, 'Others')));
addpath(genpath(fullfile(repoRoot, 'TV')));
addpath(genpath(fullfile(repoRoot, 'edges-master')));

assert(exist('log_total_variation', 'file') == 2, '缺少函数 log_total_variation（TV/log_total_variation.m）。');
assert(exist('Create_Image_space', 'file') == 2, '缺少函数 Create_Image_space（Create_Image_space.m）。');
assert(exist('phasecong', 'file') == 2, '缺少函数 phasecong（Others/phasecong.m）。');
assert(exist('edgesDetect', 'file') == 2, '缺少函数 edgesDetect（edges-master）。');
assert(exist('SAF', 'file') == 2, '缺少函数 SAF（PSATF/SAF.m）。');
assert(exist('adapthisteq', 'file') == 2, '缺少函数 adapthisteq（需要 Image Processing Toolbox）。');
assert(exist('rgb2gray', 'file') == 2, '缺少函数 rgb2gray（需要 Image Processing Toolbox）。');
assert(exist('mat2gray', 'file') == 2, '缺少函数 mat2gray（需要 Image Processing Toolbox）。');
assert(exist('im2double', 'file') == 2, '缺少函数 im2double（需要 Image Processing Toolbox）。');
assert(exist('imresize', 'file') == 2, '缺少函数 imresize（需要 Image Processing Toolbox）。');
assert(exist('imfilter', 'file') == 2, '缺少函数 imfilter（需要 Image Processing Toolbox）。');
assert(exist('fspecial', 'file') == 2, '缺少函数 fspecial（需要 Image Processing Toolbox）。');

modelFile = fullfile(repoRoot, 'edges-master', 'models', 'forest', 'modelBsds.mat');
assert(exist(modelFile, 'file') == 2, '缺少模型文件：%s', modelFile);

overrideSar = fullfile(repoRoot, 'ht_eval_for_own_origin', 'pair1_2.jpg');
overrideOpt = fullfile(repoRoot, 'ht_eval_for_own_origin', 'pair1_1.jpg');
if exist(overrideSar, 'file') == 2 && exist(overrideOpt, 'file') == 2
    sarPath = overrideSar;
    optPath = overrideOpt;
else
    [sarPath, optPath] = resolve_input_images(repoRoot, outDir);
end
sarPath = char(sarPath);
optPath = char(optPath);
sar_raw = imread(sarPath);
opt_raw = imread(optPath);

tv_lambda = 1.0;
tv_iterations = 50;

sigma_1 = 1.6;
ratio = 2^(1/3);
ScaleValue = 1.6;
nOctaves = 3;
filter = 5;
Scale = 'YES';

[opt_image_1, opt_raw_for_show] = preprocess_optical_like_demo(opt_raw);
[sar_image_2, sar_raw_db] = preprocess_sar_like_demo(sar_raw, tv_iterations, tv_lambda);

origDir = pwd;
cdCleanup = onCleanup(@() cd(origDir));
cd(repoRoot);
[opt_nonelinear_space, opt_E_space, opt_Max_space, opt_Min_space, opt_Phase_space] = Create_Image_space(opt_image_1, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);
[sar_nonelinear_space, sar_E_space, sar_Max_space, sar_Min_space, sar_Phase_space] = Create_Image_space(sar_image_2, nOctaves, Scale, ScaleValue, ratio, sigma_1, filter);

nScales = min(numel(opt_nonelinear_space), numel(sar_nonelinear_space));
clim = calc_clim(sar_raw_db);

timestamp = datestr(now, 'yyyymmdd_HHMMSS');

for s = 1:nScales
    fSar = figure('Color', 'w');
    tlSar = tiledlayout(1, 5, 'Padding', 'compact', 'TileSpacing', 'compact');
    show_gray(nexttile(tlSar), sar_nonelinear_space{s}); title(sprintf('Nonelinear %d', s));
    show_gray(nexttile(tlSar), sar_E_space{s}); title(sprintf('E %d', s));
    show_gray(nexttile(tlSar), sar_Max_space{s}); title(sprintf('Max %d', s));
    show_gray(nexttile(tlSar), sar_Min_space{s}); title(sprintf('Min %d', s));
    show_gray(nexttile(tlSar), sar_Phase_space{s}); title(sprintf('Phase %d', s));
    sgtitle(tlSar, sprintf(['SAR 尺度空间(Create_Image_space) | Log-TV(\\lambda=%.1f, iter=%d) + CLAHE | ' ...
        'nOctaves=%d, ScaleValue=%.1f, ratio=%.5f, \\sigma_1=%.1f, filter=%d | %s | 尺度 %d'], ...
        double(tv_lambda), double(tv_iterations), double(nOctaves), double(ScaleValue), double(ratio), double(sigma_1), double(filter), char(basename(sarPath)), double(s)));
    sarOut = fullfile(outDir, sprintf('filter_result_%s_SAR_Create_Image_space_scale%02d.png', timestamp, s));
    print(fSar, sarOut, '-dpng', sprintf('-r%d', dpi));
    fprintf('Saved: %s\n', sarOut);

    fOpt = figure('Color', 'w');
    tlOpt = tiledlayout(1, 5, 'Padding', 'compact', 'TileSpacing', 'compact');
    show_gray(nexttile(tlOpt), opt_nonelinear_space{s}); title(sprintf('Nonelinear %d', s));
    show_gray(nexttile(tlOpt), opt_E_space{s}); title(sprintf('E %d', s));
    show_gray(nexttile(tlOpt), opt_Max_space{s}); title(sprintf('Max %d', s));
    show_gray(nexttile(tlOpt), opt_Min_space{s}); title(sprintf('Min %d', s));
    show_gray(nexttile(tlOpt), opt_Phase_space{s}); title(sprintf('Phase %d', s));
    sgtitle(tlOpt, sprintf(['光学 尺度空间(Create_Image_space) | CLAHE(adapthisteq默认) | ' ...
        'nOctaves=%d, ScaleValue=%.1f, ratio=%.5f, \\sigma_1=%.1f, filter=%d | %s | 尺度 %d'], ...
        double(nOctaves), double(ScaleValue), double(ratio), double(sigma_1), double(filter), char(basename(optPath)), double(s)));
    optOut = fullfile(outDir, sprintf('filter_result_%s_OPT_Create_Image_space_scale%02d.png', timestamp, s));
    print(fOpt, optOut, '-dpng', sprintf('-r%d', dpi));
    fprintf('Saved: %s\n', optOut);

    fSarDb = figure('Color', 'w');
    tlSarDb = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile(tlSarDb);
    imagesc(sar_raw_db);
    axis image off;
    colormap(gca, 'gray');
    caxis(clim);
    colorbar;
    title('原始 SAR (dB)');
    nexttile(tlSarDb);
    sar_in = to_gray(sar_image_2);
    imshow(mat2gray(sar_in));
    title('输入 SAR（Log-TV+CLAHE 后）');
    sgtitle(tlSarDb, sprintf('SAR 输入检查 | %s | 尺度 %d', char(basename(sarPath)), double(s)));
    sarInOut = fullfile(outDir, sprintf('filter_result_%s_SAR_inputcheck_scale%02d.png', timestamp, s));
    print(fSarDb, sarInOut, '-dpng', sprintf('-r%d', dpi));
    fprintf('Saved: %s\n', sarInOut);
end

fprintf('Saved PNG(s) to: %s\n', outDir);
end

function [sarPath, optPath] = resolve_input_images(repoRoot, outDir)
sarPath = '';
optPath = '';

defaultCandidates = {};
defaultCandidates{end+1} = fullfile(repoRoot, 'rr.bmp');
defaultCandidates{end+1} = fullfile(repoRoot, 'sar.tif');
defaultCandidates{end+1} = fullfile(repoRoot, 'sar.png');
defaultCandidates{end+1} = fullfile(repoRoot, 'opt.tif');
defaultCandidates{end+1} = fullfile(repoRoot, 'opt.png');

defaultCandidates{end+1} = fullfile(outDir, 'rr.bmp');
defaultCandidates{end+1} = fullfile(outDir, 'sar.tif');
defaultCandidates{end+1} = fullfile(outDir, 'sar.png');
defaultCandidates{end+1} = fullfile(outDir, 'sar.jpg');
defaultCandidates{end+1} = fullfile(outDir, 'sar.jpeg');
defaultCandidates{end+1} = fullfile(outDir, 'sar.bmp');
defaultCandidates{end+1} = fullfile(outDir, 'opt.tif');
defaultCandidates{end+1} = fullfile(outDir, 'opt.png');
defaultCandidates{end+1} = fullfile(outDir, 'opt.jpg');
defaultCandidates{end+1} = fullfile(outDir, 'opt.jpeg');
defaultCandidates{end+1} = fullfile(outDir, 'opt.bmp');

existing = defaultCandidates(cellfun(@(p) exist(p, 'file') == 2, defaultCandidates));
if numel(existing) >= 2
    sarPath = existing{1};
    optPath = existing{2};
    return
end

assert(exist('cameraman.tif', 'file') == 2, '未找到默认 SAR 示例图 cameraman.tif。');
assert(exist('peppers.png', 'file') == 2, '未找到默认光学示例图 peppers.png。');
sarPath = 'cameraman.tif';
optPath = 'peppers.png';
end

function [image_1, opt_raw_for_show] = preprocess_optical_like_demo(opt_raw)
opt_raw_for_show = opt_raw;
image_3 = uint8(opt_raw);
if size(image_3, 3) > 1
    image_3 = rgb2gray(image_3);
end
image_3 = double(image_3);
image_3 = uint8(255 * mat2gray(image_3));
if size(image_3, 3) == 1
    image_3 = cat(3, image_3, image_3, image_3);
end
image_3 = adapthisteq(mat2gray(image_3(:, :, 1)));
image_3 = cat(3, image_3, image_3, image_3);
image_1 = im2double(image_3);
end

function [image_2, sar_raw_db] = preprocess_sar_like_demo(sar_raw, tv_iterations, tv_lambda)
image_4 = uint8(sar_raw);
if size(image_4, 3) > 1
    image_4 = rgb2gray(image_4);
end
sar_raw_double = double(image_4);
sar_raw_db = 10 * log10(max(sar_raw_double, eps));

image_4 = log_total_variation(sar_raw_double, tv_iterations, tv_lambda);
image_4 = uint8(255 * mat2gray(image_4));
if size(image_4, 3) == 1
    image_4 = cat(3, image_4, image_4, image_4);
end
image_4 = adapthisteq(mat2gray(image_4(:, :, 1)));
image_4 = cat(3, image_4, image_4, image_4);
image_2 = im2double(image_4);
end

function clim = calc_clim(dbImg)
mask = isfinite(dbImg);
if any(mask(:))
    clim = [min(dbImg(mask), [], 'all') max(dbImg(mask), [], 'all')];
else
    clim = [-30 0];
end
if ~isfinite(clim(1)) || ~isfinite(clim(2)) || clim(1) == clim(2)
    clim = [-30 0];
end
end

function name = basename(p)
[~, n, e] = fileparts(p);
name = [char(n) char(e)];
end

function g = to_gray(x)
if ndims(x) == 2
    g = x;
    return
end
if ndims(x) == 3 && size(x, 3) >= 1
    g = x(:, :, 1);
    return
end
g = x;
end

function show_gray(ax, x)
g = to_gray(x);
g = double(g);
imagesc(ax, g);
axis(ax, 'image');
axis(ax, 'off');
colormap(ax, 'gray');
caxis(ax, robust_clim(g));
end

function cl = robust_clim(g)
v = g(isfinite(g));
if isempty(v)
    cl = [0 1];
    return
end
v = sort(v(:));
n = numel(v);
i1 = max(1, round(0.01 * n));
i2 = max(i1, round(0.99 * n));
cl = [v(i1) v(i2)];
if ~isfinite(cl(1)) || ~isfinite(cl(2)) || cl(1) == cl(2)
    cl = [v(1) v(end)];
end
end
