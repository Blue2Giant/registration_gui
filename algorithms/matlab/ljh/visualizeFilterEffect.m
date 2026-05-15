function visualizeFilterEffect(SAR1, SAR2, optImage1, optImage2, filteredSAR1, filteredSAR2, savePath)
if nargin == 0
    runVisualizeFilterEffectTest();
    return
end
if nargin < 7
    savePath = '';
end

SAR1 = toDouble2D(SAR1);
SAR2 = toDouble2D(SAR2);
filteredSAR1 = toDouble2D(filteredSAR1);
filteredSAR2 = toDouble2D(filteredSAR2);
optImage1 = toDouble2D(optImage1);
optImage2 = toDouble2D(optImage2);

range1 = sharedRange(SAR1, filteredSAR1);
range2 = sharedRange(SAR2, filteredSAR2);

fig1 = figure('Color', 'w', 'Position', [100, 100, 1400, 360]);
t1 = tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile;
imshow(SAR1, range1);
colormap(ax1, gray(256));
axis image off;
title('SAR1');

ax2 = nexttile;
imshow(optImage1, []);
colormap(ax2, gray(256));
axis image off;
title('Optical1');

ax3 = nexttile;
imshow(SAR2, range2);
colormap(ax3, gray(256));
axis image off;
title('SAR2');

ax4 = nexttile;
imshow(optImage2, []);
colormap(ax4, gray(256));
axis image off;
title('Optical2');

diff1 = filteredSAR1 - SAR1;
diff2 = filteredSAR2 - SAR2;
diffRange = max([abs(diff1(:)); abs(diff2(:)); eps]);

stats1 = calcStats(SAR1, filteredSAR1, diff1);
stats2 = calcStats(SAR2, filteredSAR2, diff2);

fig2 = figure('Color', 'w', 'Position', [100, 100, 1400, 720]);
t2 = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax5 = nexttile;
imshow(SAR1, range1);
colormap(ax5, gray(256));
axis image off;
title('SAR1 Original');

ax6 = nexttile;
imshow(SAR2, range2);
colormap(ax6, gray(256));
axis image off;
title('SAR2 Original');

ax7 = nexttile;
imagesc(diff1);
colormap(ax7, parula(256));
axis image off;
caxis([-diffRange diffRange]);
colorbar;
title('Diff SAR1');
text(0.02, 0.98, stats1, 'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'w', 'FontSize', 11);

ax8 = nexttile;
imshow(filteredSAR1, range1);
colormap(ax8, gray(256));
axis image off;
title('SAR1 Filtered');

ax9 = nexttile;
imshow(filteredSAR2, range2);
colormap(ax9, gray(256));
axis image off;
title('SAR2 Filtered');

ax10 = nexttile;
imagesc(diff2);
colormap(ax10, parula(256));
axis image off;
caxis([-diffRange diffRange]);
colorbar;
title('Diff SAR2');
text(0.02, 0.98, stats2, 'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'w', 'FontSize', 11);

if ~isempty(savePath)
    [saveDir, baseName, ext] = fileparts(savePath);
    if isempty(saveDir)
        saveDir = pwd;
    end
    if isfolder(savePath)
        saveDir = savePath;
        baseName = 'filter_effect';
    end
    if isempty(baseName)
        baseName = 'filter_effect';
    end
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    outBase = fullfile(saveDir, baseName);
    exportgraphics(fig1, [outBase '_input.png'], 'Resolution', 300);
    exportgraphics(fig1, [outBase '_input.pdf'], 'ContentType', 'vector');
    exportgraphics(fig2, [outBase '_filter.png'], 'Resolution', 300);
    exportgraphics(fig2, [outBase '_filter.pdf'], 'ContentType', 'vector');
end
end

function img = toDouble2D(img)
if isempty(img)
    img = zeros(0, 0);
    return
end
if ndims(img) == 3
    img = rgb2gray(img);
end
img = double(img);
end

function range = sharedRange(a, b)
lo = min([a(:); b(:)]);
hi = max([a(:); b(:)]);
if lo == hi
    hi = lo + 1;
end
range = [lo hi];
end

function statsText = calcStats(orig, filt, diffImg)
meanDiff = mean(diffImg(:));
stdDiff = std(diffImg(:));
meanFilt = mean(filt(:));
varFilt = var(filt(:));
enl = (meanFilt * meanFilt) / (varFilt + eps);
epi = edgePreserveIndex(orig, filt);
statsText = sprintf('mean=%.4f  std=%.4f  ENL=%.2f  EPI=%.3f', meanDiff, stdDiff, enl, epi);
end

function epi = edgePreserveIndex(orig, filt)
orig = double(orig);
filt = double(filt);
kx = [1 0 -1; 2 0 -2; 1 0 -1] / 4;
ky = kx';
gx1 = conv2(orig, kx, 'same');
gy1 = conv2(orig, ky, 'same');
gx2 = conv2(filt, kx, 'same');
gy2 = conv2(filt, ky, 'same');
mag1 = sqrt(gx1.^2 + gy1.^2);
mag2 = sqrt(gx2.^2 + gy2.^2);
epi = sum(mag2(:)) / (sum(mag1(:)) + eps);
end

function runVisualizeFilterEffectTest()
rootDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(rootDir, 'ht_eval_for_own_affine');
sar1Path = fullfile(dataDir, 'pair1_1.jpg');
sar2Path = fullfile(dataDir, 'pair1_2.jpg');
sar1 = imread(sar1Path);
sar2 = imread(sar2Path);
if ndims(sar1) == 3
    sar1 = rgb2gray(sar1);
end
if ndims(sar2) == 3
    sar2 = rgb2gray(sar2);
end
sar1 = double(sar1);
sar2 = double(sar2);
opt1 = sar1;
opt2 = sar2;
tv_lambda = 1.0;
tv_iterations = 50;
filtered1 = log_total_variation(sar1, tv_iterations, tv_lambda);
filtered2 = log_total_variation(sar2, tv_iterations, tv_lambda);
saveDir = fullfile(pwd, 'filter_vis');
saveBase = fullfile(saveDir, 'filter_effect');
visualizeFilterEffect(sar1, sar2, opt1, opt2, filtered1, filtered2, saveBase);
assert(exist([saveBase '_input.png'], 'file') == 2);
assert(exist([saveBase '_input.pdf'], 'file') == 2);
assert(exist([saveBase '_filter.png'], 'file') == 2);
assert(exist([saveBase '_filter.pdf'], 'file') == 2);
end
