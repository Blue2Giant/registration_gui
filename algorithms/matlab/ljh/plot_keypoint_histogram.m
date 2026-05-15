%% Reproduce stacked bar chart similar to the provided figure
clear; clc; close all;

%% ===================== User parameters =====================
legendFontSize = 13;   % 图例字体大小
axisFontSize   = 14;   % 坐标轴字体大小
titleFontSize  = 16;   % 标题字体大小
labelFontSize  = 14;   % 坐标轴标签字体大小
textFontSize   = 14;   % 柱顶百分比文字大小

%% ===================== Data =====================
algorithms = {'SIFT','RIFT','DOG+NN','R2D2','CMMNet','MIFNeT','MINIMA','Ours'};

% 原始统计量
matches         = [1,   62,   3,   1.7,  16,10.81,6.45,32];
correspondences = [128, 256, 126,  80, 140,135,369,138];
keypoints       = [1024,1024,1024,1024,1024,1024,1024,1024];

% 柱顶百分比标注
topText = {'0.0%','1.4%','0.3%','0.2%','1.6%','8.52%','1.75%','3.0%'};

%% ===================== Convert to stacked parts =====================
% 为了保证总高度恒等于 keypoints，需要拆成互斥的三部分
matches_part = matches;
corr_only    = correspondences - matches;
kpt_only     = keypoints - correspondences;

% 安全检查，避免出现负数
if any(corr_only < 0)
    error('存在 correspondences < matches 的情况，请检查数据。');
end
if any(kpt_only < 0)
    error('存在 keypoints < correspondences 的情况，请检查数据。');
end

% 用这三个“部分”来堆叠
Y = [matches_part(:), corr_only(:), kpt_only(:)];

%% ===================== Colors =====================
c_matches = [243, 105, 140] / 255;   % 粉色
c_corr    = [100, 142, 219] / 255;   % 蓝色
c_kpts    = [241, 199,  96] / 255;   % 黄色

%% ===================== Plot =====================
fig = figure('Color','w','Position',[100 100 980 620]);
ax = axes(fig);
hold(ax, 'on');

b = bar(ax, Y, 'stacked', 'BarWidth', 0.8, 'LineWidth', 1.0);

b(1).FaceColor = c_matches;
b(2).FaceColor = c_corr;
b(3).FaceColor = c_kpts;

for i = 1:numel(b)
    b(i).EdgeColor = [0.92 0.92 0.92];
end

%% ===================== Axes style =====================
set(ax, ...
    'YScale', 'log', ...
    'FontSize', axisFontSize, ...
    'LineWidth', 1.0, ...
    'Box', 'off', ...
    'Layer', 'bottom', ...
    'YMinorTick', 'off');

yticks(2.^(0:12));
yticklabels(arrayfun(@(k) sprintf('2^{%d}', k), 0:12, 'UniformOutput', false));

ylim([1 2^12.3]);
xlim([0.3, numel(algorithms)+0.7]);

xticks(1:numel(algorithms));
xticklabels(algorithms);

grid(ax, 'on');
ax.GridColor = [0.82 0.82 0.82];
ax.GridAlpha = 0.8;
ax.MinorGridAlpha = 0.2;

ylabel('特征点数量 (log2 尺度)', 'FontSize', labelFontSize);

%% ===================== Legend =====================
lgd = legend({'matches', 'correspondences', 'keypoints'}, ...
    'FontSize', legendFontSize);
lgd.Box = 'on';
lgd.Units = 'normalized';

% [left bottom width height]
lgd.Position = [0.73, 0.84, 0.20, 0.11];

%% ===================== Top percentage text =====================
% 总高度现在固定就是 keypoints
totals = keypoints;

for i = 1:numel(algorithms)
    text(i, totals(i)*1.08, topText{i}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', textFontSize, ...
        'Color', [0.2 0.2 0.2], ...
        'FontWeight', 'normal');
end

%% ===================== Save =====================
exportgraphics(fig, 'matching_performance_reproduced.png', 'Resolution', 1200);