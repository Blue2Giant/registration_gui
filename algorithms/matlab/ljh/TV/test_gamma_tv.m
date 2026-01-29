function test_gamma_tv()
    clc; clear; close all;
    N = 256;
    u_clean = zeros(N, N);
    u_clean(64:192, 64:192) = 200;
    u_clean = double(u_clean);

    L = 4;
    try
        noise_gamma = gamrnd(L, 1/L, size(u_clean));
    catch
        error('需要 Statistics and Machine Learning Toolbox 才能使用 gamrnd 函数');
    end
    f_noisy = u_clean .* noise_gamma;

    lambda = 0.1;
    iter = 300;
    fprintf('正在进行 TV 迭代去噪 (L=%d)...\n', L);
    u_restored = total_variation(f_noisy, iter, lambda);

    figure('Name', 'Gamma 噪声下的 TV 测试', 'Color', 'w', 'Position', [100, 100, 1200, 400]);
    subplot(1, 3, 1);
    imshow(u_clean, [0, 255]);
    title('原始图像 (Ground Truth)');

    subplot(1, 3, 2);
    imshow(f_noisy, [0, 255]);
    title(['乘性 Gamma 噪声 (L=', num2str(L), ')']);

    subplot(1, 3, 3);
    imshow(u_restored, [0, 255]);
    title(['TV 恢复结果 (\lambda=', num2str(lambda), ')']);

    figure('Name', '强度切片对比', 'Color', 'w');
    mid_row = N/2;
    plot(u_clean(mid_row, :), 'k--', 'LineWidth', 1.5); hold on;
    plot(f_noisy(mid_row, :), 'g-', 'LineWidth', 0.5);
    plot(u_restored(mid_row, :), 'r-', 'LineWidth', 2);
    legend('原始', '含噪 Gamma', 'TV 恢复');
    title('第 128 行像素强度切片');
    grid on;
    axis([0 N -50 300]);
end
