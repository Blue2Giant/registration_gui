function test_gamma_log_tv()
    clc; clear; close all;

    %% 1. 生成原始图像 (黑色背景，白色色块)
    N = 256;
    u_clean = zeros(N, N);
    % 白色方块，强度设为 200
    u_clean(64:192, 64:192) = 200; 
    u_clean = double(u_clean); 

    %% 2. 施加乘性 Gamma 噪声 (Simulating SAR Speckle)
    % Gamma 分布参数：
    % L (Looks): 视数。L 越小，噪声越强（颗粒感越重）；L 越大，图像越平滑。
    % 典型的 SAR 图像 L 在 1 到 10 之间。
    L = 4; 
    
    % 生成均值为 1，方差为 1/L 的 Gamma 随机数
    % 注意：如果没有统计工具箱，gamrnd 可能无法使用，可以用 sum(exprnd) 替代
    try
        noise_gamma = gamrnd(L, 1/L, size(u_clean));
    catch
        error('需要 Statistics and Machine Learning Toolbox 才能使用 gamrnd 函数');
    end
    
    % 施加乘性噪声 f = u * n
    f_noisy = u_clean .* noise_gamma;

    %% 3. 使用 Log-TV 算法去噪
    % 参数设置
    % lambda: 
    %   对于 Log-TV，因为在对数域操作，数值范围变小了（从 0-255 变成 0-5.5）。
    %   而且噪声幅度相对固定（取决于 L）。
    %   建议 lambda 设置在 5 到 30 之间尝试。
    lambda = 0.1;   
    iter = 300;    % 迭代次数，适当增加以保证收敛
    
    fprintf('正在进行 Log-TV 迭代去噪 (L=%d)...\n', L);
    u_restored = log_total_variation(f_noisy, iter, lambda);

    %% 4. 结果可视化与评估
    figure('Name', 'Gamma 噪声下的 Log-TV 测试', 'Color', 'w', 'Position', [100, 100, 1200, 400]);
    
    % 原始图
    subplot(1, 3, 1); 
    imshow(u_clean, [0, 255]); 
    title('原始图像 (Ground Truth)');
    
    % 噪声图
    subplot(1, 3, 2); 
    imshow(f_noisy, [0, 255]); 
    title(['乘性 Gamma 噪声 (L=', num2str(L), ')']);
    %xlabel('特点：亮部噪声大，暗部噪声小');
    
    % 恢复图
    subplot(1, 3, 3); 
    imshow(u_restored, [0, 255]); 
    title(['Log-TV 恢复结果 (\lambda=', num2str(lambda), ')']);
    %xlabel('特点：边缘锐利，内部平滑');
    
    % (可选) 绘制一行切片来看看边缘保持情况
    figure('Name', '强度切片对比', 'Color', 'w');
    mid_row = N/2;
    plot(u_clean(mid_row, :), 'k--', 'LineWidth', 1.5); hold on;
    plot(f_noisy(mid_row, :), 'g-', 'LineWidth', 0.5);
    plot(u_restored(mid_row, :), 'r-', 'LineWidth', 2);
    legend('原始', '含噪 Gamma', 'Log-TV 恢复');
    title('第 128 行像素强度切片');
    grid on;
    axis([0 N -50 300]);
end

%% ==========================================================
%%  Log-TV 算法实现 (针对 Gamma 噪声优化版)
%% ==========================================================
function u_result = log_total_variation(u0, IterMax, lambda)
    % 1. 预处理：转换到对数域
    % Gamma 噪声通常不为负，但可能有0。加epsilon很关键。
    epsilon_log = 1e-3; 
    u0 = double(u0);
    
    % 变换：f_log 包含了 (log u + log n)
    % log n 近似服从高斯分布(虽然略微左偏)，满足 TV 的 L2 保真项假设
    f_log = log(u0 + epsilon_log); 

    %% 初始化
    u = f_log; 
    [M, N] = size(u);  
    h = 1;

    %% 迭代 (Gauss-Seidel 风格)
    for Iter = 1:IterMax    
        
        for i = 2:M-1       
            for j = 2:N-1
                % --- 计算 TV 部分的非线性扩散系数 ---
                % 东
                ux = (u(i+1,j)-u(i,j))/h;
                uy = (u(i,j+1)-u(i,j-1))/2*h;
                Grad = abs(ux) + abs(uy);
                if Grad > 1e-8, co1 = 1./Grad; else, co1 = 0; end % 也可以设为大数，这里0防止除零震荡
                
                % 西
                ux = (u(i,j)-u(i-1,j))/h;
                uy = (u(i-1,j+1)-u(i-1,j-1))/2*h;
                Grad = abs(ux) + abs(uy);
                if Grad > 1e-8, co2 = 1./Grad; else, co2 = 0; end
                
                % 南
                ux = (u(i+1,j)-u(i-1,j))/2*h;
                uy = (u(i,j+1)-u(i,j))/h;
                Grad = abs(ux) + abs(uy);
                if Grad > 1e-8, co3 = 1./Grad; else, co3 = 0; end
                
                % 北
                ux = (u(i+1,j-1)-u(i-1,j-1))/2*h;
                uy = (u(i,j)-u(i,j-1))/h;
                Grad = abs(ux) + abs(uy);
                if Grad > 1e-8, co4 = 1./Grad; else, co4 = 0; end
                
                % --- 更新公式 ---
                % 此时的保真项是 (u - f_log)^2
                % 迭代式： u_new = (f_log + weight * neighbors) / (1 + weight * sum_coeff)
                
                weight = 1 / (lambda * h * h);
                coeff_sum = co1 + co2 + co3 + co4;
                neighbors = co1*u(i+1,j) + co2*u(i-1,j) + co3*u(i,j+1) + co4*u(i,j-1);
                
                % 这里为了防止 coeff_sum 为 0 (平坦区域)，加上一个小量或者只在有梯度时更新
                if coeff_sum == 0
                     u(i,j) = (f_log(i,j) + 0) / 1; % 如果周围完全平坦，保持原值或取平均
                else
                     u(i,j) = (f_log(i,j) + weight * neighbors) / (1 + weight * coeff_sum);
                end
            end
        end
        
        % 边缘条件
        u(:,1)=u(:,2); u(:,N)=u(:,N-1);
        u(1,:)=u(2,:); u(M,:)=u(M-1,:);
    end

    %% 3. 后处理：指数还原
    u_result = exp(u);
end