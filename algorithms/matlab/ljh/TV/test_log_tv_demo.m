function test_log_tv_demo()
    clc; clear; close all;

    %% 1. 生成原始图像 (黑色背景，白色色块)
    N = 256;
    u_clean = zeros(N, N);
    % 在中间画一个白色方块 (值为 200，模拟非饱和亮度)
    u_clean(64:192, 64:192) = 200; 
    % 稍微做一点平滑，避免边缘过于生硬导致数值计算震荡（可选）
    % u_clean = imgaussfilt(u_clean, 1);
    
    u_clean = double(u_clean); % 转换为双精度

    %% 2. 生成两种噪声图像
    
    % --- 情况 A: 加性高斯噪声 (Additive Gaussian Noise) ---
    % 模型: f = u + n
    sigma_add = 30; % 噪声标准差
    noise_add = sigma_add * randn(size(u_clean));
    f_additive = u_clean + noise_add;
    
    % 截断负值，保证图像物理意义
    f_additive(f_additive < 0) = 0; 

    % --- 情况 B: 乘性高斯噪声 (Multiplicative Gaussian Noise) ---
    % 模型: f = u * (1 + n)  => 类似散斑噪声
    sigma_mult = 0.2; % 噪声强度
    noise_mult = sigma_mult * randn(size(u_clean));
    f_multiplicative = u_clean .* (1 + noise_mult);
    
    % 截断负值 (防止取对数时出现复数)
    f_multiplicative(f_multiplicative < 0) = 1e-3; 

    %% 3. 调用 Log-TV 算法进行去噪
    
    % 参数设置
    % lambda: 数据保真项权重。
    % 越大 -> 越接近原图（去噪弱，细节多）；越小 -> 越平滑（去噪强，变卡通）
    % Log-TV 因为在对数域，数值范围变小，通常 lambda 需要调整
    lambda = 20;   
    iter = 300;     % 迭代次数
    
    fprintf('正在处理加性噪声图片...\n');
    u_res_add = log_total_variation(f_additive, iter, lambda);
    
    fprintf('正在处理乘性噪声图片...\n');
    u_res_mult = log_total_variation(f_multiplicative, iter, lambda);

    %% 4. 绘图展示结果
    figure('Name', 'Log-TV 去噪效果对比', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
    
    % --- 第一行：加性噪声实验 ---
    subplot(2, 3, 1); imshow(u_clean, []); title('原始图像');
    subplot(2, 3, 2); imshow(f_additive, []); title('加性高斯噪声 (Additive)');
    subplot(2, 3, 3); imshow(u_res_add, []); title(['Log-TV 去噪结果 (\lambda=',num2str(lambda),')']);
    xlabel('预期效果：较差 (黑色背景处可能会有斑块)');

    % --- 第二行：乘性噪声实验 ---
    subplot(2, 3, 4); imshow(u_clean, []); title('原始图像');
    subplot(2, 3, 5); imshow(f_multiplicative, []); title('乘性高斯噪声 (Multiplicative)');
    subplot(2, 3, 6); imshow(u_res_mult, []); title(['Log-TV 去噪结果 (\lambda=',num2str(lambda),')']);
    xlabel('预期效果：优秀 (对数变换将乘性转为加性)');
    
end

%% ==========================================================
%%  以下是你刚才修改过的 Log-TV 函数实现
%% ==========================================================
function u_result = log_total_variation(u0, IterMax, lambda)
    % 1. 预处理：转换到对数域
    epsilon_log = 1e-1; % 防止 log(0)
    u0 = double(u0);
    
    % 核心变换：将乘性问题转化为加性问题
    f_log = log(u0 + epsilon_log); 

    %% 初始化
    u = f_log; 
    [M, N] = size(u);  
    % Energy = zeros(1, IterMax);
    h = 1;

    %% 迭代
    for Iter = 1:IterMax    
        
        for i = 2:M-1       
            for j = 2:N-1
                % 系数计算
                ux = (u(i+1,j)-u(i,j))/h;
                uy = (u(i,j+1)-u(i,j-1))/2*h;
                Grad = sqrt(ux*ux+uy*uy);
                if Grad ~= 0, co1 = 1./Grad; else, co1 = Grad; end
                
                ux = (u(i,j)-u(i-1,j))/h;
                uy = (u(i-1,j+1)-u(i-1,j-1))/2*h;
                Grad = sqrt(ux*ux+uy*uy);
                if Grad ~= 0, co2 = 1./Grad; else, co2 = Grad; end
                
                ux = (u(i+1,j)-u(i-1,j))/2*h;
                uy = (u(i,j+1)-u(i,j))/h;
                Grad = sqrt(ux*ux+uy*uy);
                if Grad ~= 0, co3 = 1./Grad; else, co3 = Grad; end
                
                ux = (u(i+1,j-1)-u(i-1,j-1))/2*h;
                uy = (u(i,j)-u(i,j-1))/h;
                Grad = sqrt(ux*ux+uy*uy);
                if Grad ~= 0, co4 = 1./Grad; else, co4 = Grad; end
                
                % 核心更新：使用对数域的 f_log 作为保真目标
                coeff_sum = co1 + co2 + co3 + co4;
                term_neighbors = co1*u(i+1,j) + co2*u(i-1,j) + co3*u(i,j+1) + co4*u(i,j-1);
                
                % 迭代公式 (对应 Energy = TV + lambda * (u-f)^2 )
                % 注意：这里的 lambda 在分母位置的推导形式决定了它的强弱
                % 根据原代码逻辑：
                numerator = f_log(i,j) + (1/(lambda*h*h)) * term_neighbors;
                denominator = 1 + (1/(lambda*h*h)) * coeff_sum;
                
                u(i,j) = numerator / denominator;     
            end
        end
        
        % 边缘条件
        for i = 2:M-1
            u(i,1)=u(i,2); u(i,N)=u(i,N-1);
        end
        for j = 2:N-1   
            u(1,j)=u(2,j); u(M,j)=u(M-1,j);
        end
        u(1,1)=u(2,2); u(1,N)=u(2,N-1);
        u(M,1)=u(M-1,2); u(M,N)=u(M-1,N-1);
    end

    %% 3. 后处理：指数还原
    u_result = exp(u);
end