function u_result = log_total_variation(u0, IterMax, lambda)
%% 1. 预处理：转换到对数域 (Log-Domain)
% 乘性噪声模型 f = u * n  =>  log(f) = log(u) + log(n)
% 转换后变成了加性噪声，可以直接套用 TV 算法

% 加一个极小值 eps 防止 log(0) 报错
epsilon_log = 1e-8; 
u0 = double(u0); % 确保是双精度
f_log = log(u0 + epsilon_log); 

%% 初始化 (在对数域进行)
u = f_log; % 现在的 u 代表的是 log(image)
[M, N] = size(u);  
Energy = zeros(1, IterMax);
% 空间离散
h = 1;

%% 迭代 (核心算法不变，只是数据变了)
for Iter = 1:IterMax    
    
    for i = 2:M-1       
        for j = 2:N-1
            % --- 梯度与系数计算 (保持原样) ---
            ux = (u(i+1,j) - u(i,j))/h;
            uy = (u(i,j+1) - u(i,j-1))/2*h;
            Grad = sqrt(ux*ux + uy*uy);
            
            if Grad ~= 0  
                co1 = 1./Grad; 
            else
                co1 = Grad; % 这里的处理保留原本逻辑，虽然数学上通常加epsilon
            end
            
            ux = (u(i,j) - u(i-1,j))/h;
            uy = (u(i-1,j+1) - u(i-1,j-1))/2*h;
            Grad = sqrt(ux*ux + uy*uy);
            if Grad ~= 0  
                co2 = 1./Grad; 
            else
                co2 = Grad;
            end
            
            ux = (u(i+1,j) - u(i-1,j))/2*h;
            uy = (u(i,j+1) - u(i,j))/h;
            Grad = sqrt(ux*ux + uy*uy);
            if Grad ~= 0 
                co3 = 1./Grad; 
            else
                co3 = Grad;
            end
            
            ux = (u(i+1,j-1) - u(i-1,j-1))/2*h;
            uy = (u(i,j) - u(i,j-1))/h;
            Grad = sqrt(ux*ux + uy*uy);
            if Grad ~= 0 
                co4 = 1./Grad; 
            else
                co4 = Grad;
            end
            
            % --- 核心更新公式修改 ---
            % 原代码: u0(i,j)
            % 新代码: f_log(i,j) -> 使用对数域的观测值
            
            coeff_sum = co1 + co2 + co3 + co4;
            term_diff = co1*u(i+1,j) + co2*u(i-1,j) + co3*u(i,j+1) + co4*u(i,j-1);
            
            % 这里的公式逻辑是高斯-赛德尔迭代的变体
            % 注意：原代码中的 lambda 作用位置比较特殊，我保持原样
            % 但把 u0(i,j) 替换为了 f_log(i,j)
            
            numerator = f_log(i,j) + (1/(lambda*h*h)) * term_diff;
            denominator = 1 + (1/(lambda*h*h)) * coeff_sum;
            
            u(i,j) = numerator * (1/denominator);
        end
    end
    
    % --- 边缘条件 (保持原样) ---
    for i = 2:M-1
        u(i,1) = u(i,2);
        u(i,N) = u(i,N-1);
    end
    for j = 2:N-1   
        u(1,j) = u(2,j);
        u(M,j) = u(M-1,j);
    end
    u(1,1) = u(2,2); u(1,N) = u(2,N-1);
    u(M,1) = u(M-1,2); u(M,N) = u(M-1,N-1);
    
    % --- 能量计算 (修改为对数域距离) ---
    en = 0.0;
    for i = 2:M-1
        for j = 2:N-1
            ux = (u(i+1,j) - u(i,j))/h;
            uy = (u(i,j+1) - u(i,j))/h;
            
            % 原代码: (u0 - u)^2
            % 新代码: (f_log - u)^2
            fid = (f_log(i,j) - u(i,j))^2;
            
            en = en + sqrt(ux*ux + uy*uy) + lambda * fid;
        end
    end
    Energy(Iter) = en;
end

%% 3. 后处理：转换回线性域 (Exp Transform)
% 这一步非常关键，把对数还原回去
u_result = exp(u);

%% 绘图
% figure;
% subplot(1,2,1); plot(Energy); title('Energy Descent');
% subplot(1,2,2); imshow(uint8(u_result)); title('Denoised Result');

end