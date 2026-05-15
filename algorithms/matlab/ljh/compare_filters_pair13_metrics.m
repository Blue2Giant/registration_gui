function compare_filters_pair13_metrics(logtv_lambda)
    clc;
    close all;

    base_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(base_dir, 'TV'));
    addpath(base_dir);

    image_path = fullfile(base_dir, 'ht_eval_for_own_origin_match', 'pair13_2.jpg');
    save_dir = fullfile(base_dir, 'filter_compare_pair13');
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    input_image = imread(image_path);
    if size(input_image, 3) == 3
        input_gray = rgb2gray(input_image);
    else
        input_gray = input_image;
    end
    input_gray = double(input_gray);

    lee_n = 4;
    lee_enl = 4.5;
    lee_mode = 'r';
    tv_lambda = 0.1;
    tv_iter = 300;
    if nargin < 1 || isempty(logtv_lambda)
        logtv_lambda = 20;
    end
    logtv_iter = 300;

    fprintf('Input image: %s\n', image_path);
    fprintf('Saving outputs to: %s\n\n', save_dir);
    fprintf('Log-TV lambda: %.6f\n\n', logtv_lambda);

    lee_img = LeeFilter(input_gray, lee_n, lee_enl, lee_mode);
    tv_img = total_variation(input_gray, tv_iter, tv_lambda);
    logtv_img = log_total_variation(input_gray, logtv_iter, logtv_lambda);

    save_result(lee_img, fullfile(save_dir, 'pair13_2_lee.png'));
    save_result(tv_img, fullfile(save_dir, 'pair13_2_tv.png'));
    save_result(logtv_img, fullfile(save_dir, 'pair13_2_logtv.png'));

    lee_metrics = compute_metrics(input_gray, lee_img);
    tv_metrics = compute_metrics(input_gray, tv_img);
    logtv_metrics = compute_metrics(input_gray, logtv_img);

    fprintf('================ Filter Comparison ================\n');
    fprintf('%-10s %-12s %-12s %-12s\n', 'Filter', 'ENL', 'SNR(dB)', 'EPI');
    fprintf('%-10s %-12.4f %-12.4f %-12.4f\n', 'Lee', lee_metrics.ENL, lee_metrics.SNR, lee_metrics.EPI);
    fprintf('%-10s %-12.4f %-12.4f %-12.4f\n', 'TV', tv_metrics.ENL, tv_metrics.SNR, tv_metrics.EPI);
    fprintf('%-10s %-12.4f %-12.4f %-12.4f\n', 'Log-TV', logtv_metrics.ENL, logtv_metrics.SNR, logtv_metrics.EPI);
    fprintf('===================================================\n');

    summary_path = fullfile(save_dir, 'metrics_summary.txt');
    fid = fopen(summary_path, 'w');
    fprintf(fid, 'Input image: %s\n', image_path);
    fprintf(fid, 'Saving outputs to: %s\n\n', save_dir);
    fprintf(fid, 'Log-TV lambda: %.6f\n\n', logtv_lambda);
    fprintf(fid, '================ Filter Comparison ================\n');
    fprintf(fid, '%-10s %-12s %-12s %-12s\n', 'Filter', 'ENL', 'SNR(dB)', 'EPI');
    fprintf(fid, '%-10s %-12.4f %-12.4f %-12.4f\n', 'Lee', lee_metrics.ENL, lee_metrics.SNR, lee_metrics.EPI);
    fprintf(fid, '%-10s %-12.4f %-12.4f %-12.4f\n', 'TV', tv_metrics.ENL, tv_metrics.SNR, tv_metrics.EPI);
    fprintf(fid, '%-10s %-12.4f %-12.4f %-12.4f\n', 'Log-TV', logtv_metrics.ENL, logtv_metrics.SNR, logtv_metrics.EPI);
    fprintf(fid, '===================================================\n');
    fclose(fid);

    lambda_tag = strrep(sprintf('%.6f', logtv_lambda), '.', 'p');
    lambda_path = fullfile(save_dir, ['metrics_summary_logtv_lambda_' lambda_tag '.txt']);
    fid = fopen(lambda_path, 'w');
    fprintf(fid, 'Input image: %s\n', image_path);
    fprintf(fid, 'Saving outputs to: %s\n\n', save_dir);
    fprintf(fid, 'Log-TV lambda: %.6f\n\n', logtv_lambda);
    fprintf(fid, '================ Filter Comparison ================\n');
    fprintf(fid, '%-10s %-12s %-12s %-12s\n', 'Filter', 'ENL', 'SNR(dB)', 'EPI');
    fprintf(fid, '%-10s %-12.4f %-12.4f %-12.4f\n', 'Lee', lee_metrics.ENL, lee_metrics.SNR, lee_metrics.EPI);
    fprintf(fid, '%-10s %-12.4f %-12.4f %-12.4f\n', 'TV', tv_metrics.ENL, tv_metrics.SNR, tv_metrics.EPI);
    fprintf(fid, '%-10s %-12.4f %-12.4f %-12.4f\n', 'Log-TV', logtv_metrics.ENL, logtv_metrics.SNR, logtv_metrics.EPI);
    fprintf(fid, '===================================================\n');
    fclose(fid);
end

function save_result(img, out_path)
    img = min(max(img, 0), 255);
    imwrite(uint8(round(img)), out_path);
end

function metrics = compute_metrics(input_img, filtered_img)
    filtered_img = double(filtered_img);
    input_img = double(input_img);

    metrics.ENL = estimate_enl(filtered_img, 32);
    metrics.SNR = estimate_snr(input_img, filtered_img);
    metrics.EPI = estimate_epi(input_img, filtered_img);
end

function enl = estimate_enl(img, patch_size)
    [h, w] = size(img);
    best_score = inf;
    best_patch = img;

    for r = 1:patch_size:(h - patch_size + 1)
        for c = 1:patch_size:(w - patch_size + 1)
            patch = img(r:r + patch_size - 1, c:c + patch_size - 1);
            patch_mean = mean(patch(:));
            patch_var = var(patch(:), 1);
            if patch_mean <= 1e-6
                continue;
            end
            cv = sqrt(max(patch_var, 0)) / patch_mean;
            if cv < best_score
                best_score = cv;
                best_patch = patch;
            end
        end
    end

    patch_mean = mean(best_patch(:));
    patch_var = var(best_patch(:), 1);
    enl = (patch_mean ^ 2) / max(patch_var, eps);
end

function snr_val = estimate_snr(input_img, filtered_img)
    residual = input_img - filtered_img;
    snr_val = 10 * log10(sum(filtered_img(:) .^ 2) / max(sum(residual(:) .^ 2), eps));
end

function epi = estimate_epi(input_img, filtered_img)
    sobel_x = fspecial('sobel');
    sobel_y = sobel_x';

    gx_in = imfilter(input_img, sobel_x, 'replicate');
    gy_in = imfilter(input_img, sobel_y, 'replicate');
    edge_in = sqrt(gx_in .^ 2 + gy_in .^ 2);

    gx_out = imfilter(filtered_img, sobel_x, 'replicate');
    gy_out = imfilter(filtered_img, sobel_y, 'replicate');
    edge_out = sqrt(gx_out .^ 2 + gy_out .^ 2);

    epi = sum(edge_in(:) .* edge_out(:)) / max(sum(edge_in(:) .^ 2), eps);
end
