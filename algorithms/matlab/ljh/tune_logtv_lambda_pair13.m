function tune_logtv_lambda_pair13()
    clc;
    close all;

    base_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(base_dir, 'TV'));

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

    logtv_iter = 300;
    lambda_list = [0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50];
    results = zeros(numel(lambda_list), 5);

    fprintf('Tuning Log-TV lambda on: %s\n', image_path);
    fprintf('Iterations: %d\n\n', logtv_iter);
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'lambda', 'ENL', 'SNR(dB)', 'EPI', 'Score');

    for k = 1:numel(lambda_list)
        lambda = lambda_list(k);
        logtv_img = log_total_variation(input_gray, logtv_iter, lambda);
        metrics = compute_metrics(input_gray, logtv_img);
        results(k, 1) = lambda;
        results(k, 2) = metrics.ENL;
        results(k, 3) = metrics.SNR;
        results(k, 4) = metrics.EPI;
    end

    score = compute_balanced_score(results(:, 2), results(:, 3), results(:, 4));
    results(:, 5) = score;

    for k = 1:size(results, 1)
        fprintf('%-10.4g %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
            results(k, 1), results(k, 2), results(k, 3), results(k, 4), results(k, 5));
    end

    [~, best_idx] = max(results(:, 5));
    best_lambda = results(best_idx, 1);
    best_img = log_total_variation(input_gray, logtv_iter, best_lambda);
    save_result(best_img, fullfile(save_dir, 'pair13_2_logtv_tuned.png'));

    summary_path = fullfile(save_dir, 'logtv_lambda_tuning.txt');
    fid = fopen(summary_path, 'w');
    fprintf(fid, 'Tuning Log-TV lambda on: %s\n', image_path);
    fprintf(fid, 'Iterations: %d\n\n', logtv_iter);
    fprintf(fid, '%-10s %-12s %-12s %-12s %-12s\n', 'lambda', 'ENL', 'SNR(dB)', 'EPI', 'Score');
    for k = 1:size(results, 1)
        fprintf(fid, '%-10.4g %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
            results(k, 1), results(k, 2), results(k, 3), results(k, 4), results(k, 5));
    end
    fprintf(fid, '\nBest lambda: %.6f\n', best_lambda);
    fprintf(fid, 'Best ENL: %.4f\n', results(best_idx, 2));
    fprintf(fid, 'Best SNR(dB): %.4f\n', results(best_idx, 3));
    fprintf(fid, 'Best EPI: %.4f\n', results(best_idx, 4));
    fclose(fid);

    fprintf('\nBest balanced lambda: %.6f\n', best_lambda);
    fprintf('Saved tuned Log-TV image to: %s\n', fullfile(save_dir, 'pair13_2_logtv_tuned.png'));
    fprintf('Saved tuning summary to: %s\n', summary_path);
end

function score = compute_balanced_score(enl_vals, snr_vals, epi_vals)
    enl_term = normalize_vector(log10(max(enl_vals, eps)));
    snr_term = normalize_vector(snr_vals);
    epi_term = normalize_vector(epi_vals);
    score = (enl_term + snr_term + epi_term) / 3;
end

function out = normalize_vector(x)
    xmin = min(x);
    xmax = max(x);
    if abs(xmax - xmin) < eps
        out = ones(size(x));
    else
        out = (x - xmin) / (xmax - xmin);
    end
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
