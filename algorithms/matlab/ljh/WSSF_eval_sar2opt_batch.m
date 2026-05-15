%使用方法的命令
%input_dir='D:\hand_craft_registration\organized_pairs_for_eval\HT_random_sar_aug_20260510'; result_dir=fullfile(pwd,'ht_random_sar_aug_eval_results'); result_tag='ht_random_sar_aug'; WSSF_eval_sar2opt_batch
if exist('input_dir', 'var')
    input_dir_in = input_dir;
else
    input_dir_in = [];
end
if exist('result_dir', 'var')
    result_dir_in = result_dir;
else
    result_dir_in = [];
end
if exist('result_tag', 'var')
    result_tag_in = result_tag;
else
    result_tag_in = [];
end

clc;
clearvars -except input_dir_in result_dir_in result_tag_in;
close all;
beep off;
warning('off');

this_file = which(mfilename);
if isempty(this_file)
    base_dir = pwd;
else
    base_dir = fileparts(this_file);
end
addpath(genpath(fullfile(base_dir, 'PSATF')));
addpath(genpath(fullfile(base_dir, 'Others')));

if ~isempty(input_dir_in)
    input_dir = input_dir_in;
elseif ~exist('input_dir', 'var') || isempty(input_dir)
    input_dir = 'D:\hand_craft_registration\organized_pairs_for_eval\HT_random_sar_aug_20260510';
end
if ~isempty(result_dir_in)
    result_dir = result_dir_in;
elseif ~exist('result_dir', 'var') || isempty(result_dir)
    result_dir = fullfile(base_dir, 'sar2opt_eval_results');
end
if ~isempty(result_tag_in)
    result_tag = result_tag_in;
elseif ~exist('result_tag', 'var') || isempty(result_tag)
    result_tag = 'sar2opt';
end

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

pair_files = dir(fullfile(input_dir, 'pair*_1.jpg'));
pair_ids = zeros(0, 1);

for i = 1:numel(pair_files)
    tokens = regexp(pair_files(i).name, '^pair(\d+)_1\.jpg$', 'tokens', 'once');
    if ~isempty(tokens)
        pair_ids(end + 1, 1) = str2double(tokens{1}); %#ok<SAGROW>
    end
end

pair_ids = sort(unique(pair_ids));
num_pairs = numel(pair_ids);

results = repmat(struct( ...
    'pair_id', 0, ...
    'rmse', NaN, ...
    'match_count', 0, ...
    'success', false, ...
    'status', ''), num_pairs, 1);

detail_path = fullfile(result_dir, [result_tag '_metrics.txt']);
fid = fopen(detail_path, 'w');
fprintf(fid, 'pair_id rmse match_count success status\n');

for i = 1:num_pairs
    pair_id = pair_ids(i);
    image_path_1 = fullfile(input_dir, sprintf('pair%d_1.jpg', pair_id));
    image_path_2 = fullfile(input_dir, sprintf('pair%d_2.jpg', pair_id));

    metrics = WSSF_eval_pair_metrics(image_path_1, image_path_2);
    results(i).pair_id = pair_id;
    results(i).rmse = metrics.rmse;
    results(i).match_count = metrics.match_count;
    results(i).success = metrics.success;
    results(i).status = metrics.status;

    fprintf(fid, 'pair%d %.6f %d %d %s\n', ...
        pair_id, metrics.rmse, metrics.match_count, metrics.success, metrics.status);

    fprintf('pair%d | rmse = %.6f | match_count = %d | success = %d | status = %s\n', ...
        pair_id, metrics.rmse, metrics.match_count, metrics.success, metrics.status);
end

fclose(fid);

all_rmse = [results.rmse];
valid_rmse = all_rmse(~isnan(all_rmse));
all_match_counts = [results.match_count];
all_success = [results.success];

summary = struct();
summary.input_dir = input_dir;
summary.total_pairs = num_pairs;
summary.success_pair_count = sum(all_success);
summary.average_match_count = mean(all_match_counts);

if isempty(valid_rmse)
    summary.average_rmse = NaN;
else
    summary.average_rmse = mean(valid_rmse);
end

save(fullfile(result_dir, [result_tag '_metrics.mat']), 'results', 'summary');

summary_path = fullfile(result_dir, [result_tag '_summary.txt']);
fid = fopen(summary_path, 'w');
fprintf(fid, 'Input directory: %s\n', summary.input_dir);
fprintf(fid, 'Total pairs: %d\n', summary.total_pairs);
fprintf(fid, 'Average RMSE: %.6f\n', summary.average_rmse);
fprintf(fid, 'Average match count: %.6f\n', summary.average_match_count);
fprintf(fid, 'Successful pair count (>4 matches): %d\n', summary.success_pair_count);
fclose(fid);

disp('==========================================');
disp(['Input directory: ' summary.input_dir]);
disp(['Total pairs: ' num2str(summary.total_pairs)]);
disp(['Average RMSE: ' num2str(summary.average_rmse)]);
disp(['Average match count: ' num2str(summary.average_match_count)]);
disp(['Successful pair count (>4 matches): ' num2str(summary.success_pair_count)]);
disp(['Detail file: ' detail_path]);
disp(['Summary file: ' summary_path]);
