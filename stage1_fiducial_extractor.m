% =========================================================================
% Batch Fiducial Extraction for Recovered ECG Signals — PARALLELIZED
% =========================================================================

clc; clear; close all;

%% ── 0. USER CONFIGURATION ───────────────────────────────────────────────
RECOVERED_MAT_DIR = '/Users/sinahassannia/Desktop/ecg_lead_cover2/recovered_mat3';
FIDUCIAL_OUT_DIR  = '/Users/sinahassannia/Desktop/ecg_lead_cover2/fiducials_recovered3';
OSET_PATH         = '/Users/sinahassannia/Documents/OSET/matlab';
POWERLINE_FREQ    = 60;
LP_CUTOFF         = 30;
NUM_WORKERS       = maxNumCompThreads;   % or set manually e.g. 8

%% ── 1. ADD OSET TO PATH (main worker + all parallel workers) ────────────
addpath(genpath(OSET_PATH));

%% ── 2. DISCOVER ALL .mat FILES ──────────────────────────────────────────
mat_files = dir(fullfile(RECOVERED_MAT_DIR, '**', '*.mat'));
num_files  = numel(mat_files);
fprintf('Found %d recovered .mat files.\n\n', num_files);

if num_files == 0
    error('No .mat files found under: %s', RECOVERED_MAT_DIR);
end

%% ── 3. PRE-COMPUTE OUTPUT PATHS (outside parfor — no broadcast issues) ──
mat_paths   = cell(num_files, 1);
fid_csvs    = cell(num_files, 1);
fid_mats    = cell(num_files, 1);
fid_dirs    = cell(num_files, 1);

for i = 1 : num_files
    mat_paths{i} = fullfile(mat_files(i).folder, mat_files(i).name);
    rel_folder   = strrep(mat_files(i).folder, RECOVERED_MAT_DIR, '');
    if ~isempty(rel_folder) && rel_folder(1) == filesep
        rel_folder = rel_folder(2:end);
    end
    fid_dirs{i}  = fullfile(FIDUCIAL_OUT_DIR, rel_folder);
    stem         = mat_files(i).name(1:end-4);
    fid_csvs{i}  = fullfile(fid_dirs{i}, [stem, '_fiducial.csv']);
    fid_mats{i}  = fullfile(fid_dirs{i}, [stem, '_fiducial.mat']);
end

% Create all output directories up front (parfor workers cannot mkdir safely)
unique_dirs = unique(fid_dirs);
for k = 1 : numel(unique_dirs)
    if ~exist(unique_dirs{k}, 'dir'), mkdir(unique_dirs{k}); end
end

%% ── 4. OPEN PARALLEL POOL ───────────────────────────────────────────────
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', NUM_WORKERS);
end
fprintf('Running on %d workers.\n\n', pool.NumWorkers);

%% ── 5. PARALLEL BATCH LOOP ──────────────────────────────────────────────
% parfor cannot use struct arrays with varying fields, so we collect
% per-iteration results into plain arrays and report after the loop.

ok_flags   = false(num_files, 1);
err_msgs   = cell (num_files, 1);

parfor i = 1 : num_files   %#ok<*PFBNS>

    % ── Add OSET inside each worker (parfor workers don't inherit path) ───
    addpath(genpath(OSET_PATH));

    mat_path = mat_paths{i};
    fid_csv  = fid_csvs {i};
    fid_mat  = fid_mats {i};

    %% ── 5a. Load ─────────────────────────────────────────────────────────
    try
        data = load(mat_path);
    catch ME
        err_msgs{i} = sprintf('[LOAD ERROR] %s : %s', mat_path, ME.message);
        continue
    end

    fs         = double(data.fs);
    ecg_data   = double(data.ecg)';          % (num_leads, N)
    lead_names = cellstr(data.lead_names);
    [C, T]     = size(ecg_data);

    %% ── 5b. Preprocessing ────────────────────────────────────────────────
    try
        for c = 1 : C
            sig = ecg_data(c, :);

            % Notch filter
            Wo  = POWERLINE_FREQ / (fs / 2);
            BW  = Wo / 45;
            [b, a] = iirnotch(Wo, BW);
            sig = filtfilt(b, a, sig);

            % Baseline removal
            half_med  = round(0.3  * fs);
            half_mean = round(0.15 * fs);
            baseline  = movmean(movmedian(sig, [half_med, half_med]), ...
                                [half_mean, half_mean]);
            sig = sig - baseline;

            % High-pass (0.1 Hz)
            sig = sig - lp_filter_zero_phase(sig, 0.1 / fs);

            % Low-pass
            sig = lp_filter_zero_phase(sig, LP_CUTOFF / fs);

            ecg_data(c, :) = sig;
        end
    catch ME
        err_msgs{i} = sprintf('[PREPROC ERROR] %s : %s', mat_path, ME.message);
        continue
    end

    %% ── 5c. Fiducial extraction ──────────────────────────────────────────
    try
        win_len     = 10 * fs;
        num_windows = ceil(T / win_len);
        fiducial_windows = struct();   % accumulate multi-window results

        for n = 1 : num_windows
            s_idx = (n - 1) * win_len + 1;
            e_idx = min(n * win_len, T);
            win_ecg = ecg_data(:, s_idx : e_idx);

            [~, ecg_fiducial_position, exit_flag] = ...
                ecg_feature_extraction(win_ecg, fs, lead_names, ...
                                       [], [], [], [], false);

            if exit_flag ~= 0
                % Non-fatal — skip this window, keep going
                continue
            end

            % CSV (OSET standard)
            ecg_fiducial_handle_csv(fid_csv, win_ecg, fs, ...
                                    ecg_fiducial_position, lead_names, s_idx);

            % .mat
            if n == 1
                parsave_mat(fid_mat, ecg_fiducial_position, ...
                            lead_names, fs, s_idx, e_idx);
            else
                win_key = sprintf('window_%02d', n);
                fiducial_windows.(win_key) = ecg_fiducial_position;
                if n == num_windows
                    parsave_mat_append(fid_mat, fiducial_windows);
                end
            end
        end

        ok_flags(i) = true;

    catch ME
        err_msgs{i} = sprintf('[EXTRACT ERROR] %s : %s', mat_path, ME.message);
    end

end   % parfor

%% ── 6. SUMMARY ──────────────────────────────────────────────────────────
n_ok   = sum(ok_flags);
n_fail = sum(~ok_flags);
fprintf('\n==========================================\n');
fprintf('Done.  OK: %d  |  Failed: %d\n', n_ok, n_fail);
failed_idx = find(~ok_flags);
for k = 1 : numel(failed_idx)
    msg = err_msgs{failed_idx(k)};
    if ~isempty(msg)
        fprintf('  %s\n', msg);
    end
end
fprintf('==========================================\n');


%% ── LOCAL HELPER FUNCTIONS (needed because parfor can't use nested fns) ─

function parsave_mat(fpath, ecg_fiducial_position, lead_names, fs, ...
                     start_index, stop_index)  %#ok<DEFNU>
% Wrapper so parfor workers can call save() without variable-capture issues.
    save(fpath, 'ecg_fiducial_position', 'lead_names', ...
         'fs', 'start_index', 'stop_index');
end

function parsave_mat_append(fpath, fiducial_windows)  %#ok<DEFNU>
    save(fpath, '-append', 'fiducial_windows');
end