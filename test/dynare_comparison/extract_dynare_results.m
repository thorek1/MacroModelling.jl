% extract_dynare_results.m
% Extracts Dynare results after stoch_simul and saves them as CSV files.
%
% Expects:
%   - model_name: string variable set before calling this script
%   - output_dir: string variable for the output directory
%   - Dynare's oo_, M_ structures populated after stoch_simul
%
% Outputs (all in output_dir/):
%   steady_state.csv        - steady state values (declaration order)
%   var_names.csv           - endogenous variable names (declaration order)
%   exo_names.csv           - exogenous shock names
%   state_var_names.csv     - state variable names
%   ghx.csv                 - state transition matrix (declaration order rows)
%   ghu.csv                 - shock impact matrix (declaration order rows)
%   irf_VARNAME_SHOCKNAME.csv - IRF for each var/shock combination
%   variance_covariance.csv - theoretical variance-covariance matrix (declaration order)
%   variance_decomposition.csv          - variance decomposition matrix
%   variance_decomposition_var_names.csv - variable names for var decomp rows
%   variance_decomposition_exo_names.csv - shock names for var decomp columns

if ~exist('output_dir', 'var')
    output_dir = [model_name '_results'];
end
mkdir(output_dir);

n_endo = M_.endo_nbr;
n_exo  = M_.exo_nbr;

%% --- Variable names ---
fid = fopen(fullfile(output_dir, 'var_names.csv'), 'w');
for i = 1:n_endo
    if iscell(M_.endo_names)
        fprintf(fid, '%s\n', M_.endo_names{i});
    else
        fprintf(fid, '%s\n', deblank(M_.endo_names(i,:)));
    end
end
fclose(fid);

fid = fopen(fullfile(output_dir, 'exo_names.csv'), 'w');
for i = 1:n_exo
    if iscell(M_.exo_names)
        fprintf(fid, '%s\n', M_.exo_names{i});
    else
        fprintf(fid, '%s\n', deblank(M_.exo_names(i,:)));
    end
end
fclose(fid);

%% --- Steady state (declaration order) ---
dlmwrite(fullfile(output_dir, 'steady_state.csv'), oo_.steady_state, 'precision', '%.16g');

%% --- Policy matrices (convert from DR order to declaration order) ---
ghx_dr = oo_.dr.ghx;
ghu_dr = oo_.dr.ghu;

ghx_decl = zeros(n_endo, size(ghx_dr, 2));
ghu_decl = zeros(n_endo, size(ghu_dr, 2));
ghx_decl(oo_.dr.order_var, :) = ghx_dr;
ghu_decl(oo_.dr.order_var, :) = ghu_dr;

dlmwrite(fullfile(output_dir, 'ghx.csv'), ghx_decl, 'precision', '%.16g');
dlmwrite(fullfile(output_dir, 'ghu.csv'), ghu_decl, 'precision', '%.16g');

% State variable names
state_var_idx = oo_.dr.state_var;
fid = fopen(fullfile(output_dir, 'state_var_names.csv'), 'w');
for i = 1:length(state_var_idx)
    si = state_var_idx(i);
    if iscell(M_.endo_names)
        fprintf(fid, '%s\n', M_.endo_names{si});
    else
        fprintf(fid, '%s\n', deblank(M_.endo_names(si,:)));
    end
end
fclose(fid);

%% --- IRFs ---
if isfield(oo_, 'irfs')
    irf_fields = fieldnames(oo_.irfs);
    for i = 1:length(irf_fields)
        fname = irf_fields{i};
        data = oo_.irfs.(fname);
        if ~isempty(data)
            dlmwrite(fullfile(output_dir, ['irf_' fname '.csv']), data, 'precision', '%.16g');
        end
    end
    fid = fopen(fullfile(output_dir, 'irf_fields.csv'), 'w');
    for i = 1:length(irf_fields)
        fprintf(fid, '%s\n', irf_fields{i});
    end
    fclose(fid);
end

%% --- Variance-covariance matrix (declaration order) ---
if isfield(oo_, 'var') && ~isempty(oo_.var)
    dlmwrite(fullfile(output_dir, 'variance_covariance.csv'), oo_.var, 'precision', '%.16g');
end

%% --- Variance decomposition ---
if isfield(oo_, 'variance_decomposition') && ~isempty(oo_.variance_decomposition)
    dlmwrite(fullfile(output_dir, 'variance_decomposition.csv'), ...
             oo_.variance_decomposition, 'precision', '%.16g');

    n_vd_rows = size(oo_.variance_decomposition, 1);
    fid = fopen(fullfile(output_dir, 'variance_decomposition_var_names.csv'), 'w');
    for i = 1:n_vd_rows
        if iscell(M_.endo_names)
            fprintf(fid, '%s\n', M_.endo_names{i});
        else
            fprintf(fid, '%s\n', deblank(M_.endo_names(i,:)));
        end
    end
    fclose(fid);

    fid = fopen(fullfile(output_dir, 'variance_decomposition_exo_names.csv'), 'w');
    for i = 1:n_exo
        if iscell(M_.exo_names)
            fprintf(fid, '%s\n', M_.exo_names{i});
        else
            fprintf(fid, '%s\n', deblank(M_.exo_names(i,:)));
        end
    end
    fclose(fid);
end

disp(['Results extracted to: ' output_dir]);
