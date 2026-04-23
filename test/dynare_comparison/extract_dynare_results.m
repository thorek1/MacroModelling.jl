% extract_dynare_results.m
% Extracts Dynare results after stoch_simul and saves them as CSV files.
%
% Expects:
%   - model_name: string variable set before calling this script
%   - output_dir: string variable for the output directory
%   - Dynare's oo_, M_, options_ structures populated after stoch_simul
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
%
% Higher-order outputs (when options_.order >= 2):
%   ghxx.csv, ghxu.csv, ghuu.csv, ghs2.csv
% Higher-order outputs (when options_.order >= 3):
%   ghxxx.csv, ghxxu.csv, ghxuu.csv, ghuuu.csv, ghxss.csv, ghuss.csv

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
if isfield(oo_, 'dr') && isfield(oo_.dr, 'state_var') && ~isempty(oo_.dr.state_var)
    state_var_idx = oo_.dr.state_var;
elseif isfield(M_, 'state_var') && ~isempty(M_.state_var)
    % Dynare may store state metadata in M_ for some solver paths.
    if isnumeric(M_.state_var)
        state_var_idx = M_.state_var;
    elseif isstruct(M_.state_var)
        if isfield(M_.state_var, 'decl')
            state_var_idx = M_.state_var.decl;
        elseif isfield(M_.state_var, 'idx')
            state_var_idx = M_.state_var.idx;
        else
            state_var_idx = find(M_.lead_lag_incidence(1, :));
        end
    else
        state_var_idx = find(M_.lead_lag_incidence(1, :));
    end
else
    % Robust fallback: lagged endogenous variables in declaration order.
    state_var_idx = find(M_.lead_lag_incidence(1, :));
end

state_var_idx = state_var_idx(:);
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

%% --- Second-order matrices (when order >= 2) ---
if options_.order >= 2 && isfield(oo_.dr, 'ghxx')
    ghxx_dr = oo_.dr.ghxx;
    ghxu_dr = oo_.dr.ghxu;
    ghuu_dr = oo_.dr.ghuu;
    ghs2_dr = oo_.dr.ghs2;

    ghxx_decl = zeros(n_endo, size(ghxx_dr, 2));
    ghxu_decl = zeros(n_endo, size(ghxu_dr, 2));
    ghuu_decl = zeros(n_endo, size(ghuu_dr, 2));
    ghs2_decl = zeros(n_endo, 1);

    ghxx_decl(oo_.dr.order_var, :) = ghxx_dr;
    ghxu_decl(oo_.dr.order_var, :) = ghxu_dr;
    ghuu_decl(oo_.dr.order_var, :) = ghuu_dr;
    ghs2_decl(oo_.dr.order_var, :) = ghs2_dr;

    dlmwrite(fullfile(output_dir, 'ghxx.csv'), ghxx_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghxu.csv'), ghxu_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghuu.csv'), ghuu_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghs2.csv'), ghs2_decl, 'precision', '%.16g');

    fprintf('Extracted second-order matrices: ghxx(%dx%d) ghxu(%dx%d) ghuu(%dx%d) ghs2(%dx1)\n', ...
            size(ghxx_decl,1), size(ghxx_decl,2), ...
            size(ghxu_decl,1), size(ghxu_decl,2), ...
            size(ghuu_decl,1), size(ghuu_decl,2), ...
            size(ghs2_decl,1));
end

%% --- Third-order matrices (when order >= 3) ---
if options_.order >= 3 && isfield(oo_.dr, 'ghxxx')
    ghxxx_dr = oo_.dr.ghxxx;
    ghxxu_dr = oo_.dr.ghxxu;
    ghxuu_dr = oo_.dr.ghxuu;
    ghuuu_dr = oo_.dr.ghuuu;
    ghxss_dr = oo_.dr.ghxss;
    ghuss_dr = oo_.dr.ghuss;

    ghxxx_decl = zeros(n_endo, size(ghxxx_dr, 2));
    ghxxu_decl = zeros(n_endo, size(ghxxu_dr, 2));
    ghxuu_decl = zeros(n_endo, size(ghxuu_dr, 2));
    ghuuu_decl = zeros(n_endo, size(ghuuu_dr, 2));
    ghxss_decl = zeros(n_endo, size(ghxss_dr, 2));
    ghuss_decl = zeros(n_endo, size(ghuss_dr, 2));

    ghxxx_decl(oo_.dr.order_var, :) = ghxxx_dr;
    ghxxu_decl(oo_.dr.order_var, :) = ghxxu_dr;
    ghxuu_decl(oo_.dr.order_var, :) = ghxuu_dr;
    ghuuu_decl(oo_.dr.order_var, :) = ghuuu_dr;
    ghxss_decl(oo_.dr.order_var, :) = ghxss_dr;
    ghuss_decl(oo_.dr.order_var, :) = ghuss_dr;

    dlmwrite(fullfile(output_dir, 'ghxxx.csv'), ghxxx_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghxxu.csv'), ghxxu_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghxuu.csv'), ghxuu_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghuuu.csv'), ghuuu_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghxss.csv'), ghxss_decl, 'precision', '%.16g');
    dlmwrite(fullfile(output_dir, 'ghuss.csv'), ghuss_decl, 'precision', '%.16g');

    fprintf('Extracted third-order matrices: ghxxx(%dx%d) ghxxu(%dx%d) ghxuu(%dx%d) ghuuu(%dx%d) ghxss(%dx%d) ghuss(%dx%d)\n', ...
            size(ghxxx_decl,1), size(ghxxx_decl,2), ...
            size(ghxxu_decl,1), size(ghxxu_decl,2), ...
            size(ghxuu_decl,1), size(ghxuu_decl,2), ...
            size(ghuuu_decl,1), size(ghuuu_decl,2), ...
            size(ghxss_decl,1), size(ghxss_decl,2), ...
            size(ghuss_decl,1), size(ghuss_decl,2));
end

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

%% --- Benchmark: component-level timing ---
% Decomposes the solution pipeline into individually timed components.
% For all models: Jacobian, first-order solve, [Hessian, second-order solve]
% For k_order models (order 3): also export bundled k_order_pert as an additional direct reference.
n_bench = 100;

exo_ss = oo_.exo_steady_state;
if isfield(oo_, 'exo_det_steady_state')
    exo_det_ss = oo_.exo_det_steady_state;
else
    exo_det_ss = zeros(M_.exo_det_nbr, 1);
end
exo_ss_full = [exo_ss; exo_det_ss];

% Decompose stochastic_solvers into individual components for every order.
dyn_endo_ss = repmat(oo_.dr.ys, 3, 1);

% ── Jacobian (dynamic_g1) ──
bench_times_jac = zeros(1, n_bench);
if options_.order >= 2
    % order >= 2 needs T_order, T outputs for Hessian computation
    for i = 1:n_bench
        tic;
        [g1_bench, T_order_bench, T_bench] = feval([M_.fname '.dynamic_g1'], ...
            dyn_endo_ss, exo_ss_full, M_.params, oo_.dr.ys, ...
            M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, ...
            M_.dynamic_g1_sparse_colptr);
        bench_times_jac(i) = toc;
    end
else
    for i = 1:n_bench
        tic;
        g1_bench = feval([M_.fname '.dynamic_g1'], ...
            dyn_endo_ss, exo_ss_full, M_.params, oo_.dr.ys, ...
            M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, ...
            M_.dynamic_g1_sparse_colptr);
        bench_times_jac(i) = toc;
    end
end
median_jac = median(bench_times_jac);
dlmwrite(fullfile(output_dir, 'benchmark_jacobian.csv'), median_jac, 'precision', '%.16g');

% ── First-order solve (dyn_first_order_solver) ──
dr_bench = oo_.dr;
bench_times_fo = zeros(1, n_bench);
for i = 1:n_bench
    tic;
    [dr_bench, ~] = dyn_first_order_solver(g1_bench, M_, dr_bench, options_, 0);
    bench_times_fo(i) = toc;
end
median_fo = median(bench_times_fo);
dlmwrite(fullfile(output_dir, 'benchmark_first_order_solve.csv'), median_fo, 'precision', '%.16g');

median_first_order_total = median_jac + median_fo;
dlmwrite(fullfile(output_dir, 'benchmark_first_order_total.csv'), median_first_order_total, 'precision', '%.16g');
dlmwrite(fullfile(output_dir, 'benchmark_first_order.csv'), median_first_order_total, 'precision', '%.16g');

fprintf('Benchmark %s (order=%d): Jac=%.1f us, FO_solve=%.1f us', ...
    model_name, options_.order, median_jac*1e6, median_fo*1e6);

if options_.order >= 2
    % ── Hessian (dynamic_g2 + build_two_dim_hessian) ──
    bench_times_hess = zeros(1, n_bench);
    for i = 1:n_bench
        tic;
        g2_v_bench = feval([M_.fname '.dynamic_g2'], dyn_endo_ss, exo_ss_full, ...
            M_.params, oo_.dr.ys, T_order_bench, T_bench);
        g2_bench = build_two_dim_hessian(M_.dynamic_g2_sparse_indices, g2_v_bench, ...
            size(g1_bench, 1), size(g1_bench, 2));
        bench_times_hess(i) = toc;
    end
    median_hess = median(bench_times_hess);
    dlmwrite(fullfile(output_dir, 'benchmark_hessian.csv'), median_hess, 'precision', '%.16g');

    % ── Second-order solve (dyn_second_order_solver) ──
    bench_times_so = zeros(1, n_bench);
    for i = 1:n_bench
        tic;
        dr_bench = dyn_second_order_solver(g1_bench, g2_bench, dr_bench, M_, ...
            options_.threads.kronecker.sparse_hessian_times_B_kronecker_C);
        bench_times_so(i) = toc;
    end
    median_so = median(bench_times_so);
    dlmwrite(fullfile(output_dir, 'benchmark_second_order_solve.csv'), median_so, 'precision', '%.16g');

    fprintf(', Hess=%.1f us, SO_solve=%.1f us', median_hess*1e6, median_so*1e6);
end

if options_.k_order_solver
    % k_order_pert remains useful as a directly measured bundled reference for order-3 runs.
    dr_korder = struct();
    if isfield(oo_.dr, 'inv_order_var'); dr_korder.inv_order_var = oo_.dr.inv_order_var; end
    if isfield(oo_.dr, 'order_var'); dr_korder.order_var = oo_.dr.order_var; end
    if isfield(oo_.dr, 'restrict_var_list'); dr_korder.restrict_var_list = oo_.dr.restrict_var_list; end
    if isfield(oo_.dr, 'restrict_columns'); dr_korder.restrict_columns = oo_.dr.restrict_columns; end
    if isfield(oo_.dr, 'obs_var'); dr_korder.obs_var = oo_.dr.obs_var; end
    dr_korder.ys = oo_.dr.ys;

    bench_times_korder = zeros(1, n_bench);
    for i = 1:n_bench
        dr_tmp = set_state_space(dr_korder, M_);
        tic;
        [dr_tmp, ~] = k_order_pert(dr_tmp, M_, options_);
        bench_times_korder(i) = toc;
    end
    median_korder = median(bench_times_korder);
    dlmwrite(fullfile(output_dir, 'benchmark_k_order_pert.csv'), median_korder, 'precision', '%.16g');

    fprintf(', k_order_pert=%.1f us', median_korder*1e6);
end

fprintf(', FO_Total=%.1f us over %d runs\n', median_first_order_total*1e6, n_bench);

disp(['Results extracted to: ' output_dir]);
