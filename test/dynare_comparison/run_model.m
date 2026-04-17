% run_model.m
% Wrapper script: runs Dynare on the specified model and extracts results.
%
% Usage (from shell):
%   octave --no-gui --eval "model_name='RBC_baseline'; run('run_model.m')"
%
% Expects:
%   model_name  - name of the .mod file (without extension)
%   output_dir  - (optional) directory for output files; defaults to [model_name '_results']
%
% The .mod file must be in the current working directory.

if ~exist('model_name', 'var')
    error('model_name must be set before running this script');
end

if ~exist('output_dir', 'var')
    output_dir = [model_name '_results'];
end

% Add common Dynare paths (apt-installed locations)
dynare_paths = {'/usr/lib/dynare/matlab', '/usr/share/dynare/matlab', ...
                '/usr/local/lib/dynare/matlab'};
for i = 1:length(dynare_paths)
    if exist(dynare_paths{i}, 'dir')
        addpath(dynare_paths{i});
    end
end

% Run Dynare (noclearall keeps workspace variables accessible)
eval(['dynare ' model_name ' noclearall']);

% Extract results to CSV
extract_dynare_results;
