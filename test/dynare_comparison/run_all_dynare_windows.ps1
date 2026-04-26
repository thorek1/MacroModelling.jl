[CmdletBinding()]
param(
    [string]$OutputDir,
    [string]$ExtractScript,
    [string]$DynareMatlabPath,
    [string]$MatlabExe,
    [ValidateRange(1, 512)]
    [int]$ThreadCount = 1,
    [ValidateRange(0, 10)]
    [int]$MaxLicenseRetries = 1,
    [ValidateRange(0, 600)]
    [int]$LicenseRetryDelaySeconds = 10,
    [string[]]$SkipModels = @(),
    [string[]]$OnlyModels = @(),
    [string[]]$BenchmarkOnlyModels = @('FRBUS'),
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $PSCommandPath

if (-not $OutputDir) {
    $OutputDir = Join-Path $scriptRoot 'output'
}

if (-not $ExtractScript) {
    $ExtractScript = Join-Path $scriptRoot 'extract_dynare_results.m'
}

function Resolve-ExistingPath {
    param(
        [string[]]$Candidates,
        [string]$Description
    )

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Could not find $Description. Checked: $($Candidates -join ', ')"
}

function Get-MatlabExecutable {
    param([string]$PreferredPath)

    $candidates = @()

    if ($PreferredPath) {
        $candidates += $PreferredPath
    }
    if ($env:MATLAB_EXE) {
        $candidates += $env:MATLAB_EXE
    }

    $matlabCommand = Get-Command matlab.exe -ErrorAction SilentlyContinue
    if ($matlabCommand) {
        $candidates += $matlabCommand.Source
    }

    $matlabRoot = 'C:\Program Files\MATLAB'
    if (Test-Path -LiteralPath $matlabRoot) {
        Get-ChildItem -LiteralPath $matlabRoot -Directory |
            Sort-Object Name -Descending |
            ForEach-Object {
                $candidates += (Join-Path $_.FullName 'bin\matlab.exe')
            }
    }

    Resolve-ExistingPath -Candidates $candidates -Description 'MATLAB executable'
}

function Get-DynareMatlabPath {
    param([string]$PreferredPath)

    $candidates = @()

    if ($PreferredPath) {
        $candidates += $PreferredPath
    }
    if ($env:DYNARE_MATLAB) {
        $candidates += $env:DYNARE_MATLAB
    }
    if ($env:DYNARE_HOME) {
        $candidates += (Join-Path $env:DYNARE_HOME 'matlab')
    }

    $candidates += 'D:\CustomTools\dynare-7.0-win\matlab'
    $candidates += 'D:\CustomTools\Dynare\7\matlab'
    $candidates += 'D:\CustomTools\dynare\7\matlab'

    $resolvedPath = Resolve-ExistingPath -Candidates $candidates -Description 'Dynare matlab directory'
    $dynareEntryPoint = Join-Path $resolvedPath 'dynare.m'
    if (-not (Test-Path -LiteralPath $dynareEntryPoint)) {
        throw "Dynare matlab directory does not contain dynare.m: $resolvedPath"
    }

    $resolvedPath
}

function ConvertTo-MatlabString {
    param([string]$Value)

    $Value.Replace('\', '/').Replace("'", "''")
}

function Set-ThreadEnvironment {
    param([int]$RequestedThreadCount)

    $threadValue = [string]$RequestedThreadCount
    $threadEnvironment = [ordered]@{
        'OMP_NUM_THREADS' = $threadValue
        'OMP_THREAD_LIMIT' = $threadValue
        'OMP_DYNAMIC' = 'FALSE'
        'MKL_NUM_THREADS' = $threadValue
        'MKL_DOMAIN_NUM_THREADS' = ('MKL_ALL={0}' -f $threadValue)
        'MKL_DYNAMIC' = 'FALSE'
        'OPENBLAS_NUM_THREADS' = $threadValue
        'BLIS_NUM_THREADS' = $threadValue
        'VECLIB_MAXIMUM_THREADS' = $threadValue
        'TBB_NUM_THREADS' = $threadValue
    }

    foreach ($name in $threadEnvironment.Keys) {
        Set-Item -Path ("Env:{0}" -f $name) -Value $threadEnvironment[$name]
    }

    $threadEnvironment
}

function Invoke-MatlabBatch {
    param(
        [string]$Executable,
        [string]$WorkingDirectory,
        [string]$BatchCommand,
        [int]$RequestedThreadCount
    )

    $logPath = Join-Path $WorkingDirectory 'matlab_console.log'
    if (Test-Path -LiteralPath $logPath) {
        Remove-Item -LiteralPath $logPath -Force
    }

    $matlabArgs = @()
    if ($RequestedThreadCount -eq 1) {
        $matlabArgs += '-singleCompThread'
    }
    $matlabArgs += '-logfile'
    $matlabArgs += $logPath
    $matlabArgs += '-batch'
    $matlabArgs += $BatchCommand

    Write-Host ("Launching MATLAB: {0} {1}" -f $Executable, ($matlabArgs -join ' '))
    Write-Host ("Streaming MATLAB log: {0}" -f $logPath)

    $proc = Start-Process -FilePath $Executable -ArgumentList $matlabArgs -WorkingDirectory $WorkingDirectory -PassThru -NoNewWindow

    $doneFlagPath = Join-Path $WorkingDirectory 'batch_done.flag'
    if (Test-Path -LiteralPath $doneFlagPath) {
        Remove-Item -LiteralPath $doneFlagPath -Force
    }

    $procId = $proc.Id
    $linesPrinted = 0
    # Primary done-signal: the MATLAB driver writes batch_done.flag at the very
    # end of run_all_dynare. Poll for that file and (as a backup) check whether
    # the MATLAB process is still alive via Get-Process. Avoid method calls on
    # the process object so this works in PowerShell Constrained Language Mode.
    while ($true) {
        Start-Sleep -Milliseconds 1000
        if (Test-Path -LiteralPath $logPath) {
            $allLines = @(Get-Content -LiteralPath $logPath -ErrorAction SilentlyContinue)
            if ($allLines.Count -gt $linesPrinted) {
                for ($idx = $linesPrinted; $idx -lt $allLines.Count; $idx++) {
                    Write-Host ("[matlab] {0}" -f $allLines[$idx])
                }
                $linesPrinted = $allLines.Count
            }
        }
        if (Test-Path -LiteralPath $doneFlagPath) { break }
        $alive = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $alive) { break }
    }

    # Drain any final log lines once MATLAB is done.
    Start-Sleep -Milliseconds 500

    if (Test-Path -LiteralPath $logPath) {
        $allLines = @(Get-Content -LiteralPath $logPath -ErrorAction SilentlyContinue)
        if ($allLines.Count -gt $linesPrinted) {
            for ($idx = $linesPrinted; $idx -lt $allLines.Count; $idx++) {
                Write-Host ("[matlab] {0}" -f $allLines[$idx])
            }
        }
    }

    if ($null -eq $proc.ExitCode) {
        return 0
    }

    return $proc.ExitCode
}

function Test-MatlabLicenseCheckoutFailure {
    param([string]$LogPath)

    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }

    $logText = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
    if (-not $logText) {
        return $false
    }

    if ($logText -match 'License checkout failed') {
        return $true
    }
    if ($logText -match 'License Manager Error\s*-97') {
        return $true
    }

    return $false
}

function Update-StochSimulDirective {
    param([string]$ModFilePath)

    $content = Get-Content -LiteralPath $ModFilePath -Raw
    $updatedContent = $content -replace 'stoch_simul\s*\((?!\s*nograph\b)', 'stoch_simul(nograph, '
    $updatedContent = $updatedContent -replace 'stoch_simul\s*;', 'stoch_simul(nograph);'

    if ($updatedContent -ne $content) {
        Set-Content -LiteralPath $ModFilePath -Value $updatedContent -Encoding ascii -NoNewline
    }
}

function New-WorkDirectory {
    param(
        [string]$WorkRoot,
        [string]$ModelName
    )

    $suffix = '{0}_{1}' -f (Get-Date -Format 'yyyyMMddHHmmssfff'), (Get-Random -Minimum 10000 -Maximum 99999)
    $workDir = Join-Path $WorkRoot ("{0}_{1}" -f $ModelName, $suffix)
    New-Item -ItemType Directory -Path $workDir | Out-Null
    $workDir
}

if (-not (Test-Path -LiteralPath $OutputDir)) {
    throw "Output directory not found: $OutputDir. Run generate_julia_results.jl first or pass -OutputDir."
}

if (-not (Test-Path -LiteralPath $ExtractScript)) {
    throw "Extract script not found: $ExtractScript"
}

$resolvedOutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$resolvedExtractScript = (Resolve-Path -LiteralPath $ExtractScript).Path
$resolvedMatlabExe = Get-MatlabExecutable -PreferredPath $MatlabExe
$resolvedDynareMatlabPath = Get-DynareMatlabPath -PreferredPath $DynareMatlabPath
$dynareMatlabLiteral = ConvertTo-MatlabString -Value $resolvedDynareMatlabPath
$threadEnvironment = Set-ThreadEnvironment -RequestedThreadCount $ThreadCount

$modelDirectories = Get-ChildItem -LiteralPath $resolvedOutputDir -Directory | Sort-Object Name
if (-not $modelDirectories) {
    throw "No model directories found under $resolvedOutputDir"
}

Write-Host "Using MATLAB at: $resolvedMatlabExe"
Write-Host "Using Dynare at: $resolvedDynareMatlabPath"
Write-Host "Output root: $resolvedOutputDir"
Write-Host "Requested thread count: $ThreadCount"
$skipModelSet = @{}
foreach ($skipName in $SkipModels) {
    if (-not [string]::IsNullOrWhiteSpace($skipName)) {
        $skipModelSet[$skipName] = $true
    }
}
if ($skipModelSet.Count -gt 0) {
    Write-Host ("Skipping models: {0}" -f (($skipModelSet.Keys | Sort-Object) -join ', '))
}
$onlyModelSet = @{}
foreach ($onlyName in $OnlyModels) {
    if (-not [string]::IsNullOrWhiteSpace($onlyName)) {
        $onlyModelSet[$onlyName] = $true
    }
}
if ($onlyModelSet.Count -gt 0) {
    Write-Host ("Restricting to models: {0}" -f (($onlyModelSet.Keys | Sort-Object) -join ', '))
}
$benchmarkOnlySet = @{}
foreach ($benchmarkName in $BenchmarkOnlyModels) {
    if (-not [string]::IsNullOrWhiteSpace($benchmarkName)) {
        $benchmarkOnlySet[$benchmarkName] = $true
    }
}
if ($benchmarkOnlySet.Count -gt 0) {
    Write-Host ("Benchmark-only models: {0}" -f (($benchmarkOnlySet.Keys | Sort-Object) -join ', '))
}
Write-Host 'Configured thread environment for MATLAB and MEX libraries:'
foreach ($name in $threadEnvironment.Keys) {
    Write-Host ("  {0}={1}" -f $name, $threadEnvironment[$name])
}

if ($ValidateOnly) {
    Write-Host 'Validation only mode enabled.'
    foreach ($modelDirectory in $modelDirectories) {
        $modelName = $modelDirectory.Name
        $modFile = Join-Path $modelDirectory.FullName "$modelName.mod"
        if (Test-Path -LiteralPath $modFile) {
            Write-Host "READY: $modelName"
        }
        else {
            Write-Warning "SKIP: No .mod file found for $modelName"
        }
    }
    return
}

$workRoot = Join-Path $scriptRoot '_dynare_work'
New-Item -ItemType Directory -Path $workRoot -Force | Out-Null

$batchRoot = Join-Path $workRoot ("batch_{0}_{1}" -f (Get-Date -Format 'yyyyMMddHHmmssfff'), (Get-Random -Minimum 10000 -Maximum 99999))
New-Item -ItemType Directory -Path $batchRoot | Out-Null

# Prepare per-model working directories under a single batch root so MATLAB can
# iterate through them in one session (avoids per-model license checkouts).
$modelEntries = @()
$failedModels = @()
$dynareStub = 'm'

foreach ($modelDirectory in $modelDirectories) {
    $modelName = $modelDirectory.Name
    $modFile = Join-Path $modelDirectory.FullName "$modelName.mod"

    if ($skipModelSet.ContainsKey($modelName)) {
        Write-Host "SKIP (configured): $modelName"
        continue
    }

    if ($onlyModelSet.Count -gt 0 -and -not $onlyModelSet.ContainsKey($modelName)) {
        Write-Host "SKIP (not in OnlyModels): $modelName"
        continue
    }

    if (-not (Test-Path -LiteralPath $modFile)) {
        Write-Warning "SKIP: No .mod file found for $modelName"
        continue
    }

    $dynareOutputDir = Join-Path $modelDirectory.FullName 'dynare'
    if (Test-Path -LiteralPath $dynareOutputDir) {
        Get-ChildItem -LiteralPath $dynareOutputDir -Force | Remove-Item -Recurse -Force
    }
    else {
        New-Item -ItemType Directory -Path $dynareOutputDir | Out-Null
    }

    $modelWorkDir = Join-Path $batchRoot $modelName
    New-Item -ItemType Directory -Path $modelWorkDir | Out-Null
    $stubModFile = Join-Path $modelWorkDir "$dynareStub.mod"
    Copy-Item -LiteralPath $modFile -Destination $stubModFile
    Copy-Item -LiteralPath $resolvedExtractScript -Destination $modelWorkDir
    Update-StochSimulDirective -ModFilePath $stubModFile

    $modelEntries += @{
        Name           = $modelName
        WorkDir        = $modelWorkDir
        DynareOutDir   = $dynareOutputDir
        BenchmarkOnly  = [bool]$benchmarkOnlySet.ContainsKey($modelName)
    }
}

if (-not $modelEntries) {
    Write-Warning 'No model entries to process. Phase 2 complete.'
    return
}

# Build the MATLAB driver that runs all models in one session.
$driverScriptPath = Join-Path $batchRoot 'run_all_dynare.m'
$workRootLiteral = ConvertTo-MatlabString -Value $batchRoot

$modelEntryLines = @()
foreach ($entry in $modelEntries) {
    $nameLiteral = ConvertTo-MatlabString -Value $entry.Name
    $workLiteral = ConvertTo-MatlabString -Value $entry.WorkDir
    $benchmarkLiteral = if ($entry.BenchmarkOnly) { 'true' } else { 'false' }
    $modelEntryLines += "model_entries(end+1) = struct('name', '$nameLiteral', 'work_dir', '$workLiteral', 'benchmark_only', $benchmarkLiteral);"
}
$modelEntriesBlock = ($modelEntryLines -join "`n    ")

$driverScript = @"
diary('matlab_batch.log');
diary on;
addpath('$dynareMatlabLiteral');

requested_threads = $ThreadCount;
thread_env_names = {'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_DYNAMIC', 'MKL_NUM_THREADS', 'MKL_DOMAIN_NUM_THREADS', 'MKL_DYNAMIC', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TBB_NUM_THREADS'};
if exist('maxNumCompThreads', 'builtin') || exist('maxNumCompThreads', 'file')
    previous_num_comp_threads = maxNumCompThreads(requested_threads);
    active_num_comp_threads = maxNumCompThreads();
    fprintf('MATLAB thread configuration: requested=%d active=%d previous=%d\n', requested_threads, active_num_comp_threads, previous_num_comp_threads);
else
    fprintf('MATLAB thread configuration: requested=%d active=maxNumCompThreads unavailable\n', requested_threads);
end
for thread_env_idx = 1:numel(thread_env_names)
    thread_env_name = thread_env_names{thread_env_idx};
    thread_env_value = getenv(thread_env_name);
    if isempty(thread_env_value)
        thread_env_value = '<unset>';
    end
    fprintf('MATLAB thread environment: %s=%s\n', thread_env_name, thread_env_value);
end

batch_root = '$workRootLiteral';
status_file = fullfile(batch_root, 'model_status.csv');
status_fid = fopen(status_file, 'w');
fprintf(status_fid, 'model,status,message\n');

model_entries = struct('name', {}, 'work_dir', {}, 'benchmark_only', {});
    $modelEntriesBlock

original_dir = pwd;
batch_start_tic = tic;
for entry_idx = 1:numel(model_entries)
    entry = model_entries(entry_idx);
    fprintf('========================================\n');
    fprintf('[%s] (%d/%d) Running Dynare on: %s\n', datestr(now, 'HH:MM:SS'), entry_idx, numel(model_entries), entry.name);
    fprintf('========================================\n');
    cd(entry.work_dir);
    model_tic = tic;
    try
        clearvars -except status_fid model_entries entry_idx entry batch_root original_dir requested_threads thread_env_names previous_num_comp_threads active_num_comp_threads batch_start_tic model_tic;
        model_name = entry.name;
        output_dir = 'dynare_output';
        benchmark_only_mode = entry.benchmark_only;
        dynare $dynareStub noclearall;
        extract_dynare_results;
        elapsed_model = toc(model_tic);
        fprintf('[%s] OK: %s in %.1f s\n', datestr(now, 'HH:MM:SS'), entry.name, elapsed_model);
        fprintf(status_fid, '%s,ok,\n', entry.name);
    catch ME
        elapsed_model = toc(model_tic);
        report_text = getReport(ME, 'extended', 'hyperlinks', 'off');
        fid = fopen('matlab_error.log', 'w');
        fprintf(fid, '%s\n', report_text);
        fclose(fid);
        fprintf('[%s] ERROR: %s after %.1f s -- %s\n', datestr(now, 'HH:MM:SS'), entry.name, elapsed_model, ME.message);
        disp(report_text);
        message = strrep(ME.message, ',', ';');
        message = strrep(message, sprintf('\n'), ' ');
        fprintf(status_fid, '%s,error,%s\n', entry.name, message);
    end
    cd(original_dir);
end
fprintf('[%s] Batch finished in %.1f s\n', datestr(now, 'HH:MM:SS'), toc(batch_start_tic));

fclose(status_fid);
done_fid = fopen('batch_done.flag', 'w');
fprintf(done_fid, 'done\n');
fclose(done_fid);
diary off;
exit(0);
"@

Set-Content -LiteralPath $driverScriptPath -Value $driverScript -Encoding ascii

Write-Host '----------------------------------------'
Write-Host ("Launching single MATLAB session for {0} model(s) at thread count {1}..." -f $modelEntries.Count, $ThreadCount)
Write-Host '----------------------------------------'

$matlabLogPath = Join-Path $batchRoot 'matlab_console.log'
$attempt = 0
$matlabExitCode = 1
while ($true) {
    $attempt += 1
    if ($attempt -gt 1) {
        Write-Warning ("Restarting MATLAB batch after license checkout error (attempt {0}/{1})." -f $attempt, ($MaxLicenseRetries + 1))
    }

    $matlabExitCode = Invoke-MatlabBatch -Executable $resolvedMatlabExe -WorkingDirectory $batchRoot -BatchCommand 'run_all_dynare' -RequestedThreadCount $ThreadCount

    if ($matlabExitCode -eq 0) {
        break
    }

    $isLicenseFailure = Test-MatlabLicenseCheckoutFailure -LogPath $matlabLogPath
    $hasRetryBudget = $attempt -le $MaxLicenseRetries
    if (-not $isLicenseFailure -or -not $hasRetryBudget) {
        break
    }

    if ($LicenseRetryDelaySeconds -gt 0) {
        Write-Host ("Waiting {0} seconds before MATLAB restart..." -f $LicenseRetryDelaySeconds)
        Start-Sleep -Seconds $LicenseRetryDelaySeconds
    }
}

$statusFile = Join-Path $batchRoot 'model_status.csv'
$statusByModel = @{}
if (Test-Path -LiteralPath $statusFile) {
    $statusRows = Import-Csv -LiteralPath $statusFile
    foreach ($row in $statusRows) {
        $statusByModel[$row.model] = $row
    }
}
else {
    Write-Warning ("MATLAB status file not produced at {0}; treating all models as failed (MATLAB exit code {1})." -f $statusFile, $matlabExitCode)
}

$keepBatch = $false
foreach ($entry in $modelEntries) {
    $row = $null
    if ($statusByModel.ContainsKey($entry.Name)) {
        $row = $statusByModel[$entry.Name]
    }
    $matlabOutputDir = Join-Path $entry.WorkDir 'dynare_output'

    if (-not $row) {
        $failedModels += $entry.Name
        Write-Warning ("No status recorded for {0}. Likely MATLAB aborted before processing it." -f $entry.Name)
        $keepBatch = $true
        continue
    }

    if ($row.status -ne 'ok') {
        $failedModels += $entry.Name
        $errMessage = $row.message
        if (-not $errMessage) { $errMessage = '<no message>' }
        Write-Warning ("Dynare failed for {0}: {1}" -f $entry.Name, $errMessage)
        $keepBatch = $true
        continue
    }

    if (-not (Test-Path -LiteralPath $matlabOutputDir)) {
        $failedModels += $entry.Name
        Write-Warning ("Status reported ok for {0} but no output directory found at {1}." -f $entry.Name, $matlabOutputDir)
        $keepBatch = $true
        continue
    }

    $outputFiles = Get-ChildItem -LiteralPath $matlabOutputDir -File
    if (-not $outputFiles) {
        $failedModels += $entry.Name
        Write-Warning ("Status ok for {0} but no CSV files produced." -f $entry.Name)
        $keepBatch = $true
        continue
    }

    foreach ($outputFile in $outputFiles) {
        Copy-Item -LiteralPath $outputFile.FullName -Destination $entry.DynareOutDir -Force
    }

    Write-Host ("Done: {0} (results copied to {1})" -f $entry.Name, $entry.DynareOutDir)
}

if ($matlabExitCode -ne 0) {
    Write-Warning ("MATLAB exited with code {0}. Successfully processed models were still copied." -f $matlabExitCode)
    $keepBatch = $true
}

if ($keepBatch) {
    Write-Warning "Keeping batch work directory for inspection: $batchRoot"
}
else {
    Remove-Item -LiteralPath $batchRoot -Recurse -Force
}

if ($failedModels.Count -gt 0) {
    Write-Warning ("Phase 2 finished with failures in: {0}" -f ($failedModels -join ', '))
    throw ("Phase 2 failed for {0} model(s): {1}. Batch kept at {2}" -f $failedModels.Count, ($failedModels -join ', '), $batchRoot)
}

if ($matlabExitCode -ne 0) {
    throw ("Phase 2 failed: MATLAB exited with code {0}. Batch kept at {1}" -f $matlabExitCode, $batchRoot)
}

Write-Host 'Phase 2 complete.'