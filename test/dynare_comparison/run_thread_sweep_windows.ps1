# Example invocation:
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\CustomTools\MacroModelling.jl\test\dynare_comparison\run_thread_sweep_windows.ps1" -JuliaExe "D:\CustomTools\julia-1.12.6\bin\julia.exe" -DynareMatlabPath "D:\CustomTools\dynare-7.0-win\matlab" -MatlabExe "C:\Program Files\MATLAB\R2024b\bin\matlab.exe"

[CmdletBinding()]
param(
    [int[]]$ThreadCounts = @(1, 2, 4, 8),
    [string]$OutputRoot,
    [string]$JuliaExe,
    [string]$GenerateJuliaScript,
    [string]$DynareScript,
    [string]$CompareScript,
    [string]$SweepCompareScript,
    [string]$DynareMatlabPath,
    [string]$MatlabExe,
    [ValidateRange(0, 10)]
    [int]$MaxLicenseRetries = 3,
    [ValidateRange(0, 600)]
    [int]$LicenseRetryDelaySeconds = 10,
    [string[]]$OnlyModels = @(),
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $PSCommandPath
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $scriptRoot 'output_thread_sweep'
}
if (-not $GenerateJuliaScript) {
    $GenerateJuliaScript = Join-Path $scriptRoot 'generate_julia_results.jl'
}
if (-not $DynareScript) {
    $DynareScript = Join-Path $scriptRoot 'run_all_dynare_windows.ps1'
}
if (-not $CompareScript) {
    $CompareScript = Join-Path $scriptRoot 'compare_results.jl'
}
if (-not $SweepCompareScript) {
    $SweepCompareScript = Join-Path $scriptRoot 'compare_thread_sweep_results.jl'
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

function Get-JuliaExecutable {
    param([string]$PreferredPath)

    $candidates = @()

    if ($PreferredPath) {
        $candidates += $PreferredPath
    }
    if ($env:JULIA_EXE) {
        $candidates += $env:JULIA_EXE
    }

    $candidates += 'D:\CustomTools\julia-1.12.6\bin\julia.exe'
    $candidates += 'D:\CustomTools\julia-1.12.4\bin\julia.exe'

    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE '.juliaup\bin\julia.exe')
    }

    $juliaCommand = Get-Command julia.exe -ErrorAction SilentlyContinue
    if ($juliaCommand) {
        $candidates += $juliaCommand.Source
    }

    $localPrograms = Join-Path $env:LOCALAPPDATA 'Programs'
    if (Test-Path -LiteralPath $localPrograms) {
        Get-ChildItem -LiteralPath $localPrograms -Directory -Filter 'Julia*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                $candidates += (Join-Path $_.FullName 'bin\julia.exe')
            }
    }

    Resolve-ExistingPath -Candidates $candidates -Description 'Julia executable'
}

function Invoke-JuliaScript {
    param(
        [string]$Executable,
        [string]$ProjectRoot,
        [string]$ScriptPath,
        [string]$OutputArgument,
        [string]$Description,
        [int]$RequestedThreadCount,
        [string[]]$ExtraScriptArgs,
        [switch]$UseThreadCount
    )

    $juliaArgs = @("--project=$ProjectRoot")
    if ($UseThreadCount) {
        $juliaArgs += "--threads=$RequestedThreadCount"
    }
    $juliaArgs += $ScriptPath
    if ($ExtraScriptArgs) {
        foreach ($extraArg in $ExtraScriptArgs) {
            if (-not [string]::IsNullOrWhiteSpace($extraArg)) {
                $juliaArgs += $extraArg
            }
        }
    }
    $juliaArgs += $OutputArgument

    Write-Host "Running Julia step: $Description"
    & $Executable @juliaArgs

    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Julia step failed ($Description) with exit code $LASTEXITCODE"
    }
}

function Invoke-DynarePhase {
    param(
        [string]$ScriptPath,
        [string]$ThreadOutputDir,
        [int]$RequestedThreadCount,
        [string]$PreferredDynareMatlabPath,
        [string]$PreferredMatlabExe,
        [int]$RequestedMaxLicenseRetries,
        [int]$RequestedLicenseRetryDelaySeconds,
        [string[]]$RequestedOnlyModels,
        [switch]$ValidationOnly
    )

    $dynareParameters = @{
        OutputDir = $ThreadOutputDir
        ThreadCount = $RequestedThreadCount
        MaxLicenseRetries = $RequestedMaxLicenseRetries
        LicenseRetryDelaySeconds = $RequestedLicenseRetryDelaySeconds
    }

    if ($PreferredDynareMatlabPath) {
        $dynareParameters.DynareMatlabPath = $PreferredDynareMatlabPath
    }
    if ($PreferredMatlabExe) {
        $dynareParameters.MatlabExe = $PreferredMatlabExe
    }
    if ($RequestedOnlyModels -and $RequestedOnlyModels.Count -gt 0) {
        $dynareParameters.OnlyModels = $RequestedOnlyModels
    }
    if ($ValidationOnly) {
        $dynareParameters.ValidateOnly = $true
    }

    Write-Host "Running Dynare step for $RequestedThreadCount thread(s)"

    # This invokes another PowerShell script, so rely on terminating errors
    # from that script rather than $LASTEXITCODE (which may be stale from a
    # previously executed native command).
    & $ScriptPath @dynareParameters

    if (-not $?) {
        throw "Dynare step failed for $RequestedThreadCount thread(s)."
    }
}

function New-StagingOutputRoot {
    param(
        [string]$FinalOutputRoot,
        [string]$ResolvedOutputParent,
        [string]$OutputRootLeaf
    )

    Join-Path $ResolvedOutputParent ("{0}.__staging_{1}_{2}" -f $OutputRootLeaf, (Get-Date -Format 'yyyyMMddHHmmssfff'), (Get-Random -Minimum 10000 -Maximum 99999))
}

function Publish-StagedOutputRoot {
    param(
        [string]$StageOutputRoot,
        [string]$FinalOutputRoot,
        [string]$ResolvedOutputParent,
        [string]$OutputRootLeaf
    )

    if (-not (Test-Path -LiteralPath $StageOutputRoot)) {
        throw "Staged sweep output not found: $StageOutputRoot"
    }

    if (Test-Path -LiteralPath $FinalOutputRoot) {
        $previousOutputRoot = Join-Path $ResolvedOutputParent ("{0}.__previous_{1}_{2}" -f $OutputRootLeaf, (Get-Date -Format 'yyyyMMddHHmmssfff'), (Get-Random -Minimum 10000 -Maximum 99999))
        Write-Host ("Moving existing output root aside: {0} -> {1}" -f $FinalOutputRoot, $previousOutputRoot)
        Move-Item -LiteralPath $FinalOutputRoot -Destination $previousOutputRoot
    }

    Write-Host ("Publishing staged sweep output: {0} -> {1}" -f $StageOutputRoot, $FinalOutputRoot)
    Move-Item -LiteralPath $StageOutputRoot -Destination $FinalOutputRoot
}

$requestedOutputRoot = $OutputRoot
$outputRootLeaf = Split-Path -Leaf $requestedOutputRoot
$outputRootParent = Split-Path -Parent $requestedOutputRoot
if ([string]::IsNullOrWhiteSpace($outputRootParent)) {
    $outputRootParent = '.'
}
if (-not (Test-Path -LiteralPath $outputRootParent)) {
    New-Item -ItemType Directory -Path $outputRootParent -Force | Out-Null
}
$resolvedOutputParent = (Resolve-Path -LiteralPath $outputRootParent).Path
$resolvedOutputRoot = Join-Path $resolvedOutputParent $outputRootLeaf
$stagingOutputRoot = New-StagingOutputRoot -FinalOutputRoot $resolvedOutputRoot -ResolvedOutputParent $resolvedOutputParent -OutputRootLeaf $outputRootLeaf
$resolvedJuliaExe = Get-JuliaExecutable -PreferredPath $JuliaExe
$resolvedGenerateJuliaScript = Resolve-ExistingPath -Candidates @($GenerateJuliaScript) -Description 'Julia phase-1 script'
$resolvedDynareScript = Resolve-ExistingPath -Candidates @($DynareScript) -Description 'Dynare phase-2 script'
$resolvedCompareScript = Resolve-ExistingPath -Candidates @($CompareScript) -Description 'Julia phase-3 script'
$resolvedSweepCompareScript = Resolve-ExistingPath -Candidates @($SweepCompareScript) -Description 'thread-sweep summary script'

$resolvedThreadCounts = $ThreadCounts | Sort-Object -Unique
if (-not $resolvedThreadCounts) {
    throw 'At least one thread count must be provided.'
}

$resolvedOnlyModels = @()
if ($OnlyModels) {
    $resolvedOnlyModels = @($OnlyModels | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
$phase1ExtraArgs = @()
if ($resolvedOnlyModels.Count -gt 0) {
    $phase1ExtraArgs += ("--only-models={0}" -f ($resolvedOnlyModels -join ','))
    Write-Host ("Restricting sweep to models: {0}" -f ($resolvedOnlyModels -join ', '))
}

Write-Host "Repository root: $repoRoot"
Write-Host "Julia executable: $resolvedJuliaExe"
Write-Host "Final sweep output root: $resolvedOutputRoot"
Write-Host "Sweep staging root: $stagingOutputRoot"
Write-Host ("Thread counts: {0}" -f ($resolvedThreadCounts -join ', '))

if ($ValidateOnly) {
    Write-Host 'Validation only mode enabled.'
    foreach ($threadCount in $resolvedThreadCounts) {
        $threadOutputDir = Join-Path $stagingOutputRoot ("threads_{0}" -f $threadCount)
        Write-Host ("Planned output directory: {0}" -f $threadOutputDir)
    }
    return
}

New-Item -ItemType Directory -Path $stagingOutputRoot -Force | Out-Null

try {
    foreach ($threadCount in $resolvedThreadCounts) {
        $threadOutputDir = Join-Path $stagingOutputRoot ("threads_{0}" -f $threadCount)

        Write-Host '========================================'
        Write-Host ("Running sweep for thread count: {0}" -f $threadCount)
        Write-Host '========================================'

        Invoke-JuliaScript -Executable $resolvedJuliaExe -ProjectRoot $repoRoot -ScriptPath $resolvedGenerateJuliaScript -OutputArgument $threadOutputDir -Description ("Phase 1 export for {0} thread(s)" -f $threadCount) -RequestedThreadCount $threadCount -ExtraScriptArgs $phase1ExtraArgs -UseThreadCount

        Invoke-DynarePhase -ScriptPath $resolvedDynareScript -ThreadOutputDir $threadOutputDir -RequestedThreadCount $threadCount -PreferredDynareMatlabPath $DynareMatlabPath -PreferredMatlabExe $MatlabExe -RequestedMaxLicenseRetries $MaxLicenseRetries -RequestedLicenseRetryDelaySeconds $LicenseRetryDelaySeconds -RequestedOnlyModels $resolvedOnlyModels

        Invoke-JuliaScript -Executable $resolvedJuliaExe -ProjectRoot $repoRoot -ScriptPath $resolvedCompareScript -OutputArgument $threadOutputDir -Description ("Phase 3 compare for {0} thread(s)" -f $threadCount) -RequestedThreadCount $threadCount -UseThreadCount
    }

    Invoke-JuliaScript -Executable $resolvedJuliaExe -ProjectRoot $repoRoot -ScriptPath $resolvedSweepCompareScript -OutputArgument $stagingOutputRoot -Description 'Cross-thread benchmark summary' -RequestedThreadCount 1

    Publish-StagedOutputRoot -StageOutputRoot $stagingOutputRoot -FinalOutputRoot $resolvedOutputRoot -ResolvedOutputParent $resolvedOutputParent -OutputRootLeaf $outputRootLeaf
}
catch {
    if (Test-Path -LiteralPath $stagingOutputRoot) {
        Write-Warning ("Keeping staged sweep output for inspection: {0}" -f $stagingOutputRoot)
    }
    throw
}

Write-Host 'Thread sweep complete.'