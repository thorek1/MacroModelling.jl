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
        [switch]$UseThreadCount
    )

    $juliaArgs = @("--project=$ProjectRoot")
    if ($UseThreadCount) {
        $juliaArgs += "--threads=$RequestedThreadCount"
    }
    $juliaArgs += $ScriptPath
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
        [switch]$ValidationOnly
    )

    $dynareParameters = @{
        OutputDir = $ThreadOutputDir
        ThreadCount = $RequestedThreadCount
    }

    if ($PreferredDynareMatlabPath) {
        $dynareParameters.DynareMatlabPath = $PreferredDynareMatlabPath
    }
    if ($PreferredMatlabExe) {
        $dynareParameters.MatlabExe = $PreferredMatlabExe
    }
    if ($ValidationOnly) {
        $dynareParameters.ValidateOnly = $true
    }

    Write-Host "Running Dynare step for $RequestedThreadCount thread(s)"

    & $ScriptPath @dynareParameters

    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Dynare step failed for $RequestedThreadCount thread(s) with exit code $LASTEXITCODE"
    }
}

$resolvedOutputRoot = $OutputRoot
$outputRootExists = Test-Path -LiteralPath $resolvedOutputRoot
if (-not $outputRootExists) {
    New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
}
$resolvedOutputRoot = (Resolve-Path -LiteralPath $resolvedOutputRoot).Path
$resolvedJuliaExe = Get-JuliaExecutable -PreferredPath $JuliaExe
$resolvedGenerateJuliaScript = Resolve-ExistingPath -Candidates @($GenerateJuliaScript) -Description 'Julia phase-1 script'
$resolvedDynareScript = Resolve-ExistingPath -Candidates @($DynareScript) -Description 'Dynare phase-2 script'
$resolvedCompareScript = Resolve-ExistingPath -Candidates @($CompareScript) -Description 'Julia phase-3 script'
$resolvedSweepCompareScript = Resolve-ExistingPath -Candidates @($SweepCompareScript) -Description 'thread-sweep summary script'

$resolvedThreadCounts = $ThreadCounts | Sort-Object -Unique
if (-not $resolvedThreadCounts) {
    throw 'At least one thread count must be provided.'
}

New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null

Write-Host "Repository root: $repoRoot"
Write-Host "Julia executable: $resolvedJuliaExe"
Write-Host "Sweep output root: $resolvedOutputRoot"
Write-Host ("Thread counts: {0}" -f ($resolvedThreadCounts -join ', '))

if ($ValidateOnly) {
    Write-Host 'Validation only mode enabled.'
    foreach ($threadCount in $resolvedThreadCounts) {
        $threadOutputDir = Join-Path $resolvedOutputRoot ("threads_{0}" -f $threadCount)
        Write-Host ("Planned output directory: {0}" -f $threadOutputDir)
    }
    return
}

foreach ($threadCount in $resolvedThreadCounts) {
    $threadOutputDir = Join-Path $resolvedOutputRoot ("threads_{0}" -f $threadCount)

    Write-Host '========================================'
    Write-Host ("Running sweep for thread count: {0}" -f $threadCount)
    Write-Host '========================================'

    Invoke-JuliaScript -Executable $resolvedJuliaExe -ProjectRoot $repoRoot -ScriptPath $resolvedGenerateJuliaScript -OutputArgument $threadOutputDir -Description ("Phase 1 export for {0} thread(s)" -f $threadCount) -RequestedThreadCount $threadCount -UseThreadCount

    Invoke-DynarePhase -ScriptPath $resolvedDynareScript -ThreadOutputDir $threadOutputDir -RequestedThreadCount $threadCount -PreferredDynareMatlabPath $DynareMatlabPath -PreferredMatlabExe $MatlabExe

    Invoke-JuliaScript -Executable $resolvedJuliaExe -ProjectRoot $repoRoot -ScriptPath $resolvedCompareScript -OutputArgument $threadOutputDir -Description ("Phase 3 compare for {0} thread(s)" -f $threadCount) -RequestedThreadCount $threadCount -UseThreadCount
}

Invoke-JuliaScript -Executable $resolvedJuliaExe -ProjectRoot $repoRoot -ScriptPath $resolvedSweepCompareScript -OutputArgument $resolvedOutputRoot -Description 'Cross-thread benchmark summary' -RequestedThreadCount 1

Write-Host 'Thread sweep complete.'