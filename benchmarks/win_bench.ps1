# Windows PowerShell bench -- mlrc-only (no gcc; rustc skipped per release scope).
param(
    [string]$MLRC = ".\mlrc.exe",
    [string]$Results = ".\results-win.md",
    [string]$Platform = "Windows 11 / x86_64"
)
$ErrorActionPreference = "Continue"
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$os  = (Get-CimInstance Win32_OperatingSystem).Caption + " " + (Get-CimInstance Win32_OperatingSystem).Version

"# MLRift benchmark -- $Platform"          | Out-File $Results -Encoding ascii
""                                         | Out-File $Results -Append -Encoding ascii
"**Date:** $(Get-Date -Format 'u')"        | Out-File $Results -Append -Encoding ascii
"**Host:** $cpu"                           | Out-File $Results -Append -Encoding ascii
"**OS:** $os"                              | Out-File $Results -Append -Encoding ascii
"**Toolchains:** mlrc=yes (gcc/rustc skipped -- release scope)" | Out-File $Results -Append -Encoding ascii
""                                         | Out-File $Results -Append -Encoding ascii

function Compile-Ms($cmd, $argv) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $cmd @argv > $null 2>&1
    $rc = $LASTEXITCODE
    $sw.Stop()
    if ($rc -ne 0) { return "FAIL" }
    return [int]$sw.ElapsedMilliseconds
}

function Median3-RuntimeMs($bin) {
    $times = @()
    for ($i = 0; $i -lt 3; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $bin > $null 2>&1
        $sw.Stop()
        $times += [int]$sw.ElapsedMilliseconds
    }
    return ($times | Sort-Object)[1]
}

function Bench-One($name) {
    $mlr = ".\$name.mlr"
    $bin = ".\$name.mlrc.exe"
    $mlrc_compile = "N/A"; $mlrc_size = "N/A"; $mlrc_run = "N/A"
    if (Test-Path $mlr) {
        $mlrc_compile = Compile-Ms $MLRC @("--arch=x86_64", "--emit=pe", $mlr, "-o", $bin)
        if (Test-Path $bin) {
            $mlrc_size = (Get-Item $bin).Length
            $mlrc_run  = Median3-RuntimeMs $bin
        }
    }
    "## $name"                                                   | Out-File $Results -Append -Encoding ascii
    ""                                                            | Out-File $Results -Append -Encoding ascii
    "| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |" | Out-File $Results -Append -Encoding ascii
    "|---|---|---|---|"                                           | Out-File $Results -Append -Encoding ascii
    "| mlrc (self-hosted) | $mlrc_compile | $mlrc_size | $mlrc_run |" | Out-File $Results -Append -Encoding ascii
    ""                                                            | Out-File $Results -Append -Encoding ascii
    Remove-Item $bin -ErrorAction SilentlyContinue
}

foreach ($p in @("fib", "sort", "sieve", "matmul")) { Bench-One $p }
Write-Host "Done. Results: $Results"
Get-Content $Results
