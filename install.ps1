# MLRift Self-Hosted Compiler Installer for Windows
# Run: irm https://raw.githubusercontent.com/Pantelis23/MLRift/main/install.ps1 | iex

$ErrorActionPreference = "Stop"
$Repo = "Pantelis23/MLRift"
$InstallDir = "$env:LOCALAPPDATA\MLRift\bin"

Write-Host "=== MLRift Self-Hosted Compiler Installer ==="

# Architecture detection
$arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
$ArchName = switch ($arch) {
    "AMD64" { "x86_64" }
    "ARM64" { "arm64" }
    default {
        Write-Host "error: unsupported architecture: $arch" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Platform: windows $ArchName"
Write-Host "Install to: $InstallDir"
Write-Host ""

# Create install directory
if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# Download mlrc compiler from GitHub releases
$BinaryName = "mlrc-windows-$ArchName.exe"
$Url = "https://github.com/$Repo/releases/latest/download/$BinaryName"
$Dest = "$InstallDir\mlrc.exe"

Write-Host "Downloading $BinaryName..."
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
} catch {
    Write-Host "error: download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual install: download from https://github.com/$Repo/releases"
    exit 1
}

# Download mlr runner
$MlrBinaryName = "mlr-windows-$ArchName.exe"
$MlrUrl = "https://github.com/$Repo/releases/latest/download/$MlrBinaryName"
$MlrDest = "$InstallDir\mlr.exe"
Write-Host "Downloading $MlrBinaryName..."
try {
    Invoke-WebRequest -Uri $MlrUrl -OutFile $MlrDest -UseBasicParsing
} catch {
    Write-Host "  warning: could not download mlr runner" -ForegroundColor Yellow
}

# Download standard library
$StdDir = "$env:LOCALAPPDATA\MLRift\std"
if (!(Test-Path $StdDir)) {
    New-Item -ItemType Directory -Path $StdDir -Force | Out-Null
}
Write-Host "Installing standard library..."
# Enumerate every std/*.mlr in the repo via the GitHub contents API so
# newly-added modules ship without an installer change. Falls back to a
# minimal core set on API errors (rate-limit or offline).
$Mods = $null
try {
    $listing = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/contents/std?ref=main" -UseBasicParsing
    $Mods = $listing | Where-Object { $_.name -match '\.mlr$' } | ForEach-Object { $_.name -replace '\.mlr$','' }
} catch {
    Write-Host "  warning: std/ listing failed, using minimal core fallback" -ForegroundColor Yellow
    $Mods = @("alloc","color","fb","fixedpoint","fmt","font","io","log","map","math","mem","memfast","net","string","time","vec","widget")
}
$modCount = 0
foreach ($mod in $Mods) {
    $modUrl = "https://raw.githubusercontent.com/$Repo/main/std/$mod.mlr"
    try {
        Invoke-WebRequest -Uri $modUrl -OutFile "$StdDir\$mod.mlr" -UseBasicParsing
        $modCount++
    } catch {
        Write-Host "  warning: could not download std/$mod.mlr" -ForegroundColor Yellow
    }
}
Write-Host "Standard library: $StdDir ($modCount modules)"

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$InstallDir;$UserPath", "User")
    Write-Host "Added $InstallDir to user PATH"
    Write-Host "Restart your terminal for PATH changes to take effect."
}

Write-Host ""
Write-Host "Installed: $Dest"
Write-Host ""
Write-Host "Usage:"
Write-Host "  mlrc --emit=pe program.mlr -o program.exe   # compile for Windows"
Write-Host "  mlrc --arch=x86_64 prog.mlr                 # native x86_64 ELF"
Write-Host "  mlrc program.mlr -o program.mlrbo           # fat binary (8 slices)"
Write-Host "  mlrc --version                              # show version"
Write-Host ""
Write-Host "=== Installation complete ==="
