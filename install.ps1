# Forge CLI Installer for Windows PowerShell
# Usage:
#   irm https://raw.githubusercontent.com/KIRTAN0902/forge/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       Installing Forge CLI...          " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Ensure uv is installed
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "[*] uv not found. Installing Astral uv (ultra-fast Python package runner)..." -ForegroundColor Yellow
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Write-Error "Failed to install uv automatically. Please install it from https://astral.sh/uv and re-run."
        exit 1
    }

    # Refresh PATH for current session
    $userPaths = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePaths = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:Path = "$userPaths;$machinePaths"
    
    $localBin = "$env:USERPROFILE\.local\bin"
    $cargoBin = "$env:USERPROFILE\.cargo\bin"
    if (Test-Path $localBin -and -not ($env:Path -like "*$localBin*")) {
        $env:Path = "$localBin;$env:Path"
    }
    if (Test-Path $cargoBin -and -not ($env:Path -like "*$cargoBin*")) {
        $env:Path = "$cargoBin;$env:Path"
    }
}

# Verify uv is now available
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    # Check fallback path
    $fallbackUv = "$env:USERPROFILE\.local\bin\uv.exe"
    $fallbackCargoUv = "$env:USERPROFILE\.cargo\bin\uv.exe"
    if (Test-Path $fallbackUv) {
        $uvExe = $fallbackUv
    } elseif (Test-Path $fallbackCargoUv) {
        $uvExe = $fallbackCargoUv
    } else {
        Write-Error "uv was installed but not found on PATH. Please restart PowerShell and run again."
        exit 1
    }
} else {
    $uvExe = "uv"
}

Write-Host "[*] Installing Forge from GitHub main..." -ForegroundColor Green
& $uvExe tool install --force git+https://github.com/KIRTAN0902/forge.git

# Verify forge executable
$forgeCmd = Get-Command forge -ErrorAction SilentlyContinue
if (-not $forgeCmd) {
    $localBin = "$env:USERPROFILE\.local\bin"
    if (Test-Path "$localBin\forge.exe") {
        $env:Path = "$localBin;$env:Path"
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Forge CLI was successfully installed! " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start commands:" -ForegroundColor White
Write-Host "  forge chat            Start the interactive coding agent" -ForegroundColor Cyan
Write-Host "  forge login           Set up your API key (NVIDIA, OpenRouter, etc.)" -ForegroundColor Cyan
Write-Host "  forge roles list      View role assignments" -ForegroundColor Cyan
Write-Host "  forge update          Pull and install the latest commit from GitHub" -ForegroundColor Cyan
Write-Host ""
