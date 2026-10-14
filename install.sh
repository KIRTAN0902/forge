#!/usr/bin/env bash
# Forge CLI Installer for Linux and macOS
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/KIRTAN0902/forge/main/install.sh | bash

set -e

echo "\033[0;36m========================================\033[0m"
echo "\033[0;36m       Installing Forge CLI...          \033[0m"
echo "\033[0;36m========================================\033[0m"

# 1. Check or install uv
if ! command -v uv >/dev/null 2>&1; then
    echo "\033[0;33m[*] uv not found. Installing Astral uv (ultra-fast Python runner)...\033[0m"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        UV_BIN="$HOME/.local/bin/uv"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        UV_BIN="$HOME/.cargo/bin/uv"
    else
        echo "\033[0;31mError: uv was installed but could not be located in PATH.\033[0m"
        exit 1
    fi
else
    UV_BIN="uv"
fi

echo "\033[0;32m[*] Installing Forge from GitHub main...\033[0m"
"$UV_BIN" tool install --force git+https://github.com/KIRTAN0902/forge.git

export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "\033[0;32m========================================\033[0m"
echo "\033[0;32m  Forge CLI was successfully installed! \033[0m"
echo "\033[0;32m========================================\033[0m"
echo ""
echo "Quick start commands:"
echo "  forge chat            Start the interactive coding agent"
echo "  forge login           Set up your API key (NVIDIA, OpenRouter, etc.)"
echo "  forge roles list      View role assignments"
echo "  forge update          Pull and install the latest commit from GitHub"
echo ""
