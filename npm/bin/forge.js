#!/usr/bin/env node

/**
 * Forge CLI npm wrapper.
 * Enables running Forge globally via `npx @kirtan0902/forge` or `npx github:KIRTAN0902/forge`.
 */

const { spawn, spawnSync } = require("child_process");
const path = require("path");

function resolveRealForgeExecutable() {
  try {
    const isWin = process.platform === "win32";
    const checkCmd = isWin ? "where.exe" : "which";
    const res = spawnSync(checkCmd, ["forge"], { encoding: "utf8" });
    if (res.status === 0 && res.stdout) {
      const lines = res.stdout.trim().split(/\r?\n/);
      for (const line of lines) {
        const trimmed = line.trim();
        // Ignore npm shims, nodejs directory, and this wrapper itself
        if (
          !trimmed.includes("node_modules") &&
          !trimmed.includes("_npx") &&
          !trimmed.toLowerCase().includes("nodejs") &&
          !trimmed.endsWith("forge.js") &&
          !trimmed.endsWith("forge.cmd") &&
          !trimmed.endsWith("forge.ps1")
        ) {
          return trimmed;
        }
      }
    }
  } catch {
    return null;
  }
  return null;
}

function resolveExecutable(cmd) {
  try {
    const isWin = process.platform === "win32";
    const checkCmd = isWin ? "where.exe" : "which";
    const res = spawnSync(checkCmd, [cmd], { encoding: "utf8" });
    if (res.status === 0 && res.stdout) {
      const lines = res.stdout.trim().split(/\r?\n/);
      return lines[0] ? lines[0].trim() : null;
    }
  } catch {
    return null;
  }
  return null;
}

function runCommand(command, args) {
  const isWin = process.platform === "win32";
  const child = spawn(command, args, {
    stdio: "inherit",
    shell: isWin && !command.endsWith(".exe"),
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code || 0);
    }
  });

  process.on("SIGINT", () => child.kill("SIGINT"));
  process.on("SIGTERM", () => child.kill("SIGTERM"));
}

function main() {
  const args = process.argv.slice(2);

  // 1. If real forge binary is installed on system (via uv tool install or pipx), run it
  const realForge = resolveRealForgeExecutable();
  if (realForge) {
    runCommand(realForge, args);
    return;
  }

  // 2. If uvx / uv is available, execute through uv with --refresh to always get latest main
  const uvxExe = resolveExecutable("uvx");
  if (uvxExe) {
    runCommand(uvxExe, ["--from", "git+https://github.com/KIRTAN0902/forge.git@main", "--refresh", "forge", ...args]);
    return;
  }

  const uvExe = resolveExecutable("uv");
  if (uvExe) {
    runCommand(uvExe, ["tool", "run", "--from", "git+https://github.com/KIRTAN0902/forge.git@main", "--refresh", "forge", ...args]);
    return;
  }

  // 3. Fallback to pipx if available
  const pipxExe = resolveExecutable("pipx");
  if (pipxExe) {
    runCommand(pipxExe, ["run", "--spec", "git+https://github.com/KIRTAN0902/forge.git", "forge", ...args]);
    return;
  }

  // 4. Prompt user to install uv
  console.error("\x1b[31m[forge]\x1b[0m Forge requires Python 3.11+ and uv to run.");
  console.error("Install uv quickly:\n");
  if (process.platform === "win32") {
    console.error("  powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n");
  } else {
    console.error("  curl -LsSf https://astral.sh/uv/install.sh | sh\n");
  }
  console.error("Then run: uv tool install git+https://github.com/KIRTAN0902/forge.git");
  process.exit(1);
}

main();
