#!/usr/bin/env node

/**
 * Forge CLI npm wrapper.
 * Enables running Forge globally via `npx github:KIRTAN0902/forge` or `npx @kirtan0902/forge`.
 */

const { spawn, spawnSync } = require("child_process");

function resolveExecutable(cmd) {
  try {
    const isWin = process.platform === "win32";
    const checkCmd = isWin ? "where.exe" : "which";
    const res = spawnSync(checkCmd, [cmd], { encoding: "utf8" });
    if (res.status === 0 && res.stdout) {
      const lines = res.stdout.trim().split(/\r?\n/);
      return lines[0] || null;
    }
  } catch {
    return null;
  }
  return null;
}

function runCommand(command, args) {
  const exe = resolveExecutable(command) || command;
  const child = spawn(exe, args, {
    stdio: "inherit",
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

  // 1. If forge binary is already installed on PATH, run it directly
  if (resolveExecutable("forge")) {
    runCommand("forge", args);
    return;
  }

  // 2. If uvx / uv is available, run directly from GitHub repository
  if (resolveExecutable("uvx")) {
    runCommand("uvx", ["--from", "git+https://github.com/KIRTAN0902/forge.git", "forge", ...args]);
    return;
  }

  if (resolveExecutable("uv")) {
    runCommand("uv", ["tool", "run", "--from", "git+https://github.com/KIRTAN0902/forge.git", "forge", ...args]);
    return;
  }

  // 3. Fallback to pipx if available
  if (resolveExecutable("pipx")) {
    runCommand("pipx", ["run", "--spec", "git+https://github.com/KIRTAN0902/forge.git", "forge", ...args]);
    return;
  }

  // 4. Fallback: prompt user to install uv or python
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
