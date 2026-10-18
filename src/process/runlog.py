"""Per-experiment report: config used, seed, git commit, attempts/players after QC, runtime."""
import subprocess
import time

import yaml

from config import ROOT, paths


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return sha + (" (uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0


def write_report(exp_id: str, cfg: dict, seed: int, n_attempts: int, n_players: int, runtime_s: float, body: str = "") -> None:
    out = paths()["outputs"] / exp_id
    out.mkdir(parents=True, exist_ok=True)
    text = (f"# {exp_id} report\n\n- Git commit: {git_commit()}\n- Seed: {seed}\n"
            f"- Attempts after QC: {n_attempts}; players: {n_players}\n- Runtime: {runtime_s:.1f} s\n\n"
            f"## Config\n\n```yaml\n{yaml.safe_dump(cfg, sort_keys=False)}```\n\n{body}\n")
    (out / "report.md").write_text(text)
