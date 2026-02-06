#!/usr/bin/env python3
"""Unified trainer entrypoint for all released methods."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return data


def _to_cli_args(args_dict: dict) -> list[str]:
    cli_args: list[str] = []
    for key, value in (args_dict or {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _resolve_config(repo_root: Path, registry: Path, config: str | None, method: str | None) -> Path:
    if config:
        cfg_path = (repo_root / config).resolve() if not Path(config).is_absolute() else Path(config).resolve()
        return cfg_path

    reg = _load_yaml(registry)
    methods = reg.get("methods", {})
    if method is None:
        method = reg.get("default_method")

    if method is None or method not in methods:
        available = ", ".join(sorted(methods.keys()))
        raise ValueError(f"Method not found. Available: {available}")

    return (repo_root / methods[method]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified training launcher")
    parser.add_argument("--registry", default="configs/PAPER_CONFIGS_FOR_RELEASE.yaml", help="Path to method registry")
    parser.add_argument("--config", default=None, help="Direct path to a method config YAML")
    parser.add_argument("--method", default=None, help="Method key in registry, e.g. dabi_net")
    parser.add_argument("--list", action="store_true", help="List available methods and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print launch command without starting training")

    args, passthrough = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    registry_path = (repo_root / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry).resolve()

    registry = _load_yaml(registry_path)

    if args.list:
        methods = registry.get("methods", {})
        print("Available methods:")
        for key in sorted(methods.keys()):
            print(f"  - {key}: {methods[key]}")
        return 0

    config_path = _resolve_config(repo_root, registry_path, args.config, args.method)
    config = _load_yaml(config_path)

    run_cfg = config.get("run", {})
    entrypoint = run_cfg.get("entrypoint")
    if not entrypoint:
        raise ValueError(f"Missing run.entrypoint in {config_path}")

    script_path = (repo_root / entrypoint).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Entrypoint script not found: {script_path}")

    base_args = _to_cli_args(run_cfg.get("args", {}))
    cmd = [sys.executable, str(script_path), *base_args, *passthrough]

    method_name = config.get("method", {}).get("name", script_path.parent.name)
    print(f"[launcher] method: {method_name}")
    print(f"[launcher] config: {config_path.relative_to(repo_root)}")
    print("[launcher] command:")
    print(" ", " ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
