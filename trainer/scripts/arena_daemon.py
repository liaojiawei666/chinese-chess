#!/usr/bin/env python3
"""arena 守护：轮询 model_dir 版本，发现新版本就低优先级触发一场 arena（最近两版对杀）。

与训练彻底解耦——训练只管写 model_*.pt + latest.json，本守护独立轮询、慢慢评，不阻塞训练。
arena 是 batch=1 串行推理；默认跑 CPU（不抢训练/自对弈的 GPU）+ nice 降优先级 +
较小工作量（16 开局 / 64 模拟，排序够用），可用 --every-versions 进一步节流。

用法：
    uv run python scripts/arena_daemon.py --checkpoints-only
    uv run python scripts/arena_daemon.py --config ../../config/gpu.json --checkpoints-only
    uv run python scripts/arena_daemon.py --once --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]

from trainer.config import default_config_path, load_config  # noqa: E402

logger = logging.getLogger(__name__)

_MODEL_RE = re.compile(r"^model_(\d{6})\.pt$")


def list_versions(model_dir: Path) -> list[int]:
    if not model_dir.is_dir():
        return []
    versions = []
    for entry in model_dir.iterdir():
        m = _MODEL_RE.match(entry.name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(versions)


def eligible_pair(
    versions: list[int], *, checkpoints_only: bool, checkpoint_every: int
) -> tuple[int, int] | None:
    if checkpoints_only:
        if checkpoint_every <= 0:
            return None
        pool = [v for v in versions if v > 0 and v % checkpoint_every == 0]
    else:
        pool = versions
    if len(pool) < 2:
        return None
    return pool[-1], pool[-2]


def torch_lib_dir() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return os.path.join(os.path.dirname(torch.__file__), "lib")


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LIBTORCH_USE_PYTORCH"] = "1"
    lib = torch_lib_dir()
    if lib:
        for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[key] = lib + (os.pathsep + env[key] if env.get(key) else "")
    return env


def build_command(args: argparse.Namespace, run_config: Path, va: int, vb: int) -> list[str]:
    cmd = [
        "cargo", "run", "--release",
        "-p", "arena", "--",
        "--run-config", str(run_config),
        "--model-dir", args.model_dir_arg,
        "--version-a", str(va),
        "--version-b", str(vb),
        "--device", args.device,
        "--report", args.report,
        "--table", args.table,
        "--num-openings", str(args.num_openings),
    ]
    if args.sims is not None:
        cmd += ["--sims", str(args.sims)]
    return cmd


def run_arena(args: argparse.Namespace, run_config: Path, va: int, vb: int) -> None:
    cmd = build_command(args, run_config, va, vb)
    logger.info("触发对杀 A=v%d vs B=v%d：%s", va, vb, " ".join(cmd))
    if args.dry_run:
        return

    preexec = (lambda: os.nice(args.nice)) if hasattr(os, "nice") else None
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, env=build_env(), preexec_fn=preexec, check=False
    )
    if result.returncode != 0:
        logger.warning("arena 退出码 %d（本轮跳过）", result.returncode)
        return

    report = REPO_ROOT / args.report
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            logger.info(
                "结果 A=v%d vs B=v%d：得分率 %.3f | %+.1f Elo | %s胜 %s和 %s负",
                va, vb,
                data.get("score_a"), data.get("elo_diff_a_vs_b"),
                data.get("wins_a"), data.get("draws"), data.get("losses_a"),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径（默认 config/{CHESS_PROFILE}.json）",
    )
    p.add_argument("--poll-interval-s", type=float, default=30.0)
    p.add_argument("--checkpoints-only", action="store_true")
    p.add_argument("--every-versions", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-openings", type=int, default=16)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--report", default="data/arena/report.json")
    p.add_argument("--table", default="data/arena/table.csv")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    config_path = (args.config or default_config_path()).resolve()
    config = load_config(config_path)
    run_config = config_path

    args.model_dir_arg = config.datagen.model_dir
    model_dir = REPO_ROOT / config.datagen.model_dir
    if not run_config.exists():
        raise SystemExit(f"未找到配置文件：{run_config}")

    checkpoint_every = config.datagen.checkpoint_every
    mode = f"checkpoints-only(every {checkpoint_every} 步)" if args.checkpoints_only else "每新版本"
    logger.info(
        "config=%s model_dir=%s poll=%.1fs mode=%s every_versions=%d nice=+%d",
        run_config, model_dir, args.poll_interval_s, mode,
        args.every_versions, args.nice,
    )

    def pick() -> tuple[int, int] | None:
        return eligible_pair(
            list_versions(model_dir),
            checkpoints_only=args.checkpoints_only,
            checkpoint_every=checkpoint_every,
        )

    if args.once:
        pair = pick()
        if pair is None:
            raise SystemExit(f"{model_dir} 候选不足两个（{mode}），无法对杀")
        run_arena(args, run_config, *pair)
        return

    start = pick()
    last_evaluated = start[0] if start else -1

    while True:
        pair = pick()
        if pair is not None:
            newest, baseline = pair
            new_count = sum(1 for v in list_versions(model_dir) if v > last_evaluated)
            throttled = (not args.checkpoints_only) and new_count < args.every_versions
            if newest > last_evaluated and not throttled:
                run_arena(args, run_config, newest, baseline)
                last_evaluated = newest
        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()
