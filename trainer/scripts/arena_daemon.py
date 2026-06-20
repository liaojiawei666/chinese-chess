#!/usr/bin/env python3
"""arena 守护：轮询 model_dir 版本，发现新版本就低优先级触发一场 arena（最近两版对杀）。

与训练彻底解耦——训练只管写 model_*.pt + latest.json，本守护独立轮询、慢慢评，不阻塞训练。
arena 是 batch=1 串行推理；默认跑 CPU（不抢训练/自对弈的 GPU）+ nice 降优先级 +
较小工作量（16 开局 / 64 模拟，排序够用），可用 --every-versions 进一步节流。

依赖：torch 特性的 arena 二进制只链接 venv 里的 Python torch 自带 libtorch，故本脚本须在
venv 下运行（uv run）。它会自动把同一份 venv torch/lib 注入子进程的动态库搜索路径。

用法：
    uv run python scripts/arena_daemon.py --checkpoints-only  # 推荐：每个新 checkpoint 评一次
    uv run python scripts/arena_daemon.py                      # 每个新版本都评（最近两版）
    uv run python scripts/arena_daemon.py --every-versions 5   # 普通模式节流：每 5 个新版本评一次
    uv run python scripts/arena_daemon.py --once               # 只评一次当前最近两版后退出
    uv run python scripts/arena_daemon.py --dry-run --once     # 只打印将执行的命令，不真跑
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 仓库根：trainer/scripts/arena_daemon.py 的上两级；用来锚定 data/ 相对路径与 datagen 清单。
REPO_ROOT = Path(__file__).resolve().parents[2]

from trainer.config import (  # noqa: E402
    DEFAULT_PROFILE,
    PROFILE_ENV_VAR,
    PROFILES,
)

_MODEL_RE = re.compile(r"^model_(\d{6})\.pt$")


def list_versions(model_dir: Path) -> list[int]:
    """列出目录下 model_{step:06d}.pt 的版本号（升序）。目录不存在则空。"""
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
    """挑出本轮要对杀的 (A=最新, B=基线)；不足两个候选则 None。

    - 普通模式：最新版本 vs 上一版本（相邻，间隔 = model_export_interval）。
    - checkpoint 模式：最新 checkpoint vs 上一 checkpoint（间隔 = checkpoint_every，
      信号更干净；keep_checkpoints≥2 保证基线仍在盘上）。version 0 是初始权重，排除。
    """
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
    """唯一 libtorch 来源的 lib 目录（注入子进程动态库搜索路径用）。"""
    try:
        import torch
    except ImportError:
        return None
    return os.path.join(os.path.dirname(torch.__file__), "lib")


def build_env() -> dict[str, str]:
    """给 arena 子进程准备复用 venv libtorch 所需的环境变量。"""
    env = os.environ.copy()
    env["LIBTORCH_USE_PYTORCH"] = "1"
    lib = torch_lib_dir()
    if lib:
        # macOS 用 DYLD_*、Linux 用 LD_*；都设上，无害且省得分平台判断。
        for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[key] = lib + (os.pathsep + env[key] if env.get(key) else "")
    return env


def build_command(args: argparse.Namespace, run_config: Path, va: int, vb: int) -> list[str]:
    cmd = [
        "cargo", "run", "--manifest-path", "datagen/Cargo.toml", "--release",
        "-p", "arena", "--features", "torch", "--",
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
    print(f"[arena-daemon] 触发对杀 A=v{va} vs B=v{vb}：{' '.join(cmd)}", flush=True)
    if args.dry_run:
        return

    # nice 降优先级：arena 抢同一张卡时尽量让训练/selfplay 优先。
    preexec = (lambda: os.nice(args.nice)) if hasattr(os, "nice") else None
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, env=build_env(), preexec_fn=preexec, check=False
    )
    if result.returncode != 0:
        print(f"[arena-daemon] arena 退出码 {result.returncode}（本轮跳过）", flush=True)
        return

    report = REPO_ROOT / args.report
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            print(
                f"[arena-daemon] 结果 A=v{va} vs B=v{vb}："
                f"得分率 {data.get('score_a'):.3f} | "
                f"{data.get('elo_diff_a_vs_b'):+.1f} Elo | "
                f"{data.get('wins_a')}胜 {data.get('draws')}和 {data.get('losses_a')}负",
                flush=True,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poll-interval-s", type=float, default=30.0, help="轮询间隔秒（默认 30）")
    p.add_argument(
        "--checkpoints-only", action="store_true",
        help="只在出现新 checkpoint（每 checkpoint_every 步）时评，对杀最新 vs 上一 checkpoint",
    )
    p.add_argument(
        "--every-versions", type=int, default=1,
        help="普通模式节流：每攒够这么多个新版本才评一次（默认 1；--checkpoints-only 时忽略）",
    )
    p.add_argument(
        "--device", default="cpu",
        help="对杀设备（默认 cpu：不抢训练/自对弈的 GPU；要快可设 cuda）",
    )
    p.add_argument("--num-openings", type=int, default=16, help="开局数，总局数=N×2（默认 16）")
    p.add_argument(
        "--sims", type=int, default=64,
        help="每手模拟数（默认 64；纯排序够用，传更小更快）",
    )
    p.add_argument("--nice", type=int, default=10, help="子进程 nice 增量，越大优先级越低（默认 10）")
    p.add_argument("--report", default="data/arena/report.json", help="JSON 报告路径")
    p.add_argument("--table", default="data/arena/table.csv", help="CSV 战绩表路径（累积趋势）")
    p.add_argument("--once", action="store_true", help="只评一次当前最近两版后退出")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    profile = os.environ.get(PROFILE_ENV_VAR, DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise SystemExit(f"未知 {PROFILE_ENV_VAR}={profile!r}，可选：{sorted(PROFILES)}")
    config = PROFILES[profile]

    # arena 的 --model-dir 用相对路径（cwd=REPO_ROOT 时命中真实目录）；轮询用绝对路径。
    args.model_dir_arg = config.datagen.model_dir
    model_dir = REPO_ROOT / config.datagen.model_dir
    run_config = REPO_ROOT / f"data/config/run-config.{profile}.json"
    if not run_config.exists():
        raise SystemExit(
            f"未找到 {run_config}；请先 `just export-config` 或 "
            f"`uv run python scripts/export_run_config.py`"
        )

    checkpoint_every = config.datagen.checkpoint_every
    mode = f"checkpoints-only(every {checkpoint_every} 步)" if args.checkpoints_only else "每新版本"
    print(
        f"[arena-daemon] profile={profile} model_dir={model_dir} "
        f"poll={args.poll_interval_s}s mode={mode} "
        f"every_versions={args.every_versions} nice=+{args.nice}",
        flush=True,
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

    # 已评过的最高版本：守护启动时把现存最高候选当基线，避免一上来把历史版本全补评一遍。
    start = pick()
    last_evaluated = start[0] if start else -1

    while True:
        pair = pick()
        if pair is not None:
            newest, baseline = pair
            # 普通模式额外要求攒够 every_versions 个新版本；checkpoint 模式靠 checkpoint_every 节流。
            new_count = sum(1 for v in list_versions(model_dir) if v > last_evaluated)
            throttled = (not args.checkpoints_only) and new_count < args.every_versions
            if newest > last_evaluated and not throttled:
                run_arena(args, run_config, newest, baseline)
                last_evaluated = newest
        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()
