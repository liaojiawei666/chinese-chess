#!/usr/bin/env python3
"""把档位的 Config 导出成跨语言运行配置 data/config/run-config.<profile>.json。

trainer（Python）与 datagen（Rust）都读这些 JSON 作为运行期参数的唯一出口，
避免常量/超参在两侧各写一份导致漂移。每个档位导出到独立文件，互不覆盖，
便于 dev（local）/ prod（gpu）两份配置长期共存。

用法：
    python trainer/scripts/export_run_config.py                  # 导出所有档位
    python trainer/scripts/export_run_config.py --profile gpu    # 只导 gpu 档
    python trainer/scripts/export_run_config.py --profile local --out data/config/run-config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 `python trainer/scripts/export_run_config.py` 运行（把 src 加入 sys.path）。
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 仓库根目录（trainer/scripts/export_run_config.py 的上两级），
# 用来把 run_config_path 这类相对路径锚定到根，避免随调用方 cwd 漂移。
REPO_ROOT = Path(__file__).resolve().parents[2]

from trainer.config import (  # noqa: E402
    PROFILES,
    to_run_config,
)


def resolve_under_root(path: str | Path) -> Path:
    """相对路径锚定到仓库根，绝对路径原样返回。"""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def profile_out_path(base: str, profile: str) -> Path:
    """把基础路径 data/config/run-config.json 派生成按档位区分的
    data/config/run-config.<profile>.json（锚定到仓库根）。"""
    p = Path(base)
    return resolve_under_root(p.with_name(f"{p.stem}.{profile}{p.suffix}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=None,
        choices=sorted(PROFILES),
        help="运行档位（默认导出所有档位）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出路径（仅在指定单个 --profile 时可用；默认按档位派生）",
    )
    return parser.parse_args()


def export_one(profile: str, out_path: Path) -> None:
    config = PROFILES[profile]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_run_config(config, profile)

    # 原子写：先写 .tmp 再 rename，避免 datagen 读到半截文件。
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"wrote {out_path} (profile={profile})")


def main() -> None:
    args = parse_args()

    if args.out is not None and args.profile is None:
        raise SystemExit("--out 只能与单个 --profile 搭配使用")

    profiles = [args.profile] if args.profile is not None else sorted(PROFILES)
    for profile in profiles:
        out_path = (
            resolve_under_root(args.out)
            if args.out is not None
            else profile_out_path(PROFILES[profile].datagen.run_config_path, profile)
        )
        export_one(profile, out_path)


if __name__ == "__main__":
    main()
