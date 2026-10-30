"""E2 消融主矩阵编排 (thought/refine 分离, docs/experiment_plan_p1_p6.md §5)

配置: thought 3 级 (raw / ATGO γ=0 / ATGO+P1 γ=0.1) × 端到端 6 配置 (C1–C6)。
C4/C5/C6 复用 thought 产物 (内置复用映射 + 硬校验: refine 的 thought 输入目录
必须与映射表一致, 不一致即报错退出, 防手工误配导致配对污染)。

默认只执行 ATGO 预处理并写出命令清单 (manifest.json); 只有显式传入
`--run_llm` 才会调用 OpenAI-compatible API。`--pipeline` 支持
all (默认) / thought_only / refine_only (复用已产物)。

输入为全量 test 文件 (E2 判定对象是端到端 acc; 历史错误子集口径已弃用)。
六个预注册边际对比: C2vC1 (ATGO 本体) / C3vC2 (P1) / C4vC2 (P2) /
C5vC4 (P1 边际) / C5vC3 (P2 边际) / C6vC5 (P3)。
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


API_KEY_ENV = "OPENAI_API_KEY"
API_KEY_PLACEHOLDER = f"<env:{API_KEY_ENV}>"

P1_GAMMA = 0.1       # E3 的 γ 敏感性扫描另行传参 --gamma 覆盖
P3_BIAS_WEIGHT = 1.0


class ProviderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LLMProvider:
    name: str
    api_key_env: str
    base_url: str
    model_name: str
    openai_api_key: str

    def manifest(self):
        return {
            "name": self.name,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "model_name": self.model_name,
        }


# ===== E2 配置定义 (experiment_plan §5.1) =====

THOUGHT_LEVELS = {
    # level -> 预处理参数; enable_atgo=False 表示 raw (跳过 Stage 0.7, 直用原始表)
    "raw": {"enable_atgo": False, "gamma": 0.0, "inject_graph_hint": False},
    "atgo": {"enable_atgo": True, "gamma": 0.0, "inject_graph_hint": False},
    "atgo_p1": {"enable_atgo": True, "gamma": P1_GAMMA, "inject_graph_hint": False},
}

E2_CONFIGS = {
    # 端到端 6 配置; thought_source 指向 thought level (C4/C5/C6 复用, 不重跑 thought)
    "C1_raw": {"thought_level": "raw", "run_thought": True,
               "enable_p2": False, "enable_p3": False},
    "C2_atgo": {"thought_level": "atgo", "run_thought": True,
                "enable_p2": False, "enable_p3": False},
    "C3_p1": {"thought_level": "atgo_p1", "run_thought": True,
              "enable_p2": False, "enable_p3": False},
    "C4_p2": {"thought_level": "atgo", "run_thought": False,  # 复用②
              "enable_p2": True, "enable_p3": False},
    "C5_p1p2": {"thought_level": "atgo_p1", "run_thought": False,  # 复用③
                "enable_p2": True, "enable_p3": False},
    "C6_p1p2p3": {"thought_level": "atgo_p1", "run_thought": False,  # 复用③
                  "enable_p2": True, "enable_p3": True},
}

# 复用映射防呆: 配置 -> (期望 thought level, 供 refine_only 校验的来源配置名)
REUSE_MAP = {
    "C4_p2": ("atgo", "C2_atgo"),
    "C5_p1p2": ("atgo_p1", "C3_p1"),
    "C6_p1p2p3": ("atgo_p1", "C3_p1"),
}


DATASETS = {
    "wikitq": {
        "input": "thought/TableQA/data/wikitq/test_lower.jsonl",  # 全量口径 (T4)
        "thought_main": "thought/TableQA/main.py",
        "refine_main": "refine/TableQA/main_tree_based.py",
    },
    "tabfact": {
        "input": "thought/TableFV/data/tabfact/test.jsonl",  # 全量口径 (T4)
        "raw2clean": "thought/TableFV/data/tabfact/raw2clean.jsonl",
        "thought_main": "thought/TableFV/main.py",
        "refine_main": "refine/TableFV/main_tree_based.py",
    },
}


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def _redact_command(cmd):
    redacted = list(cmd)
    for index, part in enumerate(redacted[:-1]):
        if part == "--openai_api_key":
            redacted[index + 1] = "<redacted>"
    return redacted


def _llm_env(openai_api_key):
    env = os.environ.copy()
    if openai_api_key == API_KEY_PLACEHOLDER:
        openai_api_key = env.get(API_KEY_ENV, "")
    if openai_api_key:
        env[API_KEY_ENV] = openai_api_key
    return env


def _resolve_provider(config_path, provider_name):
    with open(config_path, "r", encoding="utf-8") as f:
        providers = json.load(f)

    for provider in providers:
        if provider.get("name") != provider_name:
            continue
        api_key_env = provider.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ProviderConfigError(f"Missing API key env: {api_key_env}")
        return LLMProvider(
            name=provider["name"],
            api_key_env=api_key_env,
            base_url=provider["base_url"],
            model_name=provider["model_name"],
            openai_api_key=api_key,
        )
    raise ProviderConfigError(f"Unknown LLM provider: {provider_name}")


def _run(cmd, dry_run=False, env=None):
    printable_cmd = _redact_command(cmd)
    print(" ".join(f'"{part}"' if " " in part else part for part in printable_cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _preprocess_cmd(python_bin, dataset_name, spec, level_name, exp_dir, gamma):
    """ATGO 预处理 (raw 级别返回 None: 跳过 Stage 0.7)。"""
    level = THOUGHT_LEVELS[level_name]
    if not level["enable_atgo"]:
        return None, Path(spec["input"])
    output_path = exp_dir / "data_atgo.jsonl"
    cache_dir = exp_dir / "atgo_cache"
    cmd = [
        python_bin,
        "preprocess_atgo.py",
        "--dataset_path",
        spec["input"],
        "--output_path",
        str(output_path),
        "--mode",
        "row",
        "--cache_dir",
        str(cache_dir),
        "--gamma",
        str(level["gamma"] if gamma is None else gamma),
        "--inject_graph_hint",
        _bool_arg(level["inject_graph_hint"]),
    ]
    return cmd, output_path


def _thought_cmd(python_bin, dataset_name, spec, data_path, exp_dir, args):
    thought_dir = exp_dir / "thought"
    cmd = [
        python_bin,
        spec["thought_main"],
        "--dataset_path",
        str(data_path),
        "--thought_results_dir",
        str(thought_dir),
        "--base_url",
        args.base_url,
        "--openai_api_key",
        API_KEY_PLACEHOLDER,
        "--model_name",
        args.model_name,
        "--first_n",
        str(args.first_n),
        "--n_proc",
        str(args.n_proc),
        "--chunk_size",
        str(args.chunk_size),
        "--use_clarifier",
        _bool_arg(args.use_clarifier),
    ]
    if dataset_name == "tabfact":
        cmd.extend(["--raw2clean_path", spec["raw2clean"]])
    return cmd, thought_dir


def _refine_cmd(python_bin, spec, thought_dir, exp_dir, config, args):
    """refine 命令: P2/P3 经 refine 级运行参数注入 (thought 产物复用安全)。"""
    cmd = [
        python_bin,
        spec["refine_main"],
        "--thought_results_dir",
        str(thought_dir),
        "--refine_results_dir",
        str(exp_dir / "refine"),
        "--base_url",
        args.base_url,
        "--openai_api_key",
        API_KEY_PLACEHOLDER,
        "--model_name",
        args.model_name,
        "--first_n",
        str(args.first_n),
        "--n_proc",
        str(args.n_proc),
        "--chunk_size",
        str(args.chunk_size),
        "--use_clarifier",
        _bool_arg(args.use_clarifier),
        "--enable_p2",
        _bool_arg(config["enable_p2"]),
        "--bias_weight",
        str(P3_BIAS_WEIGHT if config["enable_p3"] else 0.0),
    ]
    return cmd


def _validate_reuse(root, dataset_name, config_name, config, manifest_index, args):
    """复用映射硬校验 (experiment_plan §5.2 防呆): refine 的 thought 输入目录
    必须与映射表一致, 不一致即报错退出。"""
    expected_level, source_config = REUSE_MAP[config_name]
    if config["thought_level"] != expected_level:
        raise ValueError(
            f"[复用校验] {config_name} 的 thought_level={config['thought_level']} "
            f"与映射表期望 {expected_level} 不一致"
        )
    source_dir = root / dataset_name / source_config / "thought"
    pkl = source_dir / "final_result.pkl"
    # manifest 校验 (已跑过时): 来源配置的 thought level 必须匹配
    if manifest_index.get(source_config, {}).get("thought_level") not in (None, expected_level):
        raise ValueError(
            f"[复用校验] manifest 中 {source_config} 的 thought level "
            f"={manifest_index[source_config].get('thought_level')} ≠ {expected_level}"
        )
    if args.pipeline == "refine_only" and not pkl.exists():
        raise FileNotFoundError(
            f"[复用校验] refine_only 模式: {config_name} 复用 {source_config} 的 thought 产物 "
            f"缺失 ({pkl}); 先跑 --pipeline thought_only 或 all"
        )
    return source_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="results/graph_experiments")
    parser.add_argument("--python_bin", default="python")
    parser.add_argument("--pipeline", choices=["all", "thought_only", "refine_only"],
                        default="all")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(E2_CONFIGS),
        default=sorted(E2_CONFIGS),
    )
    parser.add_argument("--gamma", type=float, default=None,
                        help="覆盖 P1 gamma (E3 敏感性扫描: 0.05/0.2/0.3; None=P1_GAMMA 默认)")
    parser.add_argument("--first_n", type=int, default=-1)
    parser.add_argument("--n_proc", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--openai_api_key", default="EMPTY")
    parser.add_argument("--model_name", default="qwen3.5-4b")
    parser.add_argument("--llm_provider_config", default="")
    parser.add_argument("--llm_provider", default="")
    parser.add_argument("--use_clarifier", action="store_true")
    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    provider = None
    if args.llm_provider_config:
        provider = _resolve_provider(args.llm_provider_config, args.llm_provider)
        args.base_url = provider.base_url
        args.openai_api_key = provider.openai_api_key
        args.model_name = provider.model_name

    root = Path(args.output_root)
    manifest_path = root / "manifest.json"
    manifest_index = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        manifest_index = {e["config"]: e for e in prev if e.get("mode") == "E2"}

    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for config_name in args.configs:
            config = E2_CONFIGS[config_name]
            level = config["thought_level"]
            run_thought = config["run_thought"] and args.pipeline in ("all", "thought_only")
            run_refine = args.pipeline in ("all", "refine_only")

            # 复用映射硬校验 (C4/C5/C6)
            reuse_thought_dir = None
            if config_name in REUSE_MAP:
                reuse_thought_dir = _validate_reuse(
                    root, dataset_name, config_name, config, manifest_index, args
                )

            exp_dir = root / dataset_name / config_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            preprocess_cmd = None
            thought_cmd = None
            refine_cmd = None
            data_path = Path(spec["input"])
            thought_dir = None

            if run_thought:
                preprocess_cmd, data_path = _preprocess_cmd(
                    args.python_bin, dataset_name, spec, level, exp_dir, args.gamma
                )
                thought_cmd, thought_dir = _thought_cmd(
                    args.python_bin, dataset_name, spec, data_path, exp_dir, args
                )
            elif config_name in REUSE_MAP:
                thought_dir = reuse_thought_dir
                data_path = reuse_thought_dir.parent / "data_atgo.jsonl"
                if level != "raw" and not args.dry_run and not data_path.exists():
                    raise FileNotFoundError(
                        f"[复用校验] {config_name} 依赖的预处理数据缺失: {data_path}"
                    )
            elif run_refine:
                # 非 C4/C5/C6 的 refine_only (C1/C2/C3 单独 refine): 用本目录 thought
                thought_dir = exp_dir / "thought"
                data_path = exp_dir / "data_atgo.jsonl"
                if level != "raw" and not args.dry_run and not data_path.exists():
                    raise FileNotFoundError(
                        f"[复用校验] {config_name} refine_only 需先跑 thought: {data_path}"
                    )

            if run_refine:
                refine_cmd = _refine_cmd(
                    args.python_bin, spec, thought_dir, exp_dir, config, args
                )

            entry = {
                "mode": "E2",
                "dataset": dataset_name,
                "config": config_name,
                "thought_level": level,
                "enable_p2": config["enable_p2"],
                "enable_p3": config["enable_p3"],
                "reuses_thought_from": REUSE_MAP.get(config_name, [None])[1]
                if config_name in REUSE_MAP else None,
                "preprocessed_data": str(data_path),
                "thought_dir": str(thought_dir) if thought_dir else None,
                "refine_dir": str(exp_dir / "refine"),
                "thought_cmd": _redact_command(thought_cmd) if thought_cmd else None,
                "refine_cmd": _redact_command(refine_cmd) if refine_cmd else None,
                "preprocess_cmd": _redact_command(preprocess_cmd) if preprocess_cmd else None,
                "llm_provider": provider.manifest() if provider else None,
            }

            if args.run_llm:
                llm_env = _llm_env(args.openai_api_key)
                if preprocess_cmd:
                    _run(preprocess_cmd, dry_run=args.dry_run, env=llm_env)
                if thought_cmd:
                    _run(thought_cmd, dry_run=args.dry_run, env=llm_env)
                if refine_cmd:
                    _run(refine_cmd, dry_run=args.dry_run, env=llm_env)

            manifest_index[config_name] = entry
            print(f"[{dataset_name}/{config_name}] thought={level} "
                  f"P2={config['enable_p2']} P3={config['enable_p3']} "
                  f"reuse={entry['reuses_thought_from']}")

    # 合并写回 manifest (保留其他 mode 的条目)
    root.mkdir(parents=True, exist_ok=True)
    all_entries = []
    seen = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for e in json.load(f):
                if e.get("mode") == "E2" and e.get("config") in manifest_index:
                    continue
                all_entries.append(e)
                seen.add((e.get("mode"), e.get("config"), e.get("dataset")))
    for config_name, entry in manifest_index.items():
        if ("E2", config_name, entry.get("dataset")) not in seen:
            all_entries.append(entry)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"\nWrote manifest to {manifest_path}")
    print("预注册边际对比: C2vC1 | C3vC2 | C4vC2 | C5vC4 | C5vC3 | C6vC5")


if __name__ == "__main__":
    main()
