"""Run the row-ATGO graph experiment matrix.

默认只执行 ATGO 预处理并写出 Thought/Refine 命令清单；只有显式传入
`--run_llm` 才会调用 OpenAI-compatible API。
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


API_KEY_ENV = "OPENAI_API_KEY"
API_KEY_PLACEHOLDER = f"<env:{API_KEY_ENV}>"


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


EXPERIMENTS = [
    {
        "name": "baseline",
        "gamma": 0.0,
        "inject_graph_hint": False,
        "bias_weight": 0.0,
        "run_refine": True,
    },
    {
        "name": "P1",
        "gamma": 0.1,
        "inject_graph_hint": False,
        "bias_weight": 0.0,
        "run_refine": False,
    },
    {
        "name": "P2",
        "gamma": 0.0,
        "inject_graph_hint": True,
        "bias_weight": 0.0,
        "run_refine": True,
    },
    {
        "name": "P3",
        "gamma": 0.0,
        "inject_graph_hint": False,
        "bias_weight": 1.0,
        "run_refine": False,
    },
    {
        "name": "P1+P2",
        "gamma": 0.1,
        "inject_graph_hint": True,
        "bias_weight": 0.0,
        "run_refine": True,
    },
    {
        "name": "P1+P3",
        "gamma": 0.1,
        "inject_graph_hint": False,
        "bias_weight": 1.0,
        "run_refine": False,
    },
    {
        "name": "P1+P2+P3",
        "gamma": 0.1,
        "inject_graph_hint": True,
        "bias_weight": 1.0,
        "run_refine": True,
        "reuse_thought_from": "P1+P3",
    },
]


DATASETS = {
    "wikitq": {
        "input": "thought/TableQA/data/wikitq/error_data.jsonl",
        "thought_main": "thought/TableQA/main.py",
        "refine_main": "refine/TableQA/main_tree_based.py",
    },
    "tabfact": {
        "input": "thought/TableFV/data/tabfact/error_data.jsonl",
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


def _preprocess_cmd(python_bin, dataset_name, spec, exp, exp_dir):
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
        str(exp["gamma"]),
        "--inject_graph_hint",
        _bool_arg(exp["inject_graph_hint"]),
        "--bias_weight",
        str(exp["bias_weight"]),
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


def _refine_cmd(python_bin, spec, thought_dir, exp_dir, args):
    return [
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
    ]


def _exp_index():
    return {exp["name"]: exp for exp in EXPERIMENTS}


def _resolve_reuse_thought_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid reuse_thought format: {item}")
        experiment, source = item.split("=", 1)
        if not experiment or not source:
            raise ValueError(f"Invalid reuse_thought format: {item}")
        overrides[experiment] = source
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="results/graph_experiments")
    parser.add_argument("--python_bin", default="/home/cocopink/venv/dstc/bin/python")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[exp["name"] for exp in EXPERIMENTS],
        default=[exp["name"] for exp in EXPERIMENTS],
    )
    parser.add_argument("--first_n", type=int, default=-1)
    parser.add_argument("--n_proc", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--base_url", default="http://localhost:11434/v1")
    parser.add_argument("--openai_api_key", default="EMPTY")
    parser.add_argument("--model_name", default="qwen3:14b")
    parser.add_argument("--llm_provider_config", default="")
    parser.add_argument("--llm_provider", default="")
    parser.add_argument("--reuse_thought", nargs="*", default=[])
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
    manifest = []
    exp_by_name = _exp_index()
    reuse_thought_overrides = _resolve_reuse_thought_overrides(args.reuse_thought)
    selected_experiments = [dict(exp_by_name[name]) for name in args.experiments]

    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for exp in selected_experiments:
            exp_dir = root / dataset_name / exp["name"]
            exp_dir.mkdir(parents=True, exist_ok=True)

            reuse_thought_from = reuse_thought_overrides.get(
                exp["name"], exp.get("reuse_thought_from")
            )
            if reuse_thought_from:
                source_exp_dir = root / dataset_name / reuse_thought_from
                thought_dir = source_exp_dir / "thought"
                data_path = source_exp_dir / "data_atgo.jsonl"
                thought_cmd = None
            else:
                preprocess_cmd, data_path = _preprocess_cmd(
                    args.python_bin,
                    dataset_name,
                    spec,
                    exp,
                    exp_dir,
                )
                _run(preprocess_cmd, dry_run=args.dry_run)

                thought_cmd, thought_dir = _thought_cmd(
                    args.python_bin,
                    dataset_name,
                    spec,
                    data_path,
                    exp_dir,
                    args,
                )

            refine_cmd = None
            if exp["run_refine"]:
                refine_cmd = _refine_cmd(args.python_bin, spec, thought_dir, exp_dir, args)

            entry = {
                "dataset": dataset_name,
                "experiment": exp["name"],
                "parameters": exp,
                "reuses_thought_from": reuse_thought_from,
                "preprocessed_data": str(data_path),
                "thought_dir": str(thought_dir),
                "refine_dir": str(exp_dir / "refine"),
                "thought_cmd": _redact_command(thought_cmd) if thought_cmd is not None else None,
                "refine_cmd": _redact_command(refine_cmd) if refine_cmd is not None else None,
                "llm_provider": provider.manifest() if provider else None,
                "uses_atgo_only": True,
            }
            manifest.append(entry)

            if args.run_llm:
                llm_env = _llm_env(args.openai_api_key)
                if thought_cmd is not None:
                    _run(thought_cmd, dry_run=args.dry_run, env=llm_env)
                if refine_cmd is not None:
                    _run(refine_cmd, dry_run=args.dry_run, env=llm_env)

    manifest_path = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
