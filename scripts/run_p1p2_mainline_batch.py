"""Batch runner for the mainline P1+P2 experiment."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.openai_batch_smoke_test import resolve_runtime_config
from scripts.openai_smoke_test import RuntimeConfig, mask_secret, run_smoke_test, validate_runtime_config


DEFAULT_LOGICAL_MODELS = ["glm-5.1"] # "qwen3.6-plus",
GPT54_FALLBACKS = [
    "gpt-5.4",
    "gpt-5.4-high",
    "gpt-5.4-medium",
    "gpt-5.4-low",
]
DATASET_SCRIPTS = {
    "tabfact": "run_FV.sh",
    "wikitq": "run_QA.sh",
}


@dataclass(frozen=True)
class ModelSelection:
    logical_model: str
    actual_model: str | None


@dataclass(frozen=True)
class SmokeRecord:
    model: str
    ok: bool
    detail: str
    latency_ms: float | None


@dataclass(frozen=True)
class RunRecord:
    logical_model: str
    actual_model: str
    dataset: str
    script: str
    return_code: int

    @property
    def ok(self) -> bool:
        return self.return_code == 0


def expand_model_fallbacks(logical_model: str) -> list[str]:
    if logical_model == "gpt-5.4":
        return list(GPT54_FALLBACKS)
    return [logical_model]


def select_model_from_fallbacks(
    logical_model: str,
    smoke_check: Callable[[str], bool],
) -> str | None:
    for candidate in expand_model_fallbacks(logical_model):
        if smoke_check(candidate):
            return candidate
    return None


def build_pipeline_env(
    base_env: dict[str, str],
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    mode: str,
    enable_p1: bool = True,
    enable_p2: bool = True,
    p1_gamma: float = 0.1,
) -> dict[str, str]:
    env = dict(base_env)
    env["MODEL_NAME"] = model_name
    env["MODE"] = mode
    env["ENABLE_P1"] = "true" if enable_p1 else "false"
    env["ENABLE_P2"] = "true" if enable_p2 else "false"
    env["P1_GAMMA"] = str(p1_gamma)
    env["HAOMIAO_URL"] = base_url
    env["HAOMIAO_AUTH_TOKEN"] = api_key
    env["HAOMIAO_AUTHEN_TOKEN"] = api_key
    env["OPENAI_API_KEY"] = api_key
    return env


def _build_runtime_config(
    *,
    base_url: str,
    api_key: str,
    provider_file: str,
    provider: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
) -> RuntimeConfig:
    return resolve_runtime_config(
        base_url=base_url,
        api_key=api_key,
        provider_file=provider_file,
        provider_name=provider,
        prompt=prompt,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        env=dict(os.environ),
    )


def _smoke_check_model(base_config: RuntimeConfig, model: str) -> SmokeRecord:
    config = dataclasses.replace(base_config, model=model)
    try:
        validate_runtime_config(config)
        result = run_smoke_test(config)
    except Exception as exc:  # noqa: BLE001 - batch runner must report provider failures
        return SmokeRecord(model=model, ok=False, detail=f"{type(exc).__name__}: {exc}", latency_ms=None)

    content = str(result.get("content") or "").replace("\n", " ").strip()
    return SmokeRecord(
        model=model,
        ok=True,
        detail=content or "<empty>",
        latency_ms=float(result["elapsed_ms"]),
    )


def _choose_model(logical_model: str, base_config: RuntimeConfig, printer: Callable[[str], None]) -> ModelSelection:
    def smoke_check(candidate: str) -> bool:
        record = _smoke_check_model(base_config, candidate)
        status = "PASS" if record.ok else "FAIL"
        latency = "-" if record.latency_ms is None else f"{record.latency_ms:.1f}ms"
        printer(f"[SMOKE] {logical_model} -> {candidate} [{status}] ({latency}) {record.detail}")
        return record.ok

    actual_model = select_model_from_fallbacks(logical_model, smoke_check)
    return ModelSelection(logical_model=logical_model, actual_model=actual_model)


def _run_dataset_script(
    dataset: str,
    logical_model: str,
    actual_model: str,
    *,
    base_url: str,
    api_key: str,
    mode: str,
    printer: Callable[[str], None],
) -> RunRecord:
    script_name = DATASET_SCRIPTS[dataset]
    env = build_pipeline_env(
        os.environ,
        base_url=base_url,
        api_key=api_key,
        model_name=actual_model,
        mode=mode,
    )
    command = ["bash", script_name]
    printer(f"[RUN] {dataset} model={actual_model} command={' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT_DIR, env=env, check=False)
    printer(f"[DONE] {dataset} model={actual_model} return_code={completed.returncode}")
    return RunRecord(
        logical_model=logical_model,
        actual_model=actual_model,
        dataset=dataset,
        script=script_name,
        return_code=completed.returncode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_LOGICAL_MODELS,
        help="Logical models to run. gpt-5.4 expands to its fallback family.",
    )
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="API key for the endpoint.")
    parser.add_argument(
        "--provider-file",
        default="configs/llm_providers.json",
        help="Provider config JSON file.",
    )
    parser.add_argument(
        "--provider",
        default="current-gpt54",
        help="Provider name to load from --provider-file.",
    )
    parser.add_argument(
        "--prompt",
        default="hello",
        help="Prompt used for the internal smoke-check step.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=2, help="Maximum tokens for smoke-check responses.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for smoke-check.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SCRIPTS),
        default=sorted(DATASET_SCRIPTS),
        help="Datasets to run.",
    )
    parser.add_argument(
        "--mode",
        choices=("orig", "new"),
        default="orig",
        help="Pipeline mode passed through to run_FV.sh and run_QA.sh.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base_config = _build_runtime_config(
        base_url=args.base_url,
        api_key=args.api_key,
        provider_file=args.provider_file,
        provider=args.provider,
        prompt=args.prompt,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    try:
        validate_runtime_config(dataclasses.replace(base_config, model=args.models[0] if args.models else ""))
    except (IndexError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = ROOT_DIR / "logs" / f"p1p2_mainline_batch-{args.mode}-{timestamp}.log"
    summary_path = ROOT_DIR / "results" / f"p1p2_mainline_batch-{args.mode}-{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        def printer(message: str) -> None:
            print(message)
            print(message, file=log_file)
            log_file.flush()

        printer("[INFO] P1+P2 mainline batch")
        printer(f"[INFO] mode      = {args.mode}")
        printer(f"[INFO] base_url = {base_config.base_url}")
        printer(f"[INFO] api_key   = {mask_secret(base_config.api_key)}")
        printer(f"[INFO] models    = {', '.join(args.models)}")
        printer(f"[INFO] datasets  = {', '.join(args.datasets)}")

        selected: list[ModelSelection] = []
        for logical_model in args.models:
            selection = _choose_model(logical_model, base_config, printer)
            selected.append(selection)
            if selection.actual_model is None:
                printer(f"[WARN] no usable model found for {logical_model}")
                continue
            printer(f"[INFO] selected {logical_model} -> {selection.actual_model}")

        run_records: list[RunRecord] = []
        for selection in selected:
            if selection.actual_model is None:
                continue
            for dataset in args.datasets:
                run_record = _run_dataset_script(
                    dataset,
                    selection.logical_model,
                    selection.actual_model,
                    base_url=base_config.base_url,
                    api_key=base_config.api_key,
                    mode=args.mode,
                    printer=printer,
                )
                run_records.append(run_record)

        passed_runs = sum(1 for record in run_records if record.ok)
        failed_runs = len(run_records) - passed_runs
        printer(
            "[SUMMARY] "
            f"mode={args.mode}, "
            f"selected={sum(1 for item in selected if item.actual_model is not None)}, "
            f"runs={len(run_records)}, passed={passed_runs}, failed={failed_runs}"
        )

    summary = {
        "timestamp": timestamp,
        "mode": args.mode,
        "base_url": base_config.base_url,
        "models": [asdict(selection) for selection in selected],
        "runs": [asdict(record) for record in run_records],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] log written to {log_path}")
    print(f"[INFO] summary written to {summary_path}")

    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
