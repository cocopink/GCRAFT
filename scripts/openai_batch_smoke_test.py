"""Batch smoke test for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.openai_smoke_test import (
    DEFAULT_PROMPT,
    DEFAULT_TIMEOUT,
    RuntimeConfig,
    mask_secret,
    normalize_base_url,
    resolve_api_key,
    run_smoke_test,
    validate_runtime_config,
)

DEFAULT_MODELS = [
    "gpt-5.3-codex",
    "gpt-5.3-codex-high",
    "gpt-5.3-codex-spark",
    "gpt-5.3-codex-xhigh",
    "gpt-5.4",
    "gpt-5.4-high",
    "gpt-5.4-low",
    "gpt-5.4-medium",
    "gpt-5.4-mini",
    "gpt-5.4-xhigh",
    "claude-opus-4-6",
    "claude-opus-4-6-cc",
    "claude-opus-4-6-thinking",
    "claude-opus-4-7",
    "claude-opus-4-7-cc",
    "claude-sonnet-4-6",
    "claude-sonnet-4-6-cc",
    "claude-sonnet-4-6-thinking",
    "DeepSeek-V3.2",
    "glm-5-turbo",
    "glm-5.1",
    "glm-5v-turbo",
    "kimi-for-coding",
    "MiniMax-M2.7",
    "qwen3.6-plus",
]
DEFAULT_PROVIDER_FILE = "configs/llm_providers.json"
DEFAULT_PROVIDER_NAME = "current-gpt54"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str
    model_name: str


@dataclass(frozen=True)
class ModelResult:
    model: str
    ok: bool
    latency_ms: float | None
    detail: str


@dataclass(frozen=True)
class ResultSummary:
    total: int
    passed: int
    failed: int
    exit_code: int


def load_provider_config(provider_file: str, provider_name: str) -> ProviderConfig | None:
    if not provider_file:
        return None

    path = Path(provider_file)
    if not path.exists():
        return None

    providers = json.loads(path.read_text(encoding="utf-8"))
    for provider in providers:
        if provider.get("name") == provider_name:
            return ProviderConfig(
                name=str(provider.get("name", "")),
                api_key_env=str(provider.get("api_key_env", "")),
                base_url=str(provider.get("base_url", "")),
                model_name=str(provider.get("model_name", "")),
            )
    raise ValueError(f"provider not found: {provider_name}")


def parse_models(explicit_models: Iterable[str], models_file: str) -> list[str]:
    models = [model.strip() for model in explicit_models if model.strip()]
    if models:
        return models

    if models_file:
        return [
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    return list(DEFAULT_MODELS)


def summarize_results(results: Iterable[ModelResult]) -> ResultSummary:
    result_list = list(results)
    passed = sum(1 for result in result_list if result.ok)
    failed = len(result_list) - passed
    return ResultSummary(
        total=len(result_list),
        passed=passed,
        failed=failed,
        exit_code=0 if failed == 0 else 1,
    )


def run_one_model(base_config: RuntimeConfig, model: str) -> ModelResult:
    config = dataclasses.replace(base_config, model=model)
    started_at = time.perf_counter()
    try:
        result = run_smoke_test(config)
    except Exception as exc:  # noqa: BLE001 - report provider/network failures
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        return ModelResult(
            model=model,
            ok=False,
            latency_ms=elapsed_ms,
            detail=f"{type(exc).__name__}: {exc}",
        )

    content = str(result.get("content") or "").replace("\n", " ").strip()
    return ModelResult(
        model=model,
        ok=True,
        latency_ms=float(result["elapsed_ms"]),
        detail=content,
    )


def resolve_runtime_config(
    *,
    base_url: str,
    api_key: str,
    provider_file: str,
    provider_name: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
    env: dict[str, str] | None = None,
) -> RuntimeConfig:
    current_env = dict(os.environ) if env is None else env
    provider = load_provider_config(provider_file, provider_name)
    provider_base_url = provider.base_url if provider is not None else ""
    provider_api_key = (
        current_env.get(provider.api_key_env, "")
        if provider is not None and provider.api_key_env
        else ""
    )
    resolved_base_url = (
        base_url
        or provider_base_url
        or current_env.get("HAOMIAO_URL")
        or current_env.get("OPENAI_BASE_URL")
        or ""
    )
    resolved_api_key = api_key or provider_api_key or resolve_api_key(current_env)
    return RuntimeConfig(
        base_url=normalize_base_url(resolved_base_url),
        api_key=resolved_api_key.strip(),
        model="",
        prompt=prompt or DEFAULT_PROMPT,
        timeout=float(timeout),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )


def build_runtime_config(args: argparse.Namespace, env: dict[str, str]) -> RuntimeConfig:
    return resolve_runtime_config(
        base_url=args.base_url,
        api_key=args.api_key,
        provider_file=args.provider_file,
        provider_name=args.provider,
        prompt=args.prompt,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        env=env,
    )


def print_result(result: ModelResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    latency = "-" if result.latency_ms is None else f"{result.latency_ms:.1f}ms"
    print(f"[{status}] {result.model} ({latency}) {result.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch smoke test for OpenAI-compatible endpoints."
    )
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="API key for the endpoint.")
    parser.add_argument(
        "--provider-file",
        default=DEFAULT_PROVIDER_FILE,
        help="Provider config JSON file.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER_NAME,
        help="Provider name to load from --provider-file.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model to test. Repeat this option to test multiple models.",
    )
    parser.add_argument(
        "--models-file",
        default="",
        help="Optional newline-delimited model file. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for each smoke test request.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=2, help="Maximum tokens for each response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for each request.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    models = parse_models(args.model, args.models_file)
    config = build_runtime_config(args, dict(os.environ))

    try:
        validate_runtime_config(dataclasses.replace(config, model=models[0] if models else ""))
    except (IndexError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print("[INFO] OpenAI-compatible batch smoke test")
    print(f"[INFO] base_url = {config.base_url}")
    print(f"[INFO] api_key   = {mask_secret(config.api_key)}")
    print(f"[INFO] models    = {len(models)}")

    results = []
    for model in models:
        result = run_one_model(config, model)
        results.append(result)
        print_result(result)

    summary = summarize_results(results)
    print(
        "[SUMMARY] "
        f"total={summary.total}, passed={summary.passed}, failed={summary.failed}"
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
