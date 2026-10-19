"""OpenAI-compatible API smoke test.

This script checks that the configured base URL and API key can be loaded
from environment variables and that a minimal chat completion request works.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Mapping

DEFAULT_MODEL = "glm-5.1"
DEFAULT_PROMPT = "hello"
DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class RuntimeConfig:
    base_url: str
    api_key: str
    model: str
    prompt: str
    timeout: float
    max_tokens: int
    temperature: float


def mask_secret(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip()
    while normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def resolve_api_key(env: Mapping[str, str]) -> str:
    return (
        env.get("HAOMIAO_AUTH_TOKEN")
        or env.get("HAOMIAO_AUTHEN_TOKEN")
        or env.get("OPENAI_API_KEY")
        or env.get("ANTHROPIC_AUTH_TOKEN")
        or ""
    )


def resolve_runtime_config(args: argparse.Namespace, env: Mapping[str, str]) -> RuntimeConfig:
    base_url = args.base_url or env.get("HAOMIAO_URL") or env.get("OPENAI_BASE_URL") or ""
    api_key = args.api_key or resolve_api_key(env)
    model = args.model or DEFAULT_MODEL
    prompt = args.prompt or DEFAULT_PROMPT
    return RuntimeConfig(
        base_url=normalize_base_url(base_url),
        api_key=api_key.strip(),
        model=model.strip(),
        prompt=prompt,
        timeout=float(args.timeout),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
    )


def validate_runtime_config(config: RuntimeConfig) -> None:
    if not config.base_url:
        raise ValueError(
            "缺少 base_url。请设置 HAOMIAO_URL，或通过 --base-url 显式传入。"
        )
    if not config.base_url.startswith(("http://", "https://")):
        raise ValueError(f"base_url 格式不合法: {config.base_url}")
    if not config.api_key:
        raise ValueError(
            "缺少 API key。请设置 HAOMIAO_AUTH_TOKEN，或通过 --api-key 显式传入。"
        )
    if not config.model:
        raise ValueError("model 不能为空。")


def build_client(config: RuntimeConfig):
    from openai import OpenAI

    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
    )


def run_smoke_test(config: RuntimeConfig) -> dict[str, object]:
    client = build_client(config)
    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": "You are a concise test harness."},
            {"role": "user", "content": config.prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    content = response.choices[0].message.content if response.choices else ""
    usage = getattr(response, "usage", None)
    return {
        "elapsed_ms": elapsed_ms,
        "content": content,
        "usage": usage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for an OpenAI-compatible endpoint."
    )
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="API key for the endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to test.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the smoke test request.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Maximum tokens for the test response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the test request.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_runtime_config(args, os.environ)

    try:
        validate_runtime_config(config)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print("[INFO] OpenAI-compatible smoke test")
    print(f"[INFO] base_url = {config.base_url}")
    print(f"[INFO] api_key   = {mask_secret(config.api_key)}")
    print(f"[INFO] model     = {config.model}")

    try:
        result = run_smoke_test(config)
    except Exception as exc:  # noqa: BLE001 - surface the real provider error
        print(f"[FAIL] request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("[PASS] request succeeded")
    print(f"[INFO] latency_ms = {result['elapsed_ms']}")
    if result["content"]:
        print(f"[INFO] content     = {result['content']}")
    usage = result["usage"]
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        print(
            "[INFO] usage       = "
            f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
