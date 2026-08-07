"""Command-line entry point for the Phase 1 local inference baseline."""

from __future__ import annotations

import argparse
import json

from low_latency_llm_serving.devices import SUPPORTED_DEVICES, SUPPORTED_DTYPES
from low_latency_llm_serving.inference import (
    DEFAULT_MODEL_ID,
    GenerationSettings,
    LocalLLM,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and measure a compact causal LLM locally."
    )
    parser.add_argument(
        "--prompt",
        default="Explain the difference between latency and throughput in two sentences.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    llm = LocalLLM(model_id=args.model, device=args.device, dtype=args.dtype)
    result = llm.generate(args.prompt, settings)

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"\nResponse:\n{result.response}\n")
    print("Measurements:")
    print(f"  model:              {result.model_id}")
    print(f"  device / dtype:     {result.device} / {result.dtype}")
    print(f"  input tokens:       {result.input_tokens}")
    print(f"  output tokens:      {result.output_tokens}")
    print(f"  model load:         {result.load_seconds:.3f} s")
    print(f"  generation:         {result.generation_seconds:.3f} s")
    print(f"  generation speed:   {result.tokens_per_second:.2f} tokens/s")
    print(f"  process RSS:        {result.process_rss_mb:.1f} MiB")


if __name__ == "__main__":
    main()

