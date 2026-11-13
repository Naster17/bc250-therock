#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D67 owned synthetic LoRA benchmark data (no user data, exact token lengths).

Generates deterministic pseudo-conversations whose chat-template token length is
EXACTLY the requested value, so single-host and two-host runs see identical
per-update token counts at each sequence length. Runs on the board under the
application environment (CPU-only tokenizer work); the same seed on both hosts
produces byte-identical files (verified by the orchestrator).

The word generator is pure logic and unit-testable on the workstation; the
tokenizer is imported lazily in main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

FIXTURE = "make_owned_data_d67"
SEED = 6701
SAMPLES = 256
LENGTHS = (1024, 4096, 8192)
VOCAB = (
    "brel",
    "tov",
    "mux",
    "sen",
    "dral",
    "kiv",
    "pon",
    "wex",
    "zul",
    "glim",
    "farn",
    "hob",
    "jux",
    "kren",
    "lorm",
    "nesh",
    "piv",
    "quor",
    "ruld",
    "skep",
    "tarn",
    "vex",
    "worp",
    "yund",
    "zeph",
    "blin",
    "crov",
    "dwin",
    "flik",
    "gorn",
)
PROMPT_PREFIX = "Continue the pattern study with a detailed explanation. "
COMPLETION_PREFIX = "Here is the detailed explanation of the pattern study. "


def words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(count))


def example(rng: random.Random, prompt_words: int, completion_words: int) -> dict:
    return {
        "prompt": [
            {"role": "user", "content": PROMPT_PREFIX + words(rng, prompt_words)}
        ],
        "completion": [
            {
                "role": "assistant",
                "content": COMPLETION_PREFIX + words(rng, completion_words),
            }
        ],
    }


def token_length(tokenizer, example_row: dict) -> int:
    tokens = tokenizer.apply_chat_template(
        example_row["prompt"] + example_row["completion"],
        tokenize=True,
        add_generation_prompt=False,
    )
    # Newer transformers returns a dict-like BatchEncoding (a UserDict, not a
    # dict subclass; len == key count), older versions a list or tensor;
    # normalize to a flat token-id list.
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    elif isinstance(tokens, dict) and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    while (
        isinstance(tokens, (list, tuple))
        and tokens
        and isinstance(tokens[0], (list, tuple))
    ):
        tokens = tokens[0]
    return len(tokens)


def fit_completion_words(
    tokenizer, prompt_seed: int, completion_seed: int, prompt_words: int, target: int
) -> int:
    """Binary-search the completion word count for an exact total token length.

    Each evaluation must be deterministic for its word count (fresh per-evaluation
    RNGs), otherwise the search is non-monotonic and can skip the target.
    """

    def length(n: int) -> int:
        row = example(random.Random(prompt_seed), prompt_words, 0)
        row["completion"] = [
            {
                "role": "assistant",
                "content": COMPLETION_PREFIX + words(random.Random(completion_seed), n),
            }
        ]
        return token_length(tokenizer, row)

    lo, hi = 1, target
    while lo < hi:
        mid = (lo + hi) // 2
        if length(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    for n in range(max(1, lo - 8), lo + 9):
        if length(n) == target:
            return n
    raise ValueError(f"no exact completion word count for target {target}")


def estimate_prompt_words(tokenizer, prompt_seed: int, length: int) -> int:
    """Size the prompt to ~35% of the target using measured tokens-per-word.

    Nonsense syllables can cost 2-3 tokens each, so a fixed word fraction of the
    target can overshoot the whole budget before the completion is added.
    """
    small = token_length(tokenizer, example(random.Random(prompt_seed), 100, 1))
    large = token_length(tokenizer, example(random.Random(prompt_seed), 200, 1))
    per_word = max((large - small) / 100.0, 0.5)
    return max(8, int(0.35 * length / per_word))


def generate(tokenizer, length: int, samples: int, seed: int) -> list[dict]:
    rows = []
    for i in range(samples):
        prompt_seed = seed + length * 100000 + i
        prompt_words = estimate_prompt_words(tokenizer, prompt_seed, length)
        fit = None
        # Vary both the prompt word count and the completion text: with coarse
        # token costs a fixed completion sequence can skip the target for every
        # prompt variant (modular alignment), so retry with fresh completions.
        for attempt in range(4):
            completion_seed = prompt_seed + 1 + attempt * 7919
            for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6):
                pw = prompt_words + delta
                try:
                    n = fit_completion_words(
                        tokenizer, prompt_seed, completion_seed, pw, length
                    )
                    fit = (pw, completion_seed, n)
                    break
                except ValueError:
                    continue
            if fit:
                break
        if fit is None:
            raise ValueError(f"sample {i}: no exact fit for length {length}")
        prompt_words, completion_seed, completion_words = fit
        row = example(random.Random(prompt_seed), prompt_words, 0)
        row["completion"] = [
            {
                "role": "assistant",
                "content": COMPLETION_PREFIX
                + words(random.Random(completion_seed), completion_words),
            }
        ]
        actual = token_length(tokenizer, row)
        if actual != length:
            raise ValueError(f"length {actual} != {length}")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="*", type=int, default=list(LENGTHS))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fixture": FIXTURE,
        "seed": args.seed,
        "samples_per_length": args.samples,
        "tokenizer": args.tokenizer,
        "generator_sha256": sha256_file(Path(__file__)),
        "files": {},
    }
    for length in args.lengths:
        rows = generate(tokenizer, length, args.samples, args.seed)
        path = args.output_dir / f"owned-{length}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        manifest["files"][str(length)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "samples": len(rows),
            "exact_token_length": length,
        }
        print(
            f"D67_DATA length={length} samples={len(rows)} sha256={manifest['files'][str(length)]['sha256']}",
            flush=True,
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("D67_DATA complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
