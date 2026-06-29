#!/usr/bin/env python3
"""Export a tiktoken encoding as a Hugging Face Fast Tokenizer JSON file.

Install dependencies:
    python -m pip install tiktoken tokenizers

Usage:
    python scripts/export_tiktoken_tokenizer_json.py
    python scripts/export_tiktoken_tokenizer_json.py --output tokenizer.json
    python scripts/export_tiktoken_tokenizer_json.py --output tokenizer.json --compact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def list_available_encoding_names() -> list[str]:
    try:
        import tiktoken
    except ImportError:
        return ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base"]

    return tiktoken.list_encoding_names()


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-unicode mapping used by ByteLevel BPE tokenizers."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


BYTE_ENCODER = bytes_to_unicode()


def token_bytes_to_string(token: bytes) -> str:
    return "".join(BYTE_ENCODER[b] for b in token)


def bpe_parts_for_token(mergeable_ranks: dict[bytes, int], token: bytes, rank: int) -> list[bytes]:
    parts = [bytes([byte]) for byte in token]

    while len(parts) > 1:
        pair_rank = None
        pair_index = None
        for index, (left, right) in enumerate(zip(parts, parts[1:])):
            candidate_rank = mergeable_ranks.get(left + right)
            if candidate_rank is None or candidate_rank >= rank:
                continue
            if pair_rank is None or candidate_rank < pair_rank:
                pair_rank = candidate_rank
                pair_index = index

        if pair_index is None:
            break

        parts[pair_index : pair_index + 2] = [parts[pair_index] + parts[pair_index + 1]]

    return parts


def recover_bpe_merges(mergeable_ranks: dict[bytes, int]) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []

    for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1]):
        if len(token) == 1:
            continue

        parts = bpe_parts_for_token(mergeable_ranks, token, rank)
        if len(parts) != 2:
            continue

        merges.append((token_bytes_to_string(parts[0]), token_bytes_to_string(parts[1])))

    return merges


def fill_vocab_holes(vocab: dict[str, int]) -> None:
    used_ids = set(vocab.values())
    max_id = max(used_ids)

    for token_id in range(max_id + 1):
        if token_id in used_ids:
            continue
        placeholder = f"<|reserved_token_{token_id}|>"
        if placeholder in vocab:
            raise ValueError(f"Reserved placeholder already exists in vocab: {placeholder}")
        vocab[placeholder] = token_id


def stringify_merges(tokenizer_json: str, compact: bool) -> str:
    payload = json.loads(tokenizer_json)
    payload["model"]["merges"] = [
        " ".join(merge) if isinstance(merge, list) else merge
        for merge in payload["model"]["merges"]
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
    )


def build_fast_tokenizer_json(encoding_name: str, compact: bool, array_merges: bool) -> str:
    try:
        import tiktoken
        from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python -m pip install tiktoken tokenizers"
        ) from exc

    encoding = tiktoken.get_encoding(encoding_name)

    vocab = {
        token_bytes_to_string(token): rank
        for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda item: item[1])
    }
    vocab.update(encoding._special_tokens)
    fill_vocab_holes(vocab)

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=recover_bpe_merges(encoding._mergeable_ranks)))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(encoding._pat_str), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [token for token, _ in sorted(encoding._special_tokens.items(), key=lambda item: item[1])]
    )

    tokenizer_json = tokenizer.to_str(pretty=not compact)
    if array_merges:
        return tokenizer_json
    return stringify_merges(tokenizer_json, compact)


def parse_args() -> argparse.Namespace:
    available_encodings = list_available_encoding_names()
    parser = argparse.ArgumentParser(
        description="Export a tiktoken encoding to tokenizer.json.",
        epilog="Available encodings: " + ", ".join(available_encodings),
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="tiktoken encoding name to export. Defaults to o200k_base.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tokenizer.json",
        help="Output JSON path. Defaults to ./tokenizer.json.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact single-line JSON instead of indented JSON.",
    )
    parser.add_argument(
        "--array-merges",
        action="store_true",
        help="Keep model.merges as two-item arrays instead of the default space-separated strings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        build_fast_tokenizer_json(args.encoding, args.compact, args.array_merges),
        encoding="utf-8",
    )
    print(f"Wrote {args.encoding} Fast Tokenizer JSON to {output_path}")


if __name__ == "__main__":
    main()
