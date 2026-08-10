#!/usr/bin/env python3
"""Export a tiktoken encoding as a Hugging Face Fast Tokenizer JSON file.

Install dependencies:
    python -m pip install tiktoken tokenizers

Usage:
    python scripts/export_tiktoken_tokenizer_json.py
    python scripts/export_tiktoken_tokenizer_json.py --output gpt-o200k-base/tokenizer.json
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

NUM_RESERVED_SPECIAL_TOKENS = 256

# Matches Kimi-K3 official tokenization_kimi.py pat_str character-by-character
PATTERNS = {
    "kimi-k3": "|".join([
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]),
}


def load_mergeable_ranks(vocab_file: str) -> dict[bytes, int]:
    try:
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python -m pip install tiktoken"
        ) from exc
    return load_tiktoken_bpe(vocab_file)


def load_special_tokens(
    tokenizer_config_path: str | None,
    num_base_tokens: int,
    num_reserved: int = NUM_RESERVED_SPECIAL_TOKENS,
) -> dict[str, int]:
    """Generate special token table matching official tokenization_kimi.py rules.

    tokenizer_config.json's added_tokens_decoder provides named tokens,
    remaining reserved slots are named <|reserved_token_{i}>.
    """
    named: dict[int, str] = {}
    if tokenizer_config_path:
        config = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))
        named = {
            int(token_id): entry["content"]
            for token_id, entry in config.get("added_tokens_decoder", {}).items()
        }
    return {
        named.get(i, f"<|reserved_token_{i}|>"): i
        for i in range(num_base_tokens, num_base_tokens + num_reserved)
    }


def build_reference_encoding(mergeable_ranks, pat_str, special_tokens):
    """Build tiktoken reference Encoding for --verify and test comparisons."""
    import tiktoken

    return tiktoken.Encoding(
        name="tiktoken-reference",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


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


def build_fast_tokenizer_json(
    encoding_name: str | None = None,
    compact: bool = False,
    array_merges: bool = False,
    vocab_file: str | None = None,
    pattern: str | None = None,
    tokenizer_config: str | None = None,
) -> str:
    try:
        import tiktoken
        from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python -m pip install tiktoken tokenizers"
        ) from exc

    if vocab_file:
        mergeable_ranks = load_mergeable_ranks(vocab_file)
        pat_str = PATTERNS[pattern] if pattern else PATTERNS["kimi-k3"]
        num_base = len(mergeable_ranks)
        special_tokens = load_special_tokens(tokenizer_config, num_base)
    else:
        encoding = tiktoken.get_encoding(encoding_name)
        mergeable_ranks = encoding._mergeable_ranks
        pat_str = encoding._pat_str
        special_tokens = encoding._special_tokens

    vocab = {
        token_bytes_to_string(token): rank
        for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1])
    }
    vocab.update(special_tokens)
    fill_vocab_holes(vocab)

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=recover_bpe_merges(mergeable_ranks)))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pat_str), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [token for token, _ in sorted(special_tokens.items(), key=lambda item: item[1])]
    )

    tokenizer_json = tokenizer.to_str(pretty=not compact)
    if array_merges:
        return tokenizer_json
    return stringify_merges(tokenizer_json, compact)


VERIFY_TEXTS = [
    "你好世界",
    "今天天气真不错，我们一起去公园散步吧！",
    "Hello world! 这是中英混排测试。Python3.12发布了。",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "2024年全球GDP增长3.2%，中国贡献了约30%。",
    "1234567890",
    "😀🎉🔥 表情符号测试 emoji test",
    "   leading spaces\n\ttabs\there\n\nmultiple newlines",
    "don't can't it's they've I'm you'll",
    "",
    "<|open|>literal open tag<|close|> and [BOS] token",
    "https://kimi.moonshot.cn/chat?q=测试",
    '{"key": "值", "nested": {"arr": [1, 2, 3]}}',
]


def verify_tokenizer_json(
    output_path: Path,
    mergeable_ranks: dict[bytes, int],
    pat_str: str,
    special_tokens: dict[str, int],
) -> None:
    """Compare generated tokenizer.json against tiktoken reference on test texts."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(output_path))
    ref = build_reference_encoding(mergeable_ranks, pat_str, special_tokens)

    all_passed = True
    for text in VERIFY_TEXTS:
        our_ids = tokenizer.encode(text).ids
        ref_ids = ref.encode(text, allowed_special="all")
        if our_ids != ref_ids:
            all_passed = False
            print(f"[FAIL] {text!r}")
            print(f"  ours: {our_ids}")
            print(f"  ref:  {ref_ids}")

    if all_passed:
        print(f"[VERIFY OK] All {len(VERIFY_TEXTS)} texts match tiktoken reference.")
    else:
        raise SystemExit(1)


def default_output_path(encoding_name: str) -> Path:
    return Path(f"gpt-{encoding_name.replace('_', '-')}") / "tokenizer.json"


def parse_args() -> argparse.Namespace:
    available_encodings = list_available_encoding_names()
    parser = argparse.ArgumentParser(
        description="Export a tiktoken encoding to tokenizer.json.",
        epilog="Available encodings: " + ", ".join(available_encodings),
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="tiktoken encoding name to export (e.g. o200k_base). If not given, defaults to o200k_base.",
    )
    parser.add_argument(
        "--vocab-file",
        help="Path to a tiktoken.model BPE vocab file (e.g. kimi-k3/tiktoken.model). "
             "Mutually exclusive with --encoding.",
    )
    parser.add_argument(
        "--pattern",
        choices=list(PATTERNS.keys()),
        help="Named pre-tokenization pattern (required with --vocab-file).",
    )
    parser.add_argument(
        "--tokenizer-config",
        help="Path to tokenizer_config.json for special token names (optional with --vocab-file).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path. Defaults to ./gpt-{encoding}/tokenizer.json.",
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated tokenizer.json against tiktoken reference (requires --vocab-file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.vocab_file and args.encoding:
        raise SystemExit("--vocab-file and --encoding are mutually exclusive.")
    if args.vocab_file and not args.pattern:
        raise SystemExit("--pattern is required when using --vocab-file.")

    if args.vocab_file:
        encoding_label = f"vocab={Path(args.vocab_file).stem} pattern={args.pattern}"
        encoding_name: str | None = None
        if not args.output:
            raise SystemExit("--output is required when using --vocab-file.")
    else:
        encoding_name = args.encoding or "o200k_base"
        encoding_label = encoding_name

    output_path = Path(args.output) if args.output else default_output_path(encoding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        build_fast_tokenizer_json(
            encoding_name=encoding_name,
            compact=args.compact,
            array_merges=args.array_merges,
            vocab_file=args.vocab_file,
            pattern=args.pattern,
            tokenizer_config=args.tokenizer_config,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {encoding_label} Fast Tokenizer JSON to {output_path}")

    if args.verify:
        if not args.vocab_file:
            raise SystemExit("--verify requires --vocab-file.")
        mergeable_ranks = load_mergeable_ranks(args.vocab_file)
        pat_str = PATTERNS[args.pattern]
        num_base = len(mergeable_ranks)
        special_tokens = load_special_tokens(args.tokenizer_config, num_base)
        verify_tokenizer_json(output_path, mergeable_ranks, pat_str, special_tokens)


if __name__ == "__main__":
    main()
