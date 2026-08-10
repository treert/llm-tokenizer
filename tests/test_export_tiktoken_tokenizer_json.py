from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_tiktoken_tokenizer_json.py"

sys.path.insert(0, str(ROOT / "scripts"))
from export_tiktoken_tokenizer_json import (
    PATTERNS,
    build_reference_encoding,
    load_mergeable_ranks,
    load_special_tokens,
)

KIMI_DIR = ROOT / "kimi-k3"
KIMI_VOCAB = KIMI_DIR / "tiktoken.model"
KIMI_CONFIG = KIMI_DIR / "tokenizer_config.json"


class ExportTiktokenTokenizerJsonTest(unittest.TestCase):
    def test_help_lists_available_encodings(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

        for encoding_name in tiktoken.list_encoding_names():
            self.assertIn(encoding_name, result.stdout)

    def test_default_output_uses_encoding_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(
                [sys.executable, str(SCRIPT)],
                check=True,
                cwd=tmp_dir,
            )

            output_path = Path(tmp_dir) / "gpt-o200k-base" / "tokenizer.json"
            self.assertTrue(output_path.is_file())

    def test_exports_standard_fast_tokenizer_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "tokenizer.json"

            subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(output_path)],
                check=True,
                cwd=ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertTrue(payload["version"])
            self.assertEqual(payload["model"]["type"], "BPE")
            self.assertEqual(payload["pre_tokenizer"]["type"], "Sequence")
            self.assertEqual(payload["decoder"]["type"], "ByteLevel")
            self.assertIsInstance(payload["model"]["merges"][0], str)
            self.assertIn(
                "<|endoftext|>", {token["content"] for token in payload["added_tokens"]}
            )
            vocab_ids = set(payload["model"]["vocab"].values())
            self.assertEqual(vocab_ids, set(range(max(vocab_ids) + 1)))

            tokenizer = Tokenizer.from_file(str(output_path))
            self.assertTrue(tokenizer.encode("hello world").ids)

            tiktoken_encoding = tiktoken.get_encoding("o200k_base")
            for text in ["hello world", "你好，世界", "2 + 2 = 4"]:
                self.assertEqual(tokenizer.encode(text).ids, tiktoken_encoding.encode(text))

    def test_can_export_array_merges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "tokenizer.json"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output",
                    str(output_path),
                    "--array-merges",
                ],
                check=True,
                cwd=ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertIsInstance(payload["model"]["merges"][0], list)
            self.assertEqual(len(payload["model"]["merges"][0]), 2)
            self.assertTrue(Tokenizer.from_file(str(output_path)).encode("hello world").ids)


@unittest.skipUnless(KIMI_VOCAB.is_file(), "kimi-k3/tiktoken.model 不存在")
class KimiK3HelpersTest(unittest.TestCase):
    def test_kimi_k3_pattern_preserved(self) -> None:
        pattern = PATTERNS["kimi-k3"]
        self.assertIn(r"[\p{Han}]+", pattern)
        self.assertIn(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)", pattern)
        self.assertIn(r"\p{N}{1,3}", pattern)
        self.assertIn(r"&&[^\p{Han}]", pattern)

    def test_load_special_tokens_from_kimi_config(self) -> None:
        special = load_special_tokens(str(KIMI_CONFIG), 163584)
        self.assertEqual(len(special), 256)
        self.assertEqual(special["[BOS]"], 163584)
        self.assertEqual(special["[EOS]"], 163585)
        self.assertEqual(special["<|open|>"], 163587)
        self.assertEqual(special["<|close|>"], 163588)
        self.assertEqual(special["<|sep|>"], 163589)
        self.assertEqual(special["[UNK]"], 163838)
        self.assertEqual(special["[PAD]"], 163839)
        # 163600 is a unnamed reserved slot
        self.assertEqual(special["<|reserved_token_163600|>"], 163600)

    def test_load_special_tokens_without_config(self) -> None:
        special = load_special_tokens(None, 100, num_reserved=3)
        self.assertEqual(
            special,
            {
                "<|reserved_token_100|>": 100,
                "<|reserved_token_101|>": 101,
                "<|reserved_token_102|>": 102,
            },
        )

    def test_load_mergeable_ranks(self) -> None:
        ranks = load_mergeable_ranks(str(KIMI_VOCAB))
        self.assertEqual(len(ranks), 163584)
        self.assertEqual(ranks[b"!"], 0)


@unittest.skipUnless(KIMI_VOCAB.is_file(), "kimi-k3/tiktoken.model 不存在")
class ExportKimiK3TokenizerJsonTest(unittest.TestCase):
    def test_converts_kimi_k3_vocab_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "tokenizer.json"
            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--vocab-file", str(KIMI_VOCAB),
                    "--pattern", "kimi-k3",
                    "--tokenizer-config", str(KIMI_CONFIG),
                    "--output", str(output_path),
                ],
                check=True, cwd=ROOT,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"]["type"], "BPE")
            self.assertEqual(payload["pre_tokenizer"]["type"], "Sequence")
            self.assertEqual(payload["decoder"]["type"], "ByteLevel")

            vocab_ids = set(payload["model"]["vocab"].values())
            self.assertEqual(len(vocab_ids), 163840)
            self.assertEqual(vocab_ids, set(range(163840)))

            added = {t["content"]: t["id"] for t in payload["added_tokens"]}
            self.assertEqual(added["[BOS]"], 163584)
            self.assertEqual(added["<|open|>"], 163587)

            tokenizer = Tokenizer.from_file(str(output_path))
            self.assertTrue(tokenizer.encode("你好世界").ids)

    def test_verify_matches_tiktoken_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "tokenizer.json"
            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--vocab-file", str(KIMI_VOCAB),
                    "--pattern", "kimi-k3",
                    "--tokenizer-config", str(KIMI_CONFIG),
                    "--output", str(output_path),
                    "--verify",
                ],
                check=True, cwd=ROOT,
            )
            # If --verify failed, it would raise CalledProcessError


if __name__ == "__main__":
    unittest.main()
