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


if __name__ == "__main__":
    unittest.main()
