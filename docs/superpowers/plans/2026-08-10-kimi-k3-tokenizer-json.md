# Kimi-K3 Tokenizer.json 转换接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩展 `scripts/export_tiktoken_tokenizer_json.py`，把 `kimi-k3/tiktoken.model` 转换为标准 `kimi-k3/tokenizer.json` 并与 tiktoken 参考实现比对一致，使 `llm_tokenizer.py --dir kimi-k3` 可用。

**Architecture:** 复用现有 tiktoken→Fast Tokenizer 转换管线（vocab/merges 恢复、fill_vocab_holes），新增自定义词表文件入口（`--vocab-file` + `--pattern` + `--tokenizer-config`）和 `--verify` 对照校验。`llm_tokenizer.py` 逻辑零改动。

**Tech Stack:** Python 3、tiktoken 0.13.0、tokenizers 0.22.2（均已安装，不新增依赖）、unittest。

设计文档：`docs/superpowers/specs/2026-08-10-kimi-k3-tokenizer-json-design.md`

## Global Constraints

- OS 为 Windows，shell 为 PowerShell；Python 命令用 `python`
- `PATTERNS["kimi-k3"]` 必须与 `G:\llm-models\Kimi-K3\tokenization_kimi.py` 中的 `pat_str` 逐字符一致
- 特殊 token 命名规则与官方一致：未命名槽位为 `<|reserved_token_{i}|>`，id 区间 163584–163839 共 256 个
- 不改动 `llm_tokenizer.py` 的任何分词逻辑，只更新 docstring 与帮助文本
- 现有 o200k 测试（`ExportTiktokenTokenizerJsonTest`）必须保持全部通过
- 转换产物 `kimi-k3/tokenizer.json` 的 vocab id 必须恰好是 0–163839 连续整数

---

### Task 1: 复制模型文件 + 脚本新增 K3 辅助函数

**Files:**
- Create: `kimi-k3/tiktoken.model`（复制自 `G:\llm-models\Kimi-K3\tiktoken.model`）
- Create: `kimi-k3/tokenizer_config.json`（复制自 `G:\llm-models\Kimi-K3\tokenizer_config.json`）
- Modify: `scripts/export_tiktoken_tokenizer_json.py`
- Test: `tests/test_export_tiktoken_tokenizer_json.py`

**Interfaces:**
- Produces（后续任务依赖这些确切签名）:
  - `PATTERNS: dict[str, str]` — key `"kimi-k3"` 对应官方 pat_str
  - `NUM_RESERVED_SPECIAL_TOKENS = 256`
  - `load_mergeable_ranks(vocab_file: str) -> dict[bytes, int]`
  - `load_special_tokens(tokenizer_config_path: str | None, num_base_tokens: int, num_reserved: int = NUM_RESERVED_SPECIAL_TOKENS) -> dict[str, int]`
  - `build_reference_encoding(mergeable_ranks: dict[bytes, int], pat_str: str, special_tokens: dict[str, int]) -> tiktoken.Encoding`

- [ ] **Step 1: 复制模型文件到 kimi-k3/**

```powershell
Copy-Item 'G:\llm-models\Kimi-K3\tiktoken.model' 'f:\MyGit\llm-tokenizer\kimi-k3\tiktoken.model'
Copy-Item 'G:\llm-models\Kimi-K3\tokenizer_config.json' 'f:\MyGit\llm-tokenizer\kimi-k3\tokenizer_config.json'
```

验证：`Get-Item f:\MyGit\llm-tokenizer\kimi-k3\*` 应看到 2 个文件，`tiktoken.model` 大小 2,795,286 字节。

- [ ] **Step 2: 写失败测试**

在 `tests/test_export_tiktoken_tokenizer_json.py` 顶部 import 区之后、`ROOT` 定义之后追加：

```python
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
```

在文件末尾（`if __name__ == "__main__":` 之前）追加：

```python
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
        # 163600 是 tokenizer_config.json 未命名的空洞槽位
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
```

注意：`test_load_mergeable_ranks` 中断言 `ranks[b"!"] == 0` 的依据是 `tiktoken.model` 首行 `IQ== 0`（`IQ==` 即字节 `0x21`）。

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json.KimiK3HelpersTest -v`
Expected: FAIL（`ModuleNotFoundError` 或 `ImportError: cannot import name 'PATTERNS'`）

- [ ] **Step 4: 实现辅助函数**

在 `scripts/export_tiktoken_tokenizer_json.py` 的 `BYTE_ENCODER = bytes_to_unicode()` 之后插入：

```python
NUM_RESERVED_SPECIAL_TOKENS = 256

# 与 Kimi-K3 官方 tokenization_kimi.py 中的 pat_str 逐字符一致
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
    """按官方 tokenization_kimi.py 的规则生成特殊 token 表：
    tokenizer_config.json 的 added_tokens_decoder 提供命名 token，
    其余保留槽位命名为 <|reserved_token_{i}>。
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
    """构建 tiktoken 参考 Encoding，用于 --verify 和单元测试对照。"""
    import tiktoken

    return tiktoken.Encoding(
        name="tiktoken-reference",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json.KimiK3HelpersTest -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 6: 同时跑既有测试确认无回归**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json -v`
Expected: 既有 4 个 o200k 测试 + 新增 4 个全部 PASS

- [ ] **Step 7: Commit**

```powershell
git add kimi-k3 scripts/export_tiktoken_tokenizer_json.py tests/test_export_tiktoken_tokenizer_json.py
git commit -m "feat: add kimi-k3 helpers (pattern, loaders) to tiktoken export script"
```

---

### Task 2: build_fast_tokenizer_json 支持自定义词表源 + CLI 参数

**Files:**
- Modify: `scripts/export_tiktoken_tokenizer_json.py`（`build_fast_tokenizer_json`、`parse_args`、`main`）
- Test: `tests/test_export_tiktoken_tokenizer_json.py`

**Interfaces:**
- Consumes: Task 1 的 `PATTERNS`、`load_mergeable_ranks`、`load_special_tokens`
- Produces:
  - `build_fast_tokenizer_json(encoding_name: str | None, compact: bool, array_merges: bool, vocab_file: str | None = None, pattern: str | None = None, tokenizer_config: str | None = None) -> str`（前三个参数与现状一致，保持既有调用兼容）
  - CLI 新参数：`--vocab-file PATH`、`--pattern {kimi-k3}`、`--tokenizer-config PATH`；`--encoding` 默认值从 `"o200k_base"` 改为 `None`（main 中 `None` 时回落 `o200k_base`，行为不变）

- [ ] **Step 1: 写失败测试**

在 `tests/test_export_tiktoken_tokenizer_json.py` 末尾（`if __name__ == "__main__":` 之前）追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json.ExportKimiK3TokenizerJsonTest -v`
Expected: FAIL（`error: unrecognized arguments: --vocab-file`）

- [ ] **Step 3: 重构 build_fast_tokenizer_json 并扩展 CLI**

将 `build_fast_tokenizer_json` 整体替换为：

```python
def build_fast_tokenizer_json(
    encoding_name: str | None,
    compact: bool,
    array_merges: bool,
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
        if pattern not in PATTERNS:
            raise SystemExit(
                f"--pattern is required with --vocab-file (choices: {sorted(PATTERNS)})"
            )
        mergeable_ranks = load_mergeable_ranks(vocab_file)
        pat_str = PATTERNS[pattern]
        special_tokens = load_special_tokens(tokenizer_config, len(mergeable_ranks))
    else:
        encoding = tiktoken.get_encoding(encoding_name or "o200k_base")
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
```

在 `parse_args` 中把 `--encoding` 的 `default="o200k_base"` 改为 `default=None`，并在其后追加：

```python
    parser.add_argument(
        "--vocab-file",
        help="Path to a tiktoken .model BPE file. Overrides --encoding.",
    )
    parser.add_argument(
        "--pattern",
        choices=sorted(PATTERNS),
        help="Pre-tokenization pattern preset, required with --vocab-file.",
    )
    parser.add_argument(
        "--tokenizer-config",
        help="Path to tokenizer_config.json providing named special tokens.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported tokenizer against a tiktoken reference encoding.",
    )
```

把 `main` 改为：

```python
def main() -> None:
    args = parse_args()
    encoding_name = args.encoding or "o200k_base"
    output_path = Path(args.output) if args.output else default_output_path(encoding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        build_fast_tokenizer_json(
            args.encoding,
            args.compact,
            args.array_merges,
            vocab_file=args.vocab_file,
            pattern=args.pattern,
            tokenizer_config=args.tokenizer_config,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Fast Tokenizer JSON to {output_path}")
```

（`--verify` 的处理在 Task 3 接入，本 Task 只加参数定义。）

- [ ] **Step 4: 运行全部测试确认通过**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json -v`
Expected: 全部 PASS（既有 o200k 用例不受影响，因为 `--encoding` 缺省时 main 回落 `o200k_base`）

- [ ] **Step 5: Commit**

```powershell
git add scripts/export_tiktoken_tokenizer_json.py tests/test_export_tiktoken_tokenizer_json.py
git commit -m "feat: support --vocab-file/--pattern/--tokenizer-config in tiktoken export"
```

---

### Task 3: --verify 对照校验

**Files:**
- Modify: `scripts/export_tiktoken_tokenizer_json.py`（新增 `VERIFY_CORPUS`、`verify_against_reference`，`main` 接线）
- Test: `tests/test_export_tiktoken_tokenizer_json.py`

**Interfaces:**
- Consumes: Task 1 的 `build_reference_encoding`、Task 2 的 CLI 参数
- Produces:
  - `VERIFY_CORPUS: list[str]` — 13 条测试语料
  - `verify_against_reference(tokenizer_json_path, mergeable_ranks: dict[bytes, int], pat_str: str, special_tokens: dict[str, int]) -> list[str]` — 返回不一致描述列表，空列表 = 通过

- [ ] **Step 1: 写失败测试**

在 `ExportKimiK3TokenizerJsonTest` 类中追加：

```python
    def test_verify_passes_for_kimi_k3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "tokenizer.json"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--vocab-file", str(KIMI_VOCAB),
                    "--pattern", "kimi-k3",
                    "--tokenizer-config", str(KIMI_CONFIG),
                    "--output", str(output_path),
                    "--verify",
                ],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Verified", result.stdout)
```

并在 `KimiK3HelpersTest` 中追加直接比对用例：

```python
    def test_converted_tokenizer_matches_reference(self) -> None:
        from export_tiktoken_tokenizer_json import (
            VERIFY_CORPUS,
            build_fast_tokenizer_json,
            verify_against_reference,
        )

        mergeable_ranks = load_mergeable_ranks(str(KIMI_VOCAB))
        special_tokens = load_special_tokens(str(KIMI_CONFIG), len(mergeable_ranks))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "tokenizer.json"
            output_path.write_text(
                build_fast_tokenizer_json(
                    None, False, False,
                    vocab_file=str(KIMI_VOCAB),
                    pattern="kimi-k3",
                    tokenizer_config=str(KIMI_CONFIG),
                ),
                encoding="utf-8",
            )
            mismatches = verify_against_reference(
                output_path, mergeable_ranks, PATTERNS["kimi-k3"], special_tokens
            )
            self.assertEqual(mismatches, [])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json -v -k KimiK3`
Expected: FAIL（`ImportError: cannot import name 'VERIFY_CORPUS'` / `--verify` 无输出效果）

- [ ] **Step 3: 实现 VERIFY_CORPUS 与 verify_against_reference**

在 `scripts/export_tiktoken_tokenizer_json.py` 的 `build_reference_encoding` 之后插入：

```python
VERIFY_CORPUS = [
    "你好世界",
    "欢迎使用 Kimi K3 大语言模型，这是一段用于测试分词器的中文长句。",
    "Hello World, it's a test! I'M HAPPY you're here. WE'LL SEE.",
    "The quick brown fox jumps over the lazy dog 0123456789",
    "def foo(x):\n    return x + 1\n\n\nfoo(42)",
    "数字测试 0 12 123 1234 12345 999,999 3.14159",
    "emoji 🎉🎊🎈 test 👨‍👩‍👧‍👦 done",
    "混合 mixed 内容 content 123 test 混合",
    "连续空白    和\t制表符\n\n换行测试",
    "<|open|>message role=\"user\"<|sep|>你好<|close|>message<|sep|><|end_of_msg|>",
    "[BOS] [EOS] [PAD] [UNK]",
    "Trailing spaces   ",
    "   Leading spaces",
]


def verify_against_reference(
    tokenizer_json_path,
    mergeable_ranks: dict[bytes, int],
    pat_str: str,
    special_tokens: dict[str, int],
) -> list[str]:
    """用 tiktoken 参考 Encoding 逐条比对 VERIFY_CORPUS。

    参考端用 allowed_special="all"，与 tokenizers 库总是匹配 added_tokens
    的行为对齐。返回不一致描述列表，空列表表示全部一致。
    """
    from tokenizers import Tokenizer

    reference = build_reference_encoding(mergeable_ranks, pat_str, special_tokens)
    converted = Tokenizer.from_file(str(tokenizer_json_path))
    mismatches = []
    for text in VERIFY_CORPUS:
        expected = reference.encode(text, allowed_special="all")
        actual = converted.encode(text).ids
        if actual != expected:
            mismatches.append(f"{text!r}: converted={actual} != reference={expected}")
    return mismatches
```

在 `main` 的 `print(f"Wrote Fast Tokenizer JSON to {output_path}")` 之后追加：

```python
    if args.verify:
        if args.vocab_file:
            mergeable_ranks = load_mergeable_ranks(args.vocab_file)
            pat_str = PATTERNS[args.pattern]
            special_tokens = load_special_tokens(args.tokenizer_config, len(mergeable_ranks))
        else:
            import tiktoken

            encoding = tiktoken.get_encoding(encoding_name)
            mergeable_ranks = encoding._mergeable_ranks
            pat_str = encoding._pat_str
            special_tokens = encoding._special_tokens
        mismatches = verify_against_reference(
            output_path, mergeable_ranks, pat_str, special_tokens
        )
        if mismatches:
            print("Verification FAILED:", file=sys.stderr)
            for item in mismatches:
                print(f"  {item}", file=sys.stderr)
            sys.exit(1)
        print(f"Verified {len(VERIFY_CORPUS)} texts against tiktoken reference")
```

- [ ] **Step 4: 运行全部测试确认通过**

Run: `python -m unittest tests.test_export_tiktoken_tokenizer_json -v`
Expected: 全部 PASS（共 9 个：o200k 4 个 + K3 helpers 5 个 + K3 convert/verify 2 个，注意 verify 语料中若出现不一致需停下来排查，不得放行）

- [ ] **Step 5: Commit**

```powershell
git add scripts/export_tiktoken_tokenizer_json.py tests/test_export_tiktoken_tokenizer_json.py
git commit -m "feat: add --verify tiktoken reference check to export script"
```

---

### Task 4: 生成 kimi-k3/tokenizer.json + llm_tokenizer.py 文档更新 + 端到端验收

**Files:**
- Create: `kimi-k3/tokenizer.json`（脚本产物）
- Modify: `llm_tokenizer.py:1-4`（docstring）、`llm_tokenizer.py:128`（`--dir` help 文本）

**Interfaces:**
- Consumes: Task 3 完成的转换脚本
- Produces: `python llm_tokenizer.py --dir kimi-k3 "文本"` 可用

- [ ] **Step 1: 运行转换生成 kimi-k3/tokenizer.json（带校验）**

```powershell
cd f:\MyGit\llm-tokenizer
python scripts/export_tiktoken_tokenizer_json.py --vocab-file kimi-k3/tiktoken.model --pattern kimi-k3 --tokenizer-config kimi-k3/tokenizer_config.json -o kimi-k3/tokenizer.json --verify
```

Expected: 输出 `Wrote Fast Tokenizer JSON to kimi-k3\tokenizer.json` 和 `Verified 13 texts against tiktoken reference`，退出码 0。

- [ ] **Step 2: 更新 llm_tokenizer.py 文档**

将文件头部注释：

```python
# pip3 install transformers tokenizers
# python3 llm_tokenizer.py --dir ds-v4 "你的字符串"
# python3 llm_tokenizer.py --dir ds-v3 "你的字符串"
# python3 llm_tokenizer.py "你的字符串"          # 默认使用 ds-v4
```

改为：

```python
# pip3 install tokenizers
# python3 llm_tokenizer.py --dir ds-v4 "你的字符串"
# python3 llm_tokenizer.py --dir ds-v3 "你的字符串"
# python3 llm_tokenizer.py --dir kimi-k3 "你的字符串"
# python3 llm_tokenizer.py "你的字符串"          # 默认使用 ds-v4
```

将 `--dir` 的 help：

```python
        help="模型分词器目录名，默认: ds-v4（可选: ds-v3, ds-v4 等）"
```

改为：

```python
        help="模型分词器目录名，默认: ds-v4（可选: ds-v3, ds-v4, kimi-k3 等）"
```

- [ ] **Step 3: 端到端验证 llm_tokenizer.py**

```powershell
python llm_tokenizer.py --dir kimi-k3 "你好世界"
```

Expected: 打印 `[加载分词器] kimi-k3（Byte-Level BPE）` 及 Token IDs/Tokens/切分。

再与 tiktoken 参考交叉比对一次：

```powershell
python -c "import sys; sys.path.insert(0, 'scripts'); from export_tiktoken_tokenizer_json import *; from tokenizers import Tokenizer; ranks = load_mergeable_ranks('kimi-k3/tiktoken.model'); special = load_special_tokens('kimi-k3/tokenizer_config.json', len(ranks)); ref = build_reference_encoding(ranks, PATTERNS['kimi-k3'], special); tok = Tokenizer.from_file('kimi-k3/tokenizer.json'); text = '你好世界 Hello 1234'; assert tok.encode(text).ids == ref.encode(text, allowed_special='all'), 'MISMATCH'; print('llm_tokenizer 产物与 tiktoken 参考一致')"
```

Expected: 输出一致提示，无 AssertionError。

- [ ] **Step 4: 全量回归**

```powershell
python -m unittest tests.test_export_tiktoken_tokenizer_json -v
python llm_tokenizer.py --dir ds-v4 "你好世界"
```

Expected: 测试全 PASS；ds-v4 正常输出不回归。

- [ ] **Step 5: Commit**

```powershell
git add kimi-k3/tokenizer.json llm_tokenizer.py
git commit -m "feat: add kimi-k3 tokenizer.json and document --dir kimi-k3"
```

---

## Self-Review 记录

- **Spec 覆盖**：spec 改动点 1（复制文件）→ Task 1 Step 1；改动点 2（脚本扩展四项参数 + verify）→ Task 1–3；改动点 3（llm_tokenizer.py 文档）→ Task 4 Step 2；改动点 4（测试）→ 各 Task 测试步骤；验收标准 1/2/3 → Task 4 Step 3/4。无遗漏。
- **Placeholder 扫描**：所有代码步骤均含完整代码，无 TBD/TODO。
- **类型一致性**：`PATTERNS` / `load_mergeable_ranks` / `load_special_tokens` / `build_reference_encoding` / `verify_against_reference` / `VERIFY_CORPUS` 在 Task 1/3 定义，Task 2/3/4 消费处签名一致；`build_fast_tokenizer_json` 前三个位置参数保持现状兼容（Task 3 测试中 `build_fast_tokenizer_json(None, False, False, vocab_file=...)` 与之匹配）。
- **已知风险**：`--verify` 语料若不一致，先怀疑 fancy-regex 对 `&&` 交集/`(?i:)` 的兼容性，再怀疑 merges 恢复；按 systematic-debugging 处理，不得跳过放行。
