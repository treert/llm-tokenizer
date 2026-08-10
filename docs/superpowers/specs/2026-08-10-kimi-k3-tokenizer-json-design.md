# Kimi-K3 分词器接入设计（tiktoken.model → tokenizer.json）

日期：2026-08-10
状态：已获用户批准

## 背景

Kimi-K3 开源权重（`G:\llm-models\Kimi-K3`）不提供 HuggingFace `tokenizer.json`，而是：
- `tiktoken.model`：tiktoken 标准 BPE 词表（每行 `base64(token_bytes) rank`），163,584 条基础 token
- `tokenization_kimi.py`：官方 `TikTokenTokenizer`，含 K3 专用 `pat_str` 预分词正则
- `tokenizer_config.json`：`added_tokens_decoder` 定义 16 个已命名特殊 token（id 163584–163839 区间内），共 256 个保留特殊 token 槽位

本项目的 `llm_tokenizer.py` 统一通过 `<dir>/tokenizer.json`（HF tokenizers 库）加载分词器，`build_tokenizer_js.py` 也依赖 tokenizer.json 生成网页版数据。因此需要把 tiktoken 词表离线转换为标准 tokenizer.json。

## 目标

`python llm_tokenizer.py --dir kimi-k3 "文本"` 正常分词，且 token ids 与 tiktoken 参考实现完全一致。

非目标（后续单独任务）：网页版 / JS 引擎适配、XTML 聊天模板。

## 数据流

```
tiktoken.model
  → tiktoken.load.load_tiktoken_bpe → {token_bytes: rank}
  → vocab：bytes 经 GPT-2 bytes_to_unicode 映射为可见字符串，id = rank
  → merges：按 rank 升序遍历，对每个多字节 token 用现有 recover_bpe_merges
    （在词表内贪心模拟合并，直到恰好剩 2 段即记为一条 merge；不足 2 段则跳过）
  → pre_tokenizer：Sequence[Split(Regex(pat_str), behavior=isolated, invert=false),
                             ByteLevel(add_prefix_space=false, use_regex=false)]
  → decoder：ByteLevel
  → added_tokens：tokenizer_config.json 的 added_tokens_decoder 16 个命名 token
    + fill_vocab_holes 以 <|reserved_token_{i}|> 补齐空洞（与官方命名一致）
  → kimi-k3/tokenizer.json
```

## 改动点

### 1. 复制模型文件

- `G:\llm-models\Kimi-K3\tiktoken.model` → `kimi-k3/tiktoken.model`（转换源）
- `G:\llm-models\Kimi-K3\tokenizer_config.json` → `kimi-k3/tokenizer_config.json`（特殊 token 元数据；与其他目录保持一致）

### 2. 扩展 `scripts/export_tiktoken_tokenizer_json.py`

复用现有转换管线（vocab 生成、merges 恢复、fill_vocab_holes、compact/array-merges 输出均不变），新增参数：

- `--vocab-file PATH`：用 `tiktoken.load.load_tiktoken_bpe` 加载自定义词表，与 `--encoding` 互斥
- `--pattern NAME`：内置 pattern 注册表，新增 `"kimi-k3"` 预设，pat_str 原样取自官方
  `tokenization_kimi.py`（8 条 alternation，含 `[\p{Han}]+`、`[...&&[^\p{Han}]]` 交集、
  `(?i:'s|...)` 内联 flag、`\p{N}{1,3}`）；`--encoding` 模式下默认用 tiktoken encoding 自带 pattern
- `--tokenizer-config PATH`：读取 `added_tokens_decoder`，按其中 `content`/`special` 命名特殊 token；
  缺省时所有保留槽位都用 `<|reserved_token_{i}|>`
- `--verify`：转换完成后构建 tiktoken 参考 `Encoding`（同 pat_str + 全部 256 个特殊 token），
  对内置测试语料逐条比对 `encode` ids，不一致则打印差异并以非零码退出

最终命令：

```bash
python scripts/export_tiktoken_tokenizer_json.py \
  --vocab-file kimi-k3/tiktoken.model \
  --pattern kimi-k3 \
  --tokenizer-config kimi-k3/tokenizer_config.json \
  -o kimi-k3/tokenizer.json --verify
```

### 3. `llm_tokenizer.py`

逻辑零改动（`Tokenizer.from_file` + 现有 ByteLevel 检测直接兼容），仅更新 docstring 与 `--dir` 帮助文本提及 kimi-k3。

### 4. 测试 `tests/test_export_tiktoken_tokenizer_json.py`

新增 K3 用例：执行转换（子进程），加载产物与 tiktoken 参考 Encoding 比对语料：
- 中文长句、中英混排、英文大小写与缩写（`'s`/`'T` 等）
- 代码片段（含缩进、连续换行）
- 数字：1–3 位、4 位以上连写
- emoji、标点、连续空白
- 含 `<|open|>`、`[BOS]` 字面量的文本（验证特殊 token 匹配行为与参考一致）

现有 o200k 用例不改动，必须保持通过。

## 风险与对策

| 风险 | 对策 |
|---|---|
| pat_str 的 `&&` 字符类交集、`(?i:)` 内联 flag 在 tokenizers 底层（fancy-regex）行为差异 | `--verify` 与单元测试全量比对兜底；若 fancy-regex 不兼容则改写为等价 pattern |
| merges 恢复算法在 K3 词表上的正确性 | 该算法已在 o200k 上有对照测试；K3 转换后同样语料比对 |
| 特殊 token 匹配行为差异（tokenizers 的 added_tokens vs tiktoken 的 allowed_special） | 测试语料覆盖特殊 token 字面量 |

## 验收标准

1. `python llm_tokenizer.py --dir kimi-k3 "你好世界"` 输出 ids 与 tiktoken 参考一致
2. `python -m unittest tests.test_export_tiktoken_tokenizer_json` 全部通过（含新增 K3 用例）
3. `python llm_tokenizer.py --dir ds-v4 "你好世界"` 等既有用法不回归
