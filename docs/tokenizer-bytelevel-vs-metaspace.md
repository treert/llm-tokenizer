# ByteLevel BPE vs Metaspace BPE 分词器原理对比

基于本项目 `llm-tokenizer` 的源码分析，对比 DeepSeek-V4（ByteLevel BPE）和 NLLB-200（Metaspace BPE）两种分词器的核心差异。

---

## 一、核心流程图

```
输入文本 "🔤💡"
         │
         │ encode()
         ├──────────────────────────────┬──────────────────────────────┐
         ▼                              ▼                              │
  this.byteLevel?                  this.byteLevel?                    │
    YES ──► encodeByteLevel()      NO ──► encodeMetaspace()           │
         (DeepSeek-V4)                   (NLLB-200)                   │
         │                              │                              │
         ▼                              ▼                              │
  GPT-2 regex 分词               addPrefixSpace + 空格→▁              │
         │                              │                              │
         ▼                              ▼                              │
  TextEncoder → UTF-8 字节       [...piece] → Unicode 码点            │
         │                              │                              │
         ▼                              ▼                              │
  bytesToUnicodeStr()             BPE(码点数组)                        │
  (字节 → 映射的 Unicode 字符)           │                              │
         │                              │                              │
         ▼                              ▼                              │
  BPE(映射字符数组)              vocab[d] → id / 0(unk)               │
         │                                                             │
         ▼                                                             │
  vocab[d] → id / 0(unk)                                              │
                                                                      │
  多头输出 vs 单 token 输出 ──────────────────────────────────────────┘
```

---

## 二、关键分岔点：`encode()` 入口

```javascript
// tokenizer_engine.js:328-333
encode(text) {
    if (this.byteLevel) {
        return this.encodeByteLevel(text);  // DeepSeek-V4
    }
    return this.encodeMetaspace(text);       // NLLB-200
}
```

配置驱动：

| 模型 | `decoder.type` | `pre_tokenizer.type` | 走哪个方法 |
|------|---------------|---------------------|-----------|
| DeepSeek-V4 | `ByteLevel` | `Sequence(Split, ByteLevel, ...)` | `encodeByteLevel()` |
| NLLB-200 | `Metaspace` | `Metaspace` | `encodeMetaspace()` |

---

## 三、ByteLevel BPE（DeepSeek-V4）：字节是原子

### 3.1 流程

```
文本 "我" → UTF-8 编码 → [0xE6, 0x88, 0x91] → 字节→Unicode映射 → 'æĪĳ' → BPE合并
```

### 3.2 代码定位

```javascript
// tokenizer_engine.js:173-193
_encodeSegmentByteLevel(segment, tokens, ids) {
    // ...
    const wordBytes = new TextEncoder().encode(word);      // ① UTF-8 字节
    const wordUnicode = this.bytesToUnicodeStr(wordBytes); // ② 字节→映射字符
    const bpeTokens = this.bpe(wordUnicode);               // ③ BPE 合并
    for (const bt of bpeTokens) {
        const id = this.vocab[bt];
        tokens.push(bt);
        ids.push(id !== undefined ? id : 0);
    }
}
```

### 3.3 字节映射规则

不可打印字节（0x00-0x20, 0x7F, 0x80-0xA0, 0xAD）被映射到 `U+0100` 往后的码点：

| 原始字节 | 映射字符 |
|---------|---------|
| 0x00 | `Ā` (U+0100) |
| 0x01 | `ā` (U+0101) |
| 0xE6 | `æ` (可打印区间，直接映射) |
| 0x88 | `Ī` (U+0100+) |
| 0x91 | `ĳ` (U+0100+) |

### 3.4 Emoji 处理

`🔤` 的 UTF-8 编码 = `[0xF0, 0x9F, 0x94, 0xA4]` → **4 个原子符号**进入 BPE。

BPE 需要将这些映射字符逐步合并，可能产生**多个 token**。

---

## 四、Metaspace BPE（NLLB-200）：Unicode 码点是原子

### 4.1 流程

```
文本 "🔤💡" → 加前缀 ▁ → '▁🔤💡' → [...'▁🔤💡'] → ['▁','🔤','💡'] → BPE合并
```

### 4.2 代码定位

```javascript
// tokenizer_engine.js:238-325
encodeMetaspace(text) {
    let processed = text;
    if (this.addPrefixSpace) {
        processed = '▁' + processed;           // ① 加前缀 ▁
    }
    processed = processed.replace(/ /g, '▁');  // ② 空格→▁

    // ③ 按 ▁ 边界分片
    // ...
    const bpeTokens = this.bpe([...piece]);    // ④ 按码点拆分，BPE 合并
    for (const bt of bpeTokens) {
        const id = this.vocab[bt];
        tokens.push(bt);
        ids.push(id !== undefined ? id : 0);
    }
}
```

### 4.3 Emoji 处理

`🔤` = **1 个 Unicode 码点**（U+1F524），BPE 输入只有 1 个原子符号。

如果 NLLB-200 vocab 中有 `🔤` → 直接 1 个 token 输出；没有 → `bpe()` 行为见下文。

---

## 五、对比总结

| | ByteLevel (DeepSeek-V4) | Metaspace (NLLB-200) |
|---|---|---|
| 原子单位 | UTF-8 单字节 (0-255) | Unicode 码点 |
| `🔤` 的原子数 | 4 个 | 1 个 |
| 中间编码 | 字节→Unicode 映射表 | 无 |
| Emoji token 数 | 可能多个 | vocab 中有则为 1 个 |
| 预分词 | GPT-2 正则 | ▁ 边界 |
| 空格表示 | 映射字节+GPT-2 正则 | `▁` 字符 |

---

## 六、BPE 算法细节：`bpe()` 方法

```javascript
// tokenizer_engine.js:92-118
bpe(symbols) {
    if (typeof symbols === 'string') {
        symbols = [...symbols];
    }
    if (symbols.length <= 1) {
        const merged = symbols[0];
        if (this.vocab[merged] !== undefined) return [merged];
        return [];  // ⚠️ 单字符未知 → 返回空数组
    }
    let word = symbols.slice();
    while (word.length > 1) {
        // 找最小 rank 的相邻符号对
        let minRank = Infinity;
        let bestIdx = -1;
        for (let i = 0; i < word.length - 1; i++) {
            const pair = word[i] + ' ' + word[i + 1];
            const rank = this.mergeRank[pair];
            if (rank !== undefined && rank < minRank) {
                minRank = rank;
                bestIdx = i;
            }
        }
        if (bestIdx === -1) break;  // 无 merge → 停止
        word[bestIdx] = word[bestIdx] + word[bestIdx + 1];
        word.splice(bestIdx + 1, 1);
    }
    return word;  // 返回剩余符号（可能部分合并）
}
```

### 单符号 vs 多符号的行为差异

| 输入 | 单符号+vocab中有 | 单符号+vocab中无 | 多符号+部分不在vocab |
|------|----------------|-----------------|-------------------|
| `bpe()` 返回 | `[符号]` | `[]`（丢弃） | 原样返回未合并的符号 |

> **对 NLLB-200 (Metaspace) 来说，单符号路径不可达。** 因为 `addPrefixSpace=true`，任何输入都会先加上 `▁` 前缀，`bpe()` 至少收到 2 个符号（`['▁', ...]`），永远走多符号路径。因此 NLLB-200 模式下，所有字符 — 即使 vocab 中没有 — 都会被保留在输出中（token 字符串原样，id=0）。
>
> ByteLevel 模式下没有前缀，单符号未知时才会触发 `return []`。

---

## 七、Python vs Web JS 未知字符处理差异

### NLLB-200 (Metaspace BPE)

因 `addPrefixSpace=true`，`▁` 永远在前，`bpe()` 最少 2 个符号，单符号丢弃路径不可达。

| | Python（HuggingFace 库） | Web JS（`tokenizer_engine.js`） |
|---|---|---|
| 未知字符 | 输出 `<unk>` token（id=3） | 保留原始符号字符串，id=0 |
| `</s>` EOS | 自动追加 | 不追加 |
| 实际输出 `🔤💡` | `▁ <unk> </s> <unk>` (4 tokens) | `▁ 🔤 💡` (3 tokens) |

### DeepSeek-V4 (ByteLevel BPE)

无前缀 `▁`，单符号路径可达。

| | Python（HuggingFace 库） | Web JS（`tokenizer_engine.js`） |
|---|---|---|
| 未知单字符 | 输出 `<unk>` token | `bpe()` 返回 `[]`，符号被丢弃 |
| 未知多符号 | 输出 `<unk>` token | 保留原始符号，id=0 | |

Python 版使用 HuggingFace 的 `tokenizers` 库：
```python
# llm_tokenizer.py:110
encoded = tokenizer.encode(text)
```
HuggingFace 库对 Metaspace 分词器有内置处理：不在 vocab 中的字符 → `<unk>`。

Web JS 简化实现：
```javascript
// tokenizer_engine.js:292-295
const id = this.vocab[bt];
tokens.push(bt);           // 保留原始符号字符串
ids.push(id !== undefined ? id : 0);  // id 为 0 (unk)
```

同样是不识别，Python 输出 `<unk>`，JS 保留原始字符但 id=0。

---

## 八、一句话本质

> ByteLevel BPE 把世界看成 **256 种字节** 的组合，Metaspace BPE 把世界看成 **Unicode 字符** 的组合。

这导致了 emoji 等多字节字符在两种分词器中的原子粒度完全不同。
