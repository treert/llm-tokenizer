# DeepSeek Tokenizer 配置对比：DS-V3 vs DS-V4

> 生成日期：2026-06-15

---

## 一、核心 BPE 模型：完全相同

基础 token 的编码逻辑完全一致。相同的纯文本在两个 tokenizer 下会产出相同的 token ID。

| 参数 | ds-v3 | ds-v4 |
|------|-------|-------|
| Vocab 大小 | 128,000 | 128,000 |
| Merge 规则数 | 127,741 | 127,741 |
| Pre-tokenizer | 3 条（数字 / CJK / 通用） | 完全相同 |
| Normalizer | 空 Sequence | 完全相同 |
| Post-processor | ByteLevel | 完全相同 |
| Decoder | ByteLevel | 完全相同 |

---

## 二、配置参数差异

| 配置项 | ds-v3 | ds-v4 |
|--------|-------|-------|
| `model_max_length` | **16,384** (16K) | **1,048,576** (1M) — 64 倍 |
| `tokenizer_class` | `LlamaTokenizerFast` | `PreTrainedTokenizerFast` |
| `chat_template` | ✅ 复杂模板（含 tool calling） | ❌ 无 |

---

## 三、Added Tokens 差异

| | ds-v3 | ds-v4 |
|---|-------|-------|
| 数量 | 818 个 | **1,283 个** (+465) |

### 🔴 Token ID 冲突

ds-v3 和 ds-v4 在以下两个 ID 上存在内容冲突，**不能混用**：

| ID | ds-v3 | ds-v4 |
|----|-------|-------|
| 128798 | `<think>` | `<｜place▁holder▁no▁798｜>` |
| 128799 | `</think>` | `<｜place▁holder▁no▁799｜>` |

ds-v4 将 `<think>` / `</think>` 移动到了 ID **128821-128822**。

---

## 四、ds-v4 新增的 465 个 Token

### 1. 推理与搜索（~8 个）

`<think>`, `</think>`, `<｜search▁begin｜>`, `<｜search▁end｜>`, `<｜search｜>`, `<｜extracted_url｜>`, `<｜read_url｜>`, `<｜end_of_query｜>`

### 2. DSML 结构化输出（~20 个）

`｜DSML｜`, `<dsml:`, `</dsml:`, `<｜task｜>`, `<｜title｜>`, `<｜answer｜>`, `<｜authority｜>`, `<｜domain｜>`, `<｜political｜>`, `<｜entity｜>`, `<｜safety｜>`, `<｜action｜>`, `<｜query｜>`, `<｜latest_reminder｜>`

### 3. 系统 / 仓库 / 文件（~8 个）

`<｜begin▁of▁repo▁name｜>`, `<｜end▁of▁repo▁name｜>`, `<｜begin▁of▁file▁name｜>`, `<｜end▁of▁file▁name｜>`, `<｜begin▁of▁file｜>`, `<｜end▁of▁file｜>`, `<｜begin▁sys｜>`, `<｜end▁sys｜>`

### 4. 多模态占位符（~415 个）

`<|place_holder_mm_span_0021|>` ~ `<|place_holder_mm_span_0435|>`，用于视觉多模态 patch/span 占位。

### 5. 多模态图像标记（4 个）

`<｜image｜>`, `<｜image2｜>`, `<｜rl_image_pad｜>`, `<｜rl_image_start｜>`

### 6. 表格标记（6 个）

`<｜table｜>`, `<｜/table｜>`, `<｜tr｜>`, `<｜/tr｜>`, `<｜td｜>`, `<｜/td｜>`

### 7. 空间坐标标记（8 个）

`<｜polygon｜>`, `<｜/polygon｜>`, `<｜point｜>`, `<｜/point｜>`, `<｜box｜>`, `<｜/box｜>`, `<｜ref｜>`, `<｜/ref｜>`

---

## 五、总结

1. **基础编码完全兼容** — 普通文本 tokenize 结果相同
2. **ds-v4 是多模态版本** — 新增视觉 / 表格 / 坐标 / DSML 结构化输出专用 token
3. **不可混用** — `<think>` 等 token ID 冲突，两个 tokenizer 模型不可互换
4. **上下文窗口扩大 64 倍** — 16K → 1M，支持超长上下文
5. **chat_template 移除** — 对话格式化交由推理框架处理
