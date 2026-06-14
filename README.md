# LLM Tokenizer 工具

通用的大语言模型分词器测试工具，支持 DeepSeek-V3、DeepSeek-V4 以及任意 GPT-2 风格 byte-level BPE 分词器。

支持 **Python 命令行**、**网页可视化** 和 **Node.js 测试** 三种使用方式。

---

## 目录结构

```
├── llm_tokenizer.py      # Python 命令行工具
├── tokenizer_engine.js   # 独立分词引擎（浏览器 / Node.js 通用）
├── test_tokenizer.js     # Node.js 测试脚本
├── index.html            # 网页版可视化工具
├── ds-v3/                # DeepSeek-V3 分词器
│   ├── tokenizer.json    # 完整分词器数据（供 Python 读取）
│   └── tokenizer.js      # 精简分词器数据（供网页 file:// 模式读取）
├── ds-v4/                # DeepSeek-V4 分词器
│   ├── tokenizer.json
│   └── tokenizer.js
├── nllb-200/             # NLLB-200 分词器（Metaspace BPE）
│   ├── tokenizer.json
│   └── tokenizer.js
└── README.md
```

> `tokenizer_config.json` 为 HuggingFace transformers 格式的元数据，当前工具不依赖，可按需删除。

---

## 方式一：Python 命令行

### 安装依赖

```bash
pip install tokenizers
```

### 用法

```bash
# 默认使用 ds-v4 分词器
python llm_tokenizer.py "你好世界，Hello World!"

# 指定模型目录
python llm_tokenizer.py --dir ds-v3 "你好世界"

# 交互模式
python llm_tokenizer.py --dir ds-v4

# 查看帮助
python llm_tokenizer.py -h
```

### 输出示例

```
[加载分词器] ds-v4
输入:   '你好世界'
Token IDs: [27746, 8007, 34965, 35948, 60713, 29547, 49431, 30681, 41137, 83574, 3063]
Tokens:    ['å', 'è½', 'è', '¨', 'ä', '¶', 'è', '¨', 'è', '§', 'è']
切分:      ['你', '好', '世', '界']
```

---

## 方式二：网页版可视化

### 直接打开（推荐）

直接双击 `index.html` 用浏览器打开即可。内置 DeepSeek-V3 / V4 切换，也支持加载自定义 `tokenizer.json`。

> 网页版在 `file://` 协议下通过 `<script>` 标签加载分词数据，首次加载可能需要几秒（数据约 4.5MB）。

### 通过 HTTP 服务器打开（可选）

```bash
python -m http.server 8080
# 浏览器打开 http://localhost:8080/index.html
```

### 网页功能

- **模型切换** — 点击 DeepSeek-V4 / DeepSeek-V3 按钮切换分词器
- **自定义加载** — 支持上传任意 GPT-2 风格的 `tokenizer.json`
- **实时分词** — 输入文本自动分词，无需手动触发
- **TEXT / TOKEN IDS** — 切换显示可读文本和 Token ID
- **颜色高亮** — 每个 token 用不同颜色标注，hover 查看详情
- **统计数据** — 显示 Token 数量、字符数量、Token/字符比

---

## 添加新模型

1. 新建子目录，如 `qwen/`
2. 放入对应的 `tokenizer.json`
3. 运行生成 `.js` 文件：

```bash
python build_tokenizer_js.py qwen
```

4. 在 `index.html` 中 `<div class="model-selector">` 内添加按钮。
5. Python 端无需额外配置，直接用 `--dir qwen` 即可。

---

## 方式三：Node.js 测试

适用于开发调试和自动化测试，无需浏览器。

```bash
# 测试默认模型（ds-v4）
node test_tokenizer.js

# 测试指定模型
node test_tokenizer.js ds-v4 nllb-200

# 编程调用
node -e "
const { BPETokenizer } = require('./tokenizer_engine.js');
const data = JSON.parse(require('fs').readFileSync('./ds-v4/tokenizer.json','utf-8'));
const tok = new BPETokenizer(data);
console.log(tok.encode('Hello world!'));
"
```

---

## 技术说明

- 分词算法为 GPT-2 风格的 Byte-Level BPE（字节级 BPE），同时支持 Metaspace BPE（如 NLLB-200）
- 每个原始字节映射到一个可见 Unicode 字符，再通过 BPE 合并规则组合成 token
- 特殊 token（如 `<｜User｜>`）在编码时按最长匹配优先处理
- Python 版使用 HuggingFace `tokenizers` 库，网页版 / Node.js 共用 `tokenizer_engine.js` 纯 JavaScript 实现
