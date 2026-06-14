#!/usr/bin/env python3
"""
将 tokenizer.json 转换为 tokenizer.js，供 index.html 在 file:// 协议下加载使用。
精简掉不必要的字段，只保留 vocab、merges 和 added_tokens。

用法:
    python build_tokenizer_js.py ds-v3 ds-v4 qwen ...
"""

import json
import sys

model_dirs = sys.argv[1:] if len(sys.argv) > 1 else ['ds-v3', 'ds-v4']

for model in model_dirs:
    json_path = f'{model}/tokenizer.json'
    js_path = f'{model}/tokenizer.js'

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    slim = {
        'model': {
            'vocab': data['model']['vocab'],
            'merges': data['model']['merges']
        },
        'added_tokens': data.get('added_tokens', []),
        'decoder': data.get('decoder', {}),
        'pre_tokenizer': data.get('pre_tokenizer', {})
    }

    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(f'window.__TOKENIZER_DATA__ = {json.dumps(slim, ensure_ascii=False)};')

    raw_mb = __import__('os').path.getsize(json_path) / 1024 / 1024
    slim_mb = __import__('os').path.getsize(js_path) / 1024 / 1024
    print(f'{model}: {raw_mb:.1f}MB -> {slim_mb:.1f}MB  ✓')
