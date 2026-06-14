/**
 * tokenizer_engine.js 的 Node.js 测试脚本
 * 
 * 用法：
 *   node test_tokenizer.js
 *   node test_tokenizer.js ds-v4          # 指定模型
 *   node test_tokenizer.js ds-v4 nllb-200  # 测试多个模型
 */

const { BPETokenizer } = require('./tokenizer_engine.js');
const fs = require('fs');
const path = require('path');

const models = process.argv.length > 2 ? process.argv.slice(2) : ['ds-v4'];

const testCases = [
    'Hello, world!',
    '你好世界',
    '这是一个测试句子，用来演示 Tokenizer 的分词效果。',
    'function hello() {\n  console.log("Hello, World!");\n}',
    '<｜User｜>你好，请介绍一下自己。<｜Assistant｜>',
    'Deep learning (深度学习) is a branch of machine learning.',
];

for (const model of models) {
    const jsonPath = path.join(__dirname, model, 'tokenizer.json');
    if (!fs.existsSync(jsonPath)) {
        console.error(`[跳过] 找不到 ${jsonPath}`);
        continue;
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`模型: ${model}`);
    console.log(`${'='.repeat(60)}`);

    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const tok = new BPETokenizer(data);

    let passed = 0;
    let failed = 0;

    for (const text of testCases) {
        const result = tok.encode(text);
        const decoded = tok.decode(result.ids);

        // ID 数量范围检查（合法 tokenizer 至少产出 1 个 token)
        const idCountOk = result.tokens.length > 0 && result.tokens.length === result.ids.length;

        // 单 token 直接查表
        const vocabCheck = result.tokens.every(t => {
            if (tok.specialTokens.some(st => st.content === t)) return true;
            return tok.vocab[t] !== undefined;
        });

        const status = (idCountOk && vocabCheck) ? '✓' : '✗';
        if (status === '✓') passed++;
        else failed++;

        console.log(`  ${status} "${text.slice(0, 30)}${text.length > 30 ? '...' : ''}"`);
        console.log(`    tokens: ${result.tokens.length}  ids: [${result.ids.slice(0, 10).join(', ')}${result.ids.length > 10 ? ', ...' : ''}]`);
        if (result.tokens.length <= 10) {
            console.log(`    raw tokens: ${JSON.stringify(result.tokens)}`);
        }
    }

    // 往返一致性检查（decode(encode(text)) ≈ text）
    console.log(`\n  往返测试 (round-trip):`);
    let rtOk = 0;
    let rtFail = 0;
    for (const text of testCases) {
        const encoded = tok.encode(text);
        const decoded = tok.decode(encoded.ids);
        // 将空格归一化比较（BPE 可能改变空白）
        const normalizedText = text.replace(/\s+/g, ' ').trim();
        const normalizedDecoded = decoded.replace(/\s+/g, ' ').trim();
        const ok = normalizedText === normalizedDecoded;
        if (ok) rtOk++;
        else rtFail++;
        if (!ok) {
            console.log(`    ✗ 往返失败: "${text}" -> "${decoded}"`);
        }
    }
    console.log(`    ✓ ${rtOk} / ${testCases.length} 通过${rtFail > 0 ? ` (${rtFail} 失败)` : ''}`);

    console.log(`\n  编码测试: ${passed}/${testCases.length} 通过`);
}
