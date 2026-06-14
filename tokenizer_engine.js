/**
 * LLM Tokenizer — 独立分词引擎
 * 
 * 支持浏览器（全局变量）和 Node.js（require）两种环境。
 * 
 * 实现：GPT-2 风格 Byte-Level BPE + Metaspace BPE
 * 
 * 浏览器用法：
 *   <script src="tokenizer.js"></script>
 *   <script>
 *     const tok = new BPETokenizer(data);
 *     const result = tok.encode("Hello world!");
 *   </script>
 * 
 * Node.js 用法：
 *   const { BPETokenizer } = require('./tokenizer.js');
 *   const fs = require('fs');
 *   const data = JSON.parse(fs.readFileSync('./ds-v4/tokenizer.json', 'utf-8'));
 *   const tok = new BPETokenizer(data);
 *   console.log(tok.encode("Hello world!"));
 */

// ============ Byte-Level BPE 字节映射表 ============

function buildBytesToUnicode() {
    const bs = [];
    const cs = [];
    // 可打印区间直接映射
    for (let i = 33; i <= 126; i++)  { bs.push(i); cs.push(i); }  // !...~
    for (let i = 161; i <= 172; i++) { bs.push(i); cs.push(i); }  // ¡...¬
    for (let i = 174; i <= 255; i++) { bs.push(i); cs.push(i); }  // ®...ÿ
    // 不可打印字节映射到 U+0100 往后
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(256 + n);
            n++;
        }
    }
    const byteToUni = {};
    const uniToByte = {};
    for (let i = 0; i < 256; i++) {
        byteToUni[bs[i]] = String.fromCharCode(cs[i]);
        uniToByte[cs[i]] = bs[i];
    }
    return { byteToUni, uniToByte };
}

const { byteToUni, uniToByte } = buildBytesToUnicode();

// ============ BPE Tokenizer 类 ============

class BPETokenizer {
    constructor(data) {
        this.vocab = data.model?.vocab || {};
        this.merges = data.model?.merges || [];
        this.specialTokens = data.added_tokens || [];
        this.byteLevel = (data.decoder?.type === 'ByteLevel');

        // Metaspace 配置（NLLB-200 等 SentencePiece 风格）
        if (data.pre_tokenizer?.type === 'Metaspace') {
            this.metaspaceReplacement = data.pre_tokenizer.replacement || '▁';
            this.addPrefixSpace = data.pre_tokenizer.add_prefix_space || false;
        }

        // id → token string
        this.idToToken = {};
        for (const [token, id] of Object.entries(this.vocab)) {
            this.idToToken[id] = token;
        }
        for (const st of this.specialTokens) {
            this.vocab[st.content] = st.id;
            this.idToToken[st.id] = st.content;
        }

        // merge 优先级表: "a b" → rank (越小越优先)
        this.mergeRank = {};
        for (let i = 0; i < this.merges.length; i++) {
            this.mergeRank[this.merges[i]] = i;
        }

        // 预构建特殊 token 排序列表（最长匹配优先）
        this._specialMap = [];
        for (const st of this.specialTokens) {
            this._specialMap.push({ content: st.content, id: st.id, len: st.content.length });
        }
        this._specialMap.sort((a, b) => b.len - a.len);
    }

    // ---- BPE 核心算法（支持字符串或字符数组输入） ----
    bpe(symbols) {
        if (typeof symbols === 'string') {
            symbols = [...symbols];
        }
        if (symbols.length <= 1) {
            const merged = symbols[0];
            if (this.vocab[merged] !== undefined) return [merged];
            return [];
        }
        let word = symbols.slice();
        while (word.length > 1) {
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
            if (bestIdx === -1) break;
            word[bestIdx] = word[bestIdx] + word[bestIdx + 1];
            word.splice(bestIdx + 1, 1);
        }
        return word;
    }

    // ---- Byte-Level: UTF-8 字节 → unicode 字符串 ----
    bytesToUnicodeStr(bytes) {
        let result = '';
        for (const b of bytes) {
            result += byteToUni[b];
        }
        return result;
    }

    // ---- Byte-Level: 将 unicode token 字符串解码为原始文本 ----
    decodeTokenByteLevel(tokenStr) {
        const bytes = [];
        for (const ch of tokenStr) {
            const code = ch.codePointAt(0);
            if (uniToByte[code] !== undefined) {
                bytes.push(uniToByte[code]);
            }
        }
        if (bytes.length === 0) return tokenStr;
        return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
    }

    // ---- 通用: 将 token 字符串解码为可读文本 ----
    decodeToken(tokenStr) {
        if (this.byteLevel) {
            return this.decodeTokenByteLevel(tokenStr);
        }
        // Metaspace 等非 ByteLevel：▁ → 空格
        return tokenStr.replace(/▁/g, ' ');
    }

    // GPT-2 风格预分词正则（仅 ByteLevel 分词器使用）
    static get pat() {
        if (!this._pat) {
            this._pat = new RegExp(
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
                'gu'
            );
        }
        return this._pat;
    }

    // 检查并处理特殊 token
    _matchSpecial(remaining) {
        for (const sp of this._specialMap) {
            if (remaining.startsWith(sp.content)) {
                return sp;
            }
        }
        return null;
    }

    // ---- 编码文本片段（不含特殊 token），用于 ByteLevel ---- 
    _encodeSegmentByteLevel(segment, tokens, ids) {
        let remaining = segment;
        while (remaining.length > 0) {
            BPETokenizer.pat.lastIndex = 0;
            const match = BPETokenizer.pat.exec(remaining);
            if (match) {
                const word = match[0];
                const wordBytes = new TextEncoder().encode(word);
                const wordUnicode = this.bytesToUnicodeStr(wordBytes);
                const bpeTokens = this.bpe(wordUnicode);
                for (const bt of bpeTokens) {
                    const id = this.vocab[bt];
                    tokens.push(bt);
                    ids.push(id !== undefined ? id : 0);
                }
                remaining = remaining.substring(word.length);
            } else {
                remaining = remaining.substring(1);
            }
        }
    }

    // ---- Byte-Level BPE 编码 ----
    encodeByteLevel(text) {
        const tokens = [];
        const ids = [];
        let remaining = text;

        while (remaining.length > 0) {
            // 优先匹配开头的特殊 token
            const matchedSpecial = this._matchSpecial(remaining);
            if (matchedSpecial) {
                tokens.push(matchedSpecial.content);
                ids.push(matchedSpecial.id);
                remaining = remaining.substring(matchedSpecial.len);
                continue;
            }

            // 检查即将到来的文本中是否包含特殊 token（最长匹配优先）
            let nextSpecialPos = remaining.length;
            let nextSpecial = null;
            for (const sp of this._specialMap) {
                const pos = remaining.indexOf(sp.content);
                if (pos !== -1 && pos < nextSpecialPos) {
                    nextSpecialPos = pos;
                    nextSpecial = sp;
                }
            }

            // 如果下一个特殊 token 前面还有文本，先处理前面的文本
            if (nextSpecial && nextSpecialPos > 0) {
                const segment = remaining.substring(0, nextSpecialPos);
                this._encodeSegmentByteLevel(segment, tokens, ids);
                remaining = remaining.substring(nextSpecialPos);
                continue;
            }

            // 没有特殊 token，直接编码剩余文本
            this._encodeSegmentByteLevel(remaining, tokens, ids);
            break;
        }
        return { tokens, ids };
    }

    // ---- Metaspace BPE 编码（NLLB-200 等 SentencePiece 风格） ----
    encodeMetaspace(text) {
        const tokens = [];
        const ids = [];

        // 1. 空格 → ▁
        let processed = text;
        if (this.addPrefixSpace) {
            processed = '▁' + processed;
        }
        processed = processed.replace(/ /g, this.metaspaceReplacement || '▁');

        // 2. 按 ▁ 边界分片，每片分别做 BPE
        let remaining = processed;

        while (remaining.length > 0) {
            const matchedSpecial = this._matchSpecial(remaining);
            if (matchedSpecial) {
                tokens.push(matchedSpecial.content);
                ids.push(matchedSpecial.id);
                remaining = remaining.substring(matchedSpecial.len);
                continue;
            }

            // 检查即将到来的文本中是否包含特殊 token（最长匹配优先）
            let nextSpecialPos = remaining.length;
            let nextSpecial = null;
            for (const sp of this._specialMap) {
                const pos = remaining.indexOf(sp.content);
                if (pos !== -1 && pos < nextSpecialPos) {
                    nextSpecialPos = pos;
                    nextSpecial = sp;
                }
            }

            // 如果下一个特殊 token 前面还有文本，先处理前面的文本
            if (nextSpecial && nextSpecialPos > 0) {
                let segment = remaining.substring(0, nextSpecialPos);
                // 处理 segment 中的 Metaspace 边界
                while (segment.length > 0) {
                    const nextMeta = segment.indexOf('▁', 1);
                    let piece;
                    if (nextMeta === -1) {
                        piece = segment;
                        segment = '';
                    } else {
                        piece = segment.substring(0, nextMeta);
                        segment = segment.substring(nextMeta);
                    }
                    if (piece.length === 0) {
                        segment = segment.substring(1);
                        continue;
                    }
                    const bpeTokens = this.bpe([...piece]);
                    for (const bt of bpeTokens) {
                        const id = this.vocab[bt];
                        tokens.push(bt);
                        ids.push(id !== undefined ? id : 0);
                    }
                }
                remaining = remaining.substring(nextSpecialPos);
                continue;
            }

            // 找到下一个 ▁ 边界（跳过位置 0 的 ▁）
            const nextMeta = remaining.indexOf('▁', 1);
            let piece;
            if (nextMeta === -1) {
                piece = remaining;
                remaining = '';
            } else {
                piece = remaining.substring(0, nextMeta);
                remaining = remaining.substring(nextMeta);
            }

            if (piece.length === 0) {
                remaining = remaining.substring(1);
                continue;
            }

            const bpeTokens = this.bpe([...piece]);
            for (const bt of bpeTokens) {
                const id = this.vocab[bt];
                tokens.push(bt);
                ids.push(id !== undefined ? id : 0);
            }
        }
        return { tokens, ids };
    }

    // ---- 统一入口 ----
    encode(text) {
        if (this.byteLevel) {
            return this.encodeByteLevel(text);
        }
        return this.encodeMetaspace(text);
    }

    // ---- 解码：token IDs → 文本 ----
    decode(ids) {
        const tokens = [];
        for (const id of ids) {
            const token = this.idToToken[id];
            if (token !== undefined) {
                tokens.push(this.decodeToken(token));
            }
        }
        return tokens.join('');
    }
}

// ============ 环境适配：浏览器全局 + Node.js module.exports ============

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BPETokenizer, buildBytesToUnicode, byteToUni, uniToByte };
}
// 浏览器环境：BPETokenizer / byteToUni / uniToByte 已是全局变量（var/const 在顶层 script 中）
