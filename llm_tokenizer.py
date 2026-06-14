# pip3 install transformers tokenizers
# python3 llm_tokenizer.py --dir ds-v4 "你的字符串"
# python3 llm_tokenizer.py --dir ds-v3 "你的字符串"
# python3 llm_tokenizer.py "你的字符串"          # 默认使用 ds-v4
import sys
import os
import argparse
from tokenizers import Tokenizer


def load_tokenizer(model_dir="ds-v4"):
    """加载指定模型目录下的 tokenizer.json，返回 (tokenizer, is_byte_level)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(base_dir, model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"找不到 tokenizer 文件: {tokenizer_path}")
    tok = Tokenizer.from_file(tokenizer_path)
    byte_level = _is_byte_level(tok)
    algo_name = "Byte-Level BPE" if byte_level else "BPE"
    print(f"[加载分词器] {model_dir}（{algo_name}）")
    return tok, byte_level


# --- GPT-2 风格 byte-level token → 可读文本 的转换工具 ---

def _build_byte_decoder():
    """构建 GPT-2 byte-level 的逆映射表：Unicode 字符 → 原始字节值 (0-255)

    GPT-2 的 Byte-level BPE 把每个原始字节 (0x00-0xFF) 映射到一个可见的
    Unicode 字符，这样才能把字节序列当作文本存进 BPE 词表里。

    映射规则：
      1. 可打印 ASCII 区间（如 ! 到 ~、¡ 到 ÿ）直接映射到自身，即 0x21→'!', 0x7E→'~'
      2. 不可打印字节（0x00-0x20）、DEL(0x7F)、以及区间间隙的字节，则按顺序
         映射到 U+0100 往后的码点，即 0x00→'Ā'(U+0100), 0x01→'ā'(U+0101)...

    举例 —— "我" 的 UTF-8 编码是 0xE6 0x88 0x91：
      0xE6 → 'æ' (在可打印区间，直接映射)
      0x88 → 'Ī' (不可打印，映射到 U+0100+?)
      0x91 → 'ĳ' (不可打印，映射到 U+0100+?)

    所以 BPE token 'æĪĳæĺ¯' 就对应字节序列 [0xE6,0x88,0x91, 0xE6,0x98,0xAF]，
    再 UTF-8 解码就是 "我是"。

    本函数构建的是逆向映射 {Unicode码点 → 原始字节}，方便把 token 字符串还原。
    """

    # --- 正向映射表：bytes[0..255] → unicode_codepoint[0..255] ---

    # 步骤1: 可打印字符区间，直接映射到自身
    #  !(0x21) ~ ~(0x7E)    33-126  标准 ASCII 可打印
    #  ¡(0xA1) ~ ¬(0xAC)    161-172  拉丁扩展
    #  ®(0xAE) ~ ÿ(0xFF)     174-255  拉丁扩展（跳过 0xAD 软连字符）
    bs = (
        list(range(ord("!"), ord("~") + 1))   # 0x21-0x7E: !"#$%...{|}~
        + list(range(ord("¡"), ord("¬") + 1)) # 0xA1-0xAC: ¡¢£¤¥...¬
        + list(range(ord("®"), ord("ÿ") + 1)) # 0xAE-0xFF: ®¯°±...ÿ
    )
    cs = bs[:]  # 这些码点直接映射到自身：chr(b) → b

    # 步骤2: 剩余的不可打印字节（0x00-0x20, 0x7F, 0xAD 等）
    # 按字节值从小到大依次映射到 U+0100 往后（即 256 + n）
    n = 0
    for b in range(256):
        if b not in bs:           # 该字节还没有映射
            bs.append(b)          #   ...记录这个字节
            cs.append(256 + n)    #   ...映射到 U+0100, U+0101, U+0102...
            n += 1

    # 最终 bs[i] = 原始字节值, cs[i] = Unicode 码点
    # 例如 bs[0]=0x00→cs[0]=256('Ā'), bs[33]=0x21→cs[33]=33('!')

    # --- 构建逆向映射：Unicode 码点 → 原始字节 ---
    # 输入字符 ch，通过 ord(ch) 查到对应原始字节值
    return {c: b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def _is_byte_level(tokenizer):
    """检测分词器是否为 ByteLevel（字节级 BPE）类型"""
    try:
        from tokenizers.decoders import ByteLevel
        return isinstance(tokenizer.decoder, ByteLevel)
    except Exception:
        return False


def bpe_token_to_text(token_str):
    """将单个 GPT-2 风格 byte-level token 字符串还原为可读文本"""
    byte_list = [_BYTE_DECODER[ord(ch)] for ch in token_str]
    return bytes(byte_list).decode("utf-8", errors="replace")


def metaspace_token_to_text(token_str):
    """将 Metaspace 风格 token 还原为可读文本，▁ 代表空格"""
    return token_str.replace("▁", " ")


# --- 主逻辑 ---

def process_text(tokenizer, text, byte_level):
    """处理并输出 token 信息"""
    encoded = tokenizer.encode(text)
    if byte_level:
        readable_tokens = [bpe_token_to_text(t) for t in encoded.tokens]
    else:
        readable_tokens = [metaspace_token_to_text(t) for t in encoded.tokens]
    print(f"输入:   {repr(text)}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Tokens:    {encoded.tokens}")
    print(f"切分:      {readable_tokens}")


def main():
    parser = argparse.ArgumentParser(
        description="通用 LLM Tokenizer 测试工具，支持多种模型分词器"
    )
    parser.add_argument(
        "--dir", "-d",
        default="ds-v4",
        help="模型分词器目录名，默认: ds-v4（可选: ds-v3, ds-v4 等）"
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="要分词的文本（可选，不填则进入交互模式）"
    )
    args = parser.parse_args()

    tokenizer, byte_level = load_tokenizer(args.dir)

    if not args.text:
        # 交互模式
        print("进入交互模式，输入字符串后回车查看 token 列表，输入 quit 退出。")
        print(f"当前模型目录: {args.dir}")
        while True:
            try:
                text = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            if text.strip().lower() == "quit":
                print("已退出。")
                break
            if not text.strip():
                continue
            process_text(tokenizer, text, byte_level)
    else:
        text = " ".join(args.text)
        process_text(tokenizer, text, byte_level)


if __name__ == "__main__":
    main()
