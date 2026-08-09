import re

# 匹配需要从单词里拆出来的标点符号，这里使用正则表达式的 compile 方法把字符串编译为正则表达式对象，方便后续使用。
_EN_PUNCT_PATTERN = re.compile(r"([.,!?;:\"()])")

def tokenize_en(sentence: str) -> list:
    # 去掉首尾空格，并把所有字母转为小写。
    sentence = sentence.strip().lower()
    # 在标点前后插入空格，这样后面按空格 split 就能把标点拆成独立 token。sub 是正则替换函数。
    sentence = _EN_PUNCT_PATTERN.sub(r" \1 ", sentence)
    tokens = sentence.split()
    return tokens

def tokenize_zh(sentence: str) -> list:
    sentence = sentence.strip()
    # 把连续空白符用 sub 替换为单个无内容，也就是把所有空白符删掉。
    sentence = re.sub(r"\s+", "", sentence)
    tokens = list(sentence)
    return tokens

def tokenize(sentence: str, lang: str) -> list:
    if lang == "en":
        return tokenize_en(sentence)
    elif lang == "zh":
        return tokenize_zh(sentence)
    else:
        raise ValueError(f"暂不支持的语言类型: {lang!r}，目前只支持 'en' 和 'zh'")