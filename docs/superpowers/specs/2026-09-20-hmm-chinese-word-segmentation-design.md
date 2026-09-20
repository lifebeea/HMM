# 基于 HMM 的中文分词工程设计方案

## 1. 目标

在空白的“项目二”目录中建立一个模块化 Python 工程，实现以下能力：

- 使用本地中文语料训练 HMM 的初始概率、转移概率和发射概率。
- 使用 BMES 标注体系和 Viterbi 算法完成中文分词。
- 提供文本清洗、分句、语句分词和篇章分词能力。
- 提供可复现的训练入口、交互入口、自动化命令和测试。
- 在首次运行模型不存在时自动训练，后续直接加载已训练模型。

训练语料使用相邻 `NLP学习` 目录中的《倚天屠龙记》文本。实施时将语料复制到工程内，项目运行不依赖原始相邻目录。

## 2. 工程结构

```text
项目二/
├── data/
│   └── raw/
│       └── train.txt
├── models/
│   └── hmm_model.json
├── src/
│   └── hmm_segmenter/
│       ├── __init__.py
│       ├── preprocess.py
│       ├── trainer.py
│       ├── model.py
│       ├── viterbi.py
│       └── cli.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   ├── test_viterbi.py
│   └── test_cli.py
├── main.py
├── requirements.txt
└── README.md
```

各模块职责：

- `preprocess.py`：文本清洗、分句。
- `trainer.py`：语料读取、BMES 标注生成、概率统计和模型训练。
- `model.py`：HMM 模型定义、概率查询、模型保存和加载。
- `viterbi.py`：Viterbi 初始化、递推、回溯和 BMES 词序列还原。
- `cli.py`：交互菜单、文件分词、命令行子命令和错误处理。
- `main.py`：工程统一入口。

## 3. 核心接口

```python
clean_text(text: str) -> str
split_sentences(text: str) -> list[str]
words_to_tags(words: list[str]) -> list[str]
train_hmm(corpus_path: str, output_path: str) -> HMMSegmenter

class HMMSegmenter:
    def cut(self, sentence: str) -> list[str]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "HMMSegmenter": ...
```

`main.py` 支持以下调用方式：

```text
python main.py
python main.py train [语料路径]
python main.py segment-file 输入文件 [-o 输出文件]
python main.py segment-text "需要分词的文本"
```

不带子命令时进入交互菜单。

## 4. 数据预处理

### 4.1 文本读取

依次尝试 `utf-8`、`gbk`、`gb18030` 编码。全部失败时抛出可由 CLI 转换为用户提示的明确异常。

### 4.2 清洗规则

`clean_text(text)` 按以下顺序处理：

1. 删除 HTML 标签。
2. 将连续空白和换行规范为单个空格。
3. 删除 `#`、`@` 等无意义特殊符号。
4. 保留中文、英文、数字、空格及常用中英文标点。
5. 去除首尾空白。

### 4.3 分句规则

`split_sentences(text)` 按以下边界切分：

- 中文句终标点：`。！？；`
- 英文句终标点：`!?;`
- 英文句点 `.`，但前后均为数字时不切分，避免拆分 `3.14`。

切分后去除空字符串和纯空白字符串，保证每个输出句非空。句末标点作为切分边界，不保留在分词结果中。

空格在训练和推理时均视为强制词边界。每个句子按空白拆成非空片段后分别交给 HMM 处理，避免英文单词、数字与相邻中文因清洗后空格消失而被错误拼接。

## 5. HMM 训练

训练流程：

1. 读取并清洗训练语料。
2. 对清洗后的文本分句。
3. 使用 `jieba` 对每个训练句中的非空片段辅助分词；空白不作为 HMM 发射字符。
4. 将词序列转换为合法 BMES 序列：
   - 单字词：`S`
   - 多字词：`B + M * (n - 2) + E`
5. 统计初始状态、状态转移和字符发射次数。
6. 将次数归一化为概率并保存模型 JSON。

模型 JSON 结构：

```json
{
  "metadata": {
    "states": ["B", "M", "E", "S"],
    "smoothing": 1e-10,
    "corpus": "data/raw/train.txt"
  },
  "pi": {},
  "A": {},
  "B": {}
}
```

`pi`、`A`、`B` 保存归一化概率。无效转移和训练中未出现的字符保留零概率，由运行时平滑处理。

## 6. Viterbi 分词

对非空句子执行以下步骤：

1. 将句子转换为字符序列。
2. 初始化每个 BMES 状态的得分：

   ```text
   V[0][state] = log(max(pi[state], 1e-10))
                 + log(max(B[state][char], 1e-10))
   ```

3. 按合法 BMES 转移执行递推：

   ```text
   V[t][state] = max(V[t-1][prev]
                     + log(max(A[prev][state], 1e-10))
                     + log(max(B[state][char], 1e-10)))
   ```

4. 记录每个位置的最优前驱状态。
5. 从最终状态为 `E` 或 `S` 的候选中选择最高得分路径。
6. 回溯最优 BMES 序列。
7. 按 BMES 规则还原词序列。

合法转移为：

```text
B -> M | E
M -> M | E
E -> B | S
S -> B | S
```

未登录字符使用 `1e-10` 平滑；`S` 状态获得较小额外权重，优先避免生僻字形成任意长词。残余不完整状态序列在还原阶段做确定性兜底，保证不丢字符。

## 7. 交互与文件输出

### 7.1 默认交互菜单

1. 语句分词模式：
   - 循环接收用户输入，空行返回菜单。
   - 输入先清洗，再分句并逐句分词。
   - 每句按空格连接词并输出。
2. 篇章分词模式：
   - 提示输入文件路径。
   - 自动识别编码，清洗全文并分句。
   - 逐句分词，每句一行写入 UTF-8 输出文件。
   - 默认输出名为原文件名加 `_seg.txt`。
3. 重新训练模型：
   - 默认读取 `data/raw/train.txt`。
   - 将模型写入 `models/hmm_model.json`。
4. 退出。

### 7.2 启动行为

- 默认模型存在时直接加载。
- 默认模型不存在时自动使用默认语料训练。
- 默认语料也不存在时，提示用户提供路径。
- 训练和篇章处理显示简洁进度。

### 7.3 错误处理

- 文件不存在：提示路径并返回菜单。
- 解码失败：列出尝试过的编码。
- 模型损坏或字段缺失：提示模型加载失败并允许重新训练。
- 清洗后无有效内容：提示输入或文件中没有可分词文本。
- 输入中断或文件结尾：正常退出，不输出未处理异常堆栈。

## 8. 测试与验收

测试使用 Python 标准库 `unittest`，不要求额外测试框架。

覆盖范围：

- 清洗规则：HTML、空白、特殊符号、中英文和数字保留。
- 分句规则：中英文标点、连续标点、空句过滤、数字小数点保护和空白边界。
- BMES 转换：单字词、双字词和多字词。
- 模型统计：概率归一化、模型保存和加载。
- Viterbi：零初始概率、未知字符、合法路径和最终状态限制。
- 分词还原：BMES 输出与原始字符保持一致。
- 文件模式：输出文件名、每句一行、词间空格、UTF-8 编码。
- CLI：文本模式、文件模式和错误路径的基本行为。

端到端验收步骤：

1. 在项目根目录运行全部单元测试。
2. 使用内置训练语料生成 `models/hmm_model.json`。
3. 使用典型中文、英文和数字混合句验证分词输出。
4. 对一个 UTF-8 文本文件执行篇章分词，检查 `_seg.txt`。
5. 从干净进程启动 `main.py`，确认可自动加载模型并进入交互菜单。

## 9. 运行环境与依赖

- Python 3.10 或更高版本。
- 运行时主要使用标准库。
- `jieba` 仅用于训练阶段生成辅助 BMES 标签。
- `requirements.txt` 固定最低兼容版本 `jieba>=0.42.1`。

核心分词和 Viterbi 算法由本工程实现，不调用 `jieba` 的分词结果作为运行时输出。
