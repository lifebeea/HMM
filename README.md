# HMM 中文分词与 Web 工作台

## 功能

- 使用 BMES 状态和 Viterbi 算法实现中文分词。
- 支持语句分词、篇章分词和结果文件输出。
- 提供本地上传页面，可在线预览并下载 `_seg.txt`。

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 训练

默认语料位于 `data/raw/train.txt`：

```bash
python3 main.py train
```

模型输出到 `models/hmm_model.json`。

## 使用

启动交互模式：

```bash
python3 main.py
```

启动 Web 页面：

```bash
python3 main.py web
```

浏览器访问 `http://127.0.0.1:5000`，上传 `.txt` 文件后查看预览并下载结果。

## 测试

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests -v
```
