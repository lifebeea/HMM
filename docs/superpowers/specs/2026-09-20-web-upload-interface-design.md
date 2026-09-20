# HMM 分词的 Web 上传页面设计方案

## 1. 目标

在既有 HMM 分词工程上增加一个面向本机或局域网单用户的教学演示页面。用户可以：

- 选择或拖放文本文件。
- 在浏览器中在线预览分词结果。
- 下载按原文件名生成的 `_seg.txt` 结果文件。
- 查看文件名、句子数和词数。
- 在失败时看到明确错误并重新提交。

本设计依赖 [HMM 分词工程设计方案](./2026-09-20-hmm-chinese-word-segmentation-design.md)，不修改 Viterbi 和 HMM 训练的核心职责。

## 2. 技术路线

采用 Flask、Jinja 和原生 CSS/JavaScript：

- Python 后端负责文件接收、文本处理、分词和结果下载。
- Jinja 提供单页面模板。
- 原生 JavaScript 使用 `fetch` 调用 JSON 接口。
- 不引入 Node.js、前端框架或构建工具。
- Web 服务为本地单用户模式，不提供账号、公网存储或模型训练接口。

## 3. 工程结构

在原设计上增加：

```text
项目二/
├── web/
│   ├── __init__.py
│   ├── app.py
│   ├── service.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── styles.css
│       └── app.js
├── instance/
│   └── results/
└── tests/
    └── test_web.py
```

模块职责：

- `web/app.py`：创建 Flask 应用、注册页面和 API 路由。
- `web/service.py`：上传文件校验、分词编排、预览生成和结果文件管理。
- `web/templates/index.html`：工作台结构和状态占位。
- `web/static/styles.css`：桌面与移动端布局。
- `web/static/app.js`：文件选择、拖放、上传请求、状态切换和下载。
- `tests/test_web.py`：Flask 测试客户端、文件处理和安全边界测试。

启动方式增加：

```text
python main.py web
```

默认监听 `127.0.0.1:5000`。禁用调试重载器，确保 HMM 模型只加载一次。

## 4. 页面结构

页面使用结果优先的平衡式布局：

1. 顶部任务栏始终可见，包含拖放/点击上传区和“开始分析”按钮。
2. 中部统计区显示文件名、句子数、词数；未分析时显示 `--`。
3. 下部结果区展示每句一行、词间空格的分词预览。
4. 结果区包含下载按钮；没有结果时按钮禁用。

视觉与响应式要求：

- 使用安静、工具化的界面，不使用营销式首屏或装饰性大标题。
- 使用中性色背景、白色工作区和少量蓝绿主色；错误使用独立红色状态。
- 使用本地内联的 Lucide SVG 表示上传、文件、下载和错误，并提供可访问名称；页面不依赖 CDN。
- 桌面端为纵向工作台，移动端转为单列。
- 文件和按钮文本必须截断或换行，不能遮挡或溢出。
- 不伪造百分比进度；后端同步处理期间使用不确定进度条。

## 5. 页面状态

页面使用明确的状态机：

1. `idle`：结果区显示空状态，下载按钮禁用。
2. `ready`：显示已选文件名和大小，“开始分析”启用。
3. `processing`：显示不确定进度条，上传区和按钮禁用，防止重复提交。
4. `complete`：更新统计和预览，下载按钮启用，提示结果已生成。
5. `error`：显示具体错误，保留已选文件以便重试。

新上传会清除上一任务的结果和错误。状态区域使用 `aria-live`，键盘可选择文件和触发下载，并支持 `prefers-reduced-motion`。

## 6. API 契约

### 6.1 页面

```text
GET /
```

返回工作台页面。

### 6.2 上传与分词

```text
POST /api/segment
Content-Type: multipart/form-data
字段名: file
```

成功响应：

```json
{
  "job_id": "UUID",
  "input_name": "demo.txt",
  "output_name": "demo_seg.txt",
  "sentence_count": 126,
  "word_count": 832,
  "preview": [
    "自然 语言 处理 是 人工 智能 的 重要 方向。",
    "HMM 通过 Viterbi 算法 寻找 最优 状态 序列。"
  ],
  "download_url": "/api/results/UUID/download"
}
```

处理流程：

1. 校验文件名、扩展名、大小和非空内容。
2. 自动尝试 `UTF-8`、`GBK`、`GB18030` 编码。
3. 调用既有 `clean_text` 和 `split_sentences`。
4. 逐句调用 HMM 分词。
5. 将结果写入 `instance/results/<UUID>.txt`。
6. 生成统计、预览和下载地址。

### 6.3 下载

```text
GET /api/results/<job_id>/download
```

响应 UTF-8 文本文件，下载名恢复为 `原文件名_seg.txt`。

### 6.4 错误响应

所有 API 错误使用统一结构：

```json
{
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "仅支持 .txt 文本文件"
  }
}
```

错误代码包括：

- `EMPTY_FILE`
- `INVALID_FILE_TYPE`
- `FILE_TOO_LARGE`
- `DECODE_ERROR`
- `EMPTY_TEXT`
- `MODEL_UNAVAILABLE`
- `RESULT_NOT_FOUND`
- `RESULT_EXPIRED`
- `INTERNAL_ERROR`

格式非法的结果 ID 返回 `RESULT_NOT_FOUND`；格式合法但对应文件已被过期清理时返回 `RESULT_EXPIRED`。模型启动加载失败时，`main.py web` 直接给出明确提示并终止；测试或异常运行时若模型对象不可用，接口返回 `MODEL_UNAVAILABLE`。

## 7. 安全与资源限制

- 仅接受扩展名为 `.txt` 的文件，文件最大 `5 MiB`。
- 输入文件名只用于安全展示和下载名，磁盘结果使用 UUID。
- UUID 必须经过格式校验，避免路径穿越。
- 结果只允许通过对应 UUID 下载，不提供目录列表。
- 结果存放在 `instance/results/`。
- 启动时和每次上传时清理超过 1 小时的结果文件。
- 预览最多输出前 20 句或 2000 个字符。
- `MAX_CONTENT_LENGTH` 设置为 `5 MiB`，并在服务层重复校验。
- 服务默认只绑定 `127.0.0.1`，不默认暴露到公网。

## 8. 测试与验收

自动化测试覆盖：

- 文件扩展名、空文件、超大文件和编码错误。
- 合法上传后返回正确的文件名、句子数、词数和预览。
- 下载内容与上传文本分词结果一致。
- UUID 不存在、结果过期和非法下载请求。
- `_seg.txt` 下载文件名正确，统计数量与输出句子一致。
- 预览截断在 20 句或 2000 字符以内。
- 页面路由返回工作台，模型不可用时返回统一错误结构。

端到端验收：

1. 运行全部 Python 单元测试。
2. 启动本地 Web 服务并访问 `127.0.0.1:5000`。
3. 上传中文、英文、数字混合的 `.txt` 文件。
4. 检查在线预览、文件名、句子数和词数。
5. 下载 `_seg.txt`，确认 UTF-8 编码、每句一行、词间空格。
6. 在桌面和约 390px 宽的移动视口检查布局、按钮状态和文本不溢出。
7. 验证错误文件不会生成结果，正常文件会在过期清理前保持可下载。

## 9. 依赖与兼容性

- Flask `>=3.0,<4`。
- 继续使用 Python 3.10 或更高版本。
- 继续使用现有 HMM 模型和 `jieba` 训练依赖。
- Web 层不得在运行时调用 `jieba` 代替自研 HMM 分词。
