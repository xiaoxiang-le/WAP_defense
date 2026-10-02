# 项目概述

本项目实现了一个面向 Web 请求的智能入侵检测原型系统，用于识别正常请求、跨站脚本攻击（XSS）和 SQL 注入攻击。系统通过对 URL 或请求 Payload 进行解码、分词和向量化，再使用传统机器学习及深度学习模型完成分类，并输出模型评估结果。

项目包含 AiWaf-1 和 AiWaf-2 两个版本：AiWaf-1 展示从网络流量采集到桌面端告警的完整检测流程；AiWaf-2 侧重多种分类模型的训练、评估和预测对比。

> 当前项目主要用于机器学习 WAF 的研究、实验和演示。代码能够检测并分类攻击请求，但尚未实现反向代理、请求阻断、规则下发和日志持久化等生产级 WAF 功能。

## 系统主要功能

- 对 URL 编码内容进行解码、统一大小写、数字泛化和特征分词。
- 识别正常请求、XSS 攻击和 SQL 注入攻击。
- 支持 RF、KNN、SVM、CNN 和 GRU 五种模型的训练与预测。
- 自动划分训练集、验证集和测试集，并保存训练后的模型。
- 计算准确率、精确率、召回率和 F1 分数，并生成混淆矩阵。
- AiWaf-1 可采集 HTTP 流量，并在 Tkinter 桌面界面中显示请求信息和检测结果。

## 系统处理流程

```text
URL / HTTP 请求 Payload
          |
          v
URL 解码、规范化与分词
          |
          v
字符 TF-IDF / 词序列与 Embedding
          |
          v
机器学习或深度学习模型
          |
          v
正常请求 / XSS 攻击 / SQL 注入攻击
          |
          v
预测结果、评估指标与可视化图表
```

## AiWaf-1：流量采集与桌面端检测

AiWaf-1 使用 Scapy 监听 HTTP 请求，提取源/目标 IPv4 或 IPv6 地址、MAC 地址、请求方法、Host、Path、User-Agent 和受限长度的请求体等信息。请求 Payload 经字符 n-gram TF-IDF 表示后，由逻辑回归模型判断为正常或恶意请求；随后结合关键字规则标注 XSS 或 SQL 注入类型及风险等级，并在 Tkinter 界面中显示结果。抓包任务在后台线程中运行，并通过 TCP 会话重组降低请求跨包时的漏检概率。

网卡、端口和请求体读取上限可以通过 `--interface`、`--port` 与 `--max-body-bytes` 参数指定；不传网卡参数时使用系统默认接口。由于其分析对象是明文 HTTP 流量，因此不能直接解析 HTTPS 加密内容。

## AiWaf-2：多模型攻击分类

AiWaf-2 是项目的主要训练和实验模块。它从 XSS 与 SQL 注入数据集中读取正常及恶意样本，先按模型实际使用的规范化结果去重并移除标签冲突，再使用固定随机种子按 6:2:2 分层划分训练集、验证集和测试集，避免等价 Payload 跨集合泄漏。传统模型使用字符 n-gram TF-IDF 特征，CNN 和 GRU 使用训练集拟合的词表、定长序列与可训练 Embedding 层。系统分别训练和比较以下五种分类模型：

- **GRU (门控循环单元)**
- **CNN (卷积神经网络)**
- **KNN (K-最近邻)**
- **SVM (支持向量机)**
- **RF (随机森林)**

训练时，新模型与特征管线先写入临时目录，只有全部选定模型训练成功后才发布到 `AiWaf-2/model`，因此中断训练不会提前覆盖可用产物。混淆矩阵保存在 `AiWaf-2/images`，完整指标写入 `training_metrics.json`。`model_manifest.json` 记录数据指纹、标签、特征参数及各模型 SHA-256，用于在预测和部分重训前校验产物是否配套。`predict.py` 会加载指定模型，对单条 Payload 分别给出预测结果；使用多个模型时还会输出多数投票的 `ENSEMBLE` 结果。

### 检测流程

1. **数据加载**：加载预定义的数据集，包含 XSS、SQL 注入以及良性样本。
2. **数据预处理**：包括 URL 解码和转换为小写处理。
3. **向量化**：传统模型使用字符 TF-IDF；神经网络使用词序列、padding 和 Embedding。
4. **模型训练**：在三类数据上训练模型。
5. **模型预测**：利用训练好的模型进行预测。
6. **模型评估**：评估模型性能，包括生成混淆矩阵图，以可视化不同类别的识别效果。

### 快速开始

项目建议使用 Conda 创建独立的 Python 3.8 环境。以下配置已在 Windows 环境中验证。

#### Conda 环境配置

在项目根目录执行：

```powershell
conda env create -f AiWaf-2\environment.yml
conda activate aiwaf-py38
```

如果环境已经存在，可执行 `conda env update -f AiWaf-2\environment.yml --prune` 更新依赖。

#### venv 环境配置

项目依赖 TensorFlow 2.13，使用 venv 时需要 Python 3.8。请在项目根目录使用 Python 3.8 解释器执行：

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

执行 `python -c "import sys; print(sys.executable)"` 可以确认当前解释器是否来自项目的 `.venv`。

### 运行项目

先进入代码目录：

```powershell
cd AiWaf-2\code
```

训练并评估全部模型：

```powershell
python trainmain.py --models all --epochs 3 --seed 42
```

首次训练或修改数据、随机种子、`--max-features`、`--max-vocab`、`--sequence-length` 后必须使用 `--models all` 全量训练。产物校验通过时，可以使用 `--models rf svm` 只更新指定模型；已有其他模型和指标不会被覆盖。仅选择 RF、KNN 或 SVM 时不会加载 TensorFlow，可减少传统模型命令的启动时间和内存占用。

使用已有模型进行预测：

```powershell
python predict.py "http://example.com/?id=1 union select password from users"
```

也可以使用 `--models svm cnn` 只调用指定模型。

批量预测时，输入文件每行放置一个 Payload。模型只加载一次，可使用 `--json` 输出便于其他程序处理的结果：

```powershell
python predict.py --input-file payloads.txt --models rf svm --json
```

### 当前评估结果

以下结果来自随机种子为 42、完成规范化去重后的固定测试集，共 7304 条样本：

| 模型 | Accuracy | Macro F1 |
| --- | ---: | ---: |
| RF | 99.60% | 99.60% |
| KNN | 99.58% | 99.56% |
| SVM | 99.66% | 99.66% |
| CNN | 99.60% | 99.60% |
| GRU | 99.45% | 99.45% |

### AiWaf-1 运行与抓包

AiWaf-1 使用 Scapy 获取网络流量。在 Windows 上进行实时抓包前，需要另外安装 [Npcap](https://npcap.com/dist/)，并建议以管理员权限运行终端。Npcap 属于系统驱动，无法仅通过 Python 依赖完成安装。

```powershell
cd AiWaf-1\code

# 重新训练 AiWaf-1 模型
python main.py --retrain

# 使用默认网卡监听 80 端口
python main.py

# 指定网卡和端口
python main.py --interface "Ethernet" --port 8080

# 限制每个 HTTP 请求体最多读取 32768 字节
python main.py --port 8080 --max-body-bytes 32768

# 将检测结果追加写入 JSONL；请求体默认不会写入日志
python main.py --port 8080 --log-file ..\logs\detections.jsonl

# 显式记录请求体（请求体可能包含密码、令牌等敏感信息）
python main.py --port 8080 --log-file ..\logs\detections.jsonl --log-body
```

### 运行测试

在项目根目录执行：

```powershell
python -m unittest discover -s tests -v
```

安装 Npcap 后，可以运行端到端集成测试。该测试会在本机启动临时 HTTP 服务，发送一条 XSS 请求，并验证抓包、URL 解析、模型分类和风险判断的完整链路：

```powershell
python tests\integration_capture.py
```

默认使用 Npcap 回环接口和 `18081` 端口，也可以通过 `--interface`、`--port` 和 `--timeout` 参数覆盖。
