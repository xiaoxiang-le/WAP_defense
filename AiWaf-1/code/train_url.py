"""AiWaf-1 的 URL 特征处理、模型训练、模型加载和预测模块。"""

import os
import pickle
import re
from pathlib import Path
from urllib.parse import unquote

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# 数据文件和模型文件均使用相对于项目目录的绝对路径，避免受启动目录影响。
BASE_DIR = Path(__file__).resolve().parent.parent
GOOD_URL_PATH = BASE_DIR / "data" / "good_fromE.txt"
BAD_URL_PATH = BASE_DIR / "data" / "badqueries.txt"
MODEL_PATH = BASE_DIR / "model" / "lg.pickle"
# 特征处理方式或模型结构发生不兼容变化时，应递增该版本号。
MODEL_VERSION = 2


def get_url():
    """读取正常和恶意 URL 数据，过滤数据文件中的空行。"""
    with GOOD_URL_PATH.open(encoding="utf-8", errors="ignore") as handle:
        good_urls = [line.strip() for line in handle if line.strip()]
    with BAD_URL_PATH.open(encoding="utf-8", errors="ignore") as handle:
        bad_urls = [line.strip() for line in handle if line.strip()]
    return good_urls, bad_urls


def normalize_url(url):
    """规范化 URL，减少相同攻击模式因写法不同而产生的特征差异。"""
    # 连续解码两次，用于还原常见的双重 URL 编码内容，并统一为小写。
    url = unquote(unquote(str(url))).lower()
    # 将不同的具体数字统一为 0，例如 id=123 和 id=456 会具有相似特征。
    url = re.sub(r"\d+", "0", url)
    # 将完整 HTTP(S) 地址归一化，避免模型过度记忆具体域名和路径。
    return re.sub(r"(http|https)://[a-z0-9\.@&/#!#\?]+", "http://u", url)


def split_word(url):
    """从 URL 中提取函数、标签、参数名等用于攻击类型判断的标记。"""
    url = normalize_url(url)
    # 正则重点保留 SQL 函数、HTML 标签、引号内容和键值参数等结构。
    pattern = r"""(?x)[\w\.]+?\(|\)|"\w+?"|'\w+?'|http://\w|</\w+>|<\w+>|<\w+|\w+=|>|[\w\.]+"""
    return [
        token
        for token in nltk.regexp_tokenize(url, pattern)
        if token not in {"0", "http://u", ""}
    ]


class Train:
    """封装 TF-IDF 特征提取器和逻辑回归分类器。"""

    def __init__(self):
        # 训练完成后，这两个属性会随整个 Train 对象一起保存到模型文件。
        self.model_version = MODEL_VERSION
        self.vectorizer = None
        self.classifier = None

    def model_train(self, seed=42):
        """训练并保存模型，返回模型在测试集上的准确率。"""
        good_urls, bad_urls = get_url()
        # 标签约定：0 表示正常 URL，1 表示恶意 URL。
        payloads, labels = _deduplicate_samples(
            good_urls + bad_urls,
            [0] * len(good_urls) + [1] * len(bad_urls),
        )
        train_data, test_data, train_labels, test_labels = train_test_split(
            payloads,
            labels,
            test_size=0.2,
            # 固定随机种子，使数据划分和训练结果可以复现。
            random_state=seed,
            # 让训练集和测试集保持相近的正常/恶意样本比例。
            stratify=labels,
        )

        # 使用字符级 TF-IDF 表示 URL，适合识别 SQL、XSS 等局部字符串模式。
        # URL 中的局部字符组合经常重复，TF-IDF 可突出具有区分度的模式。
        self.vectorizer = TfidfVectorizer(
            # 直接分析字符，不依赖自然语言中的单词边界。
            analyzer="char",
            # 提取长度为 2 到 5 的连续字符组合。
            ngram_range=(2, 5),
            # 最多保留 20000 个特征，控制模型规模和内存占用。
            max_features=20000,
            # 忽略只在一个样本中出现的低频特征，减少噪声。
            min_df=2,
            # 使用 1 + log(tf) 缩放词频，降低高频字符组合的影响。
            sublinear_tf=True,
            # 所有样本在提取特征前使用相同的 URL 规范化规则。
            preprocessor=normalize_url,
        )
        # 只使用训练集拟合词表和 IDF，防止测试集信息泄漏。
        train_vectors = self.vectorizer.fit_transform(train_data)
        test_vectors = self.vectorizer.transform(test_data)

        self.classifier = LogisticRegression(
            # liblinear 适用于当前中小规模、稀疏特征的二分类任务。
            solver="liblinear",
            # 根据各类别样本数量自动调整权重，缓解类别不平衡。
            class_weight="balanced",
            # 增加最大迭代次数，降低高维特征下模型未收敛的概率。
            max_iter=1000,
            # 固定优化过程中的随机行为，使结果更容易复现。
            random_state=seed,
        )
        self.classifier.fit(train_vectors, train_labels)
        accuracy = self.classifier.score(test_vectors, test_labels)
        print("AiWaf-1 模型准确率：{:.2%}".format(accuracy))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件，再原子替换正式模型，避免中途中断损坏已有模型。
        temporary_path = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(self, handle)
        os.replace(str(temporary_path), str(MODEL_PATH))
        return accuracy

    @classmethod
    def load_or_train(cls):
        """加载有效模型；模型不存在、损坏或版本不兼容时自动重训。"""
        try:
            with MODEL_PATH.open("rb") as handle:
                model = pickle.load(handle)
            if (
                # 同时检查版本、特征提取器和分类器，避免加载不完整模型。
                getattr(model, "model_version", None) != MODEL_VERSION
                or model.vectorizer is None
                or model.classifier is None
            ):
                raise ValueError("旧模型格式不兼容")
            return model
        except (FileNotFoundError, AttributeError, EOFError, pickle.UnpicklingError, ValueError):
            model = cls()
            model.model_train()
            return model

    def predict_labels(self, urls):
        """批量转换 URL，并返回分类器预测的 0/1 标签。"""
        vectors = self.vectorizer.transform([str(url) for url in urls])
        return self.classifier.predict(vectors)

    def predict_details(self, urls, threshold=0.5):
        """按指定恶意概率阈值返回标签、概率和中文结论。"""
        if not 0 <= threshold <= 1:
            raise ValueError("恶意判定阈值必须在 0 到 1 之间")
        urls = [str(url) for url in urls]
        if not urls:
            return []

        vectors = self.vectorizer.transform(urls)
        probabilities = self.classifier.predict_proba(vectors)
        # 从 classes_ 查找恶意类别，避免假设概率矩阵的列顺序固定。
        malicious_index = list(self.classifier.classes_).index(1)
        details = []
        for row in probabilities:
            probability = float(row[malicious_index])
            label = int(probability >= threshold)
            details.append(
                {
                    "label": label,
                    "malicious_probability": probability,
                    "message": "url为正常请求" if label == 0 else "url为恶意攻击",
                }
            )
        return details

    def predict(self, urls, threshold=0.5):
        """预测第一个 URL，并返回便于界面显示的中文分类结果。"""
        details = self.predict_details(urls, threshold=threshold)
        if not details:
            raise ValueError("至少需要一个待检测 URL")
        return details[0]["message"]


def _deduplicate_samples(payloads, labels):
    """按规范化 URL 去重，并删除标签互相冲突的样本。"""
    samples = {}
    conflicts = set()
    for payload, label in zip(payloads, labels):
        key = normalize_url(payload).strip()
        if not key:
            continue
        previous = samples.get(key)
        # 同一规范化 URL 同时标为正常和恶意时，无法提供可靠监督信号。
        if previous is not None and previous[1] != label:
            conflicts.add(key)
        else:
            samples[key] = (payload, label)
    for key in conflicts:
        samples.pop(key, None)
    ordered = [samples[key] for key in sorted(samples)]
    return [item[0] for item in ordered], [item[1] for item in ordered]

