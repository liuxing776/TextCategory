#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块
负责数据加载、预处理、词汇表构建和数据集划分
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import jieba
import re
import pickle
import os
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def convert_numpy(obj):
    """递归将 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class TextDataset(Dataset):
    """文本数据集类"""

    def __init__(self, texts, labels, vocab_to_idx, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 文本转索引
        text_indices = self.text_to_indices(text)

        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def text_to_indices(self, text):
        """将文本转换为索引序列"""
        # 分词
        words = jieba.lcut(text)
        words = [w for w in words if len(w) >= 2 and not re.match(r'^[\d\W]+$', w)]

        # 转换为索引
        indices = []
        for word in words:
            idx = self.vocab_to_idx.get(word, self.vocab_to_idx.get('<UNK>', 1))
            indices.append(idx)

        # 截断或填充
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([0] * (self.max_length - len(indices)))  # 0为padding

        return indices


class DataProcessor:
    """数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.class_weights = None
        self.data_stats = {}

    def load_data(self):
        """加载原始数据"""
        print("加载数据...")

        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")

        data_list = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                if line.strip():
                    parts = line.strip().split('_!_')
                    if len(parts) >= 4:
                        data_list.append({
                            'category_code': parts[1],
                            'category_name': parts[2],
                            'title': parts[3]
                        })

        df = pd.DataFrame(data_list)

        # 数据清洗
        original_count = len(df)
        df = df.dropna(subset=['title'])
        df = df[df['title'].str.len() > 5]
        cleaned_count = len(df)

        print(f"数据加载完成：{cleaned_count:,} 条记录（清洗前：{original_count:,}）")

        # 快速测试模式
        if self.config.quick_test:
            df = df.sample(n=min(self.config.quick_test_samples, len(df)), random_state=42)
            print(f"快速测试模式：使用 {len(df):,} 条样本")

        # 数据统计
        self._analyze_data(df)

        return df

    def _analyze_data(self, df):
        """分析数据特征"""
        print("\n数据分析:")

        # 类别分布
        category_counts = df['category_name'].value_counts()
        print("类别分布:")
        for cat, count in category_counts.items():
            percentage = count / len(df) * 100
            print(f"  {cat}: {count:,} ({percentage:.1f}%)")

        # 文本长度统计
        lengths = df['title'].str.len()
        print(f"\n文本长度统计:")
        print(f"  平均长度: {lengths.mean():.1f}")
        print(f"  中位数长度: {lengths.median():.1f}")
        print(f"  最大长度: {lengths.max()}")
        print(f"  95%分位数: {lengths.quantile(0.95):.1f}")

        # 不平衡程度
        max_count = category_counts.max()
        min_count = category_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\n数据不平衡比例: {imbalance_ratio:.1f}:1")

        # 保存统计信息
        self.data_stats = {
            'total_samples': int(len(df)),
            'num_classes': int(df['category_name'].nunique()),
            'avg_length': float(lengths.mean()),
            'max_length': int(lengths.max()),
            'imbalance_ratio': float(imbalance_ratio),
            'category_distribution': convert_numpy(category_counts.to_dict())
        }

        # 确保目录存在并保存统计结果
        os.makedirs(self.config.result_save_dir, exist_ok=True)
        stats_path = os.path.join(self.config.result_save_dir, 'data_stats.json')

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(self.data_stats), f, indent=2, ensure_ascii=False)

        print(f"数据统计已保存: {stats_path}")

    def build_vocab(self, texts, vocab_size):
        """构建词汇表"""
        print(f"构建词汇表（大小：{vocab_size:,}）...")

        # 统计词频
        word_counter = Counter()
        for text in tqdm(texts, desc="统计词频"):
            words = jieba.lcut(text)
            words = [w for w in words if len(w) >= 2 and not re.match(r'^[\d\W]+$', w)]
            word_counter.update(words)

        # 构建词汇表
        self.vocab_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_vocab = {0: '<PAD>', 1: '<UNK>'}

        # 添加高频词
        for word, count in word_counter.most_common(vocab_size - 2):
            idx = len(self.vocab_to_idx)
            self.vocab_to_idx[word] = idx
            self.idx_to_vocab[idx] = word

        print(f"词汇表构建完成：{len(self.vocab_to_idx):,} 个词")

        # 词汇覆盖率分析
        total_words = sum(word_counter.values())
        covered_words = sum(count for word, count in word_counter.most_common(vocab_size - 2))
        coverage = covered_words / total_words * 100
        print(f"词汇覆盖率: {coverage:.1f}%")

        # 确保目录存在并保存词汇表
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        vocab_path = os.path.join(self.config.model_save_dir, 'vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump((self.vocab_to_idx, self.idx_to_vocab), f)

        print(f"词汇表已保存: {vocab_path}")
        return self.vocab_to_idx

    def calculate_class_weights(self, labels):
        """计算类别权重用于处理数据不平衡"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)

        # 计算每个类别的权重（反比于样本数量）
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total_samples / (num_classes * count)

        # 转换为tensor
        weight_list = [weights[i] for i in range(num_classes)]
        self.class_weights = torch.FloatTensor(weight_list)

        print("类别权重计算完成:")
        for i, weight in enumerate(weight_list):
            class_name = self.label_encoder.classes_[i]
            print(f"  {class_name}: 权重 {weight:.3f}")

        # 确保目录存在并保存类别权重
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        weights_path = os.path.join(self.config.model_save_dir, 'class_weights.pkl')
        with open(weights_path, 'wb') as f:
            pickle.dump(self.class_weights, f)

        print(f"类别权重已保存: {weights_path}")
        return self.class_weights

    def prepare_data(self, df=None):
        """准备训练数据"""
        if df is None:
            df = self.load_data()

        print("准备训练数据...")

        # 编码标签（字符串 -> 数字）
        labels = self.label_encoder.fit_transform(df['category_name'])
        self.config.num_classes = len(self.label_encoder.classes_)

        # 确保目录存在并保存标签编码器
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        label_encoder_path = os.path.join(self.config.model_save_dir, 'label_encoder.pkl')
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"标签编码器已保存: {label_encoder_path}")

        # 计算类别权重
        if self.config.use_class_weights:
            self.calculate_class_weights(labels)

        # 构建词汇表
        self.build_vocab(df['title'].tolist(), self.config.vocab_size)

        # 获取文本数据
        texts = df['title'].tolist()

        # 首先划分训练集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels,
            test_size=self.config.test_size,
            stratify=labels,
            random_state=self.config.random_state
        )

        # 再从训练数据中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.val_size,
            stratify=y_temp,
            random_state=self.config.random_state
        )

        print(f"数据划分完成：训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")

        # 包装成 Dataset（使用统一的 max_seq_len）
        max_length = getattr(self.config, 'max_seq_len', getattr(self.config, 'max_length', 33))

        train_dataset = TextDataset(X_train, y_train, self.vocab_to_idx, max_length)
        val_dataset = TextDataset(X_val, y_val, self.vocab_to_idx, max_length)
        test_dataset = TextDataset(X_test, y_test, self.vocab_to_idx, max_length)

        # 构造 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        print("数据准备完成!")

        return train_loader, val_loader, test_loader

    def load_vocab(self, vocab_path=None):
        """加载已保存的词汇表"""
        if vocab_path is None:
            vocab_path = os.path.join(self.config.model_save_dir, 'vocab.pkl')

        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                self.vocab_to_idx, self.idx_to_vocab = pickle.load(f)
            print(f"词汇表已加载: {vocab_path}")
            return True
        else:
            print(f"词汇表文件不存在: {vocab_path}")
            return False

    def load_label_encoder(self, encoder_path=None):
        """加载已保存的标签编码器"""
        if encoder_path is None:
            encoder_path = os.path.join(self.config.model_save_dir, 'label_encoder.pkl')

        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"标签编码器已加载: {encoder_path}")
            return True
        else:
            print(f"标签编码器文件不存在: {encoder_path}")
            return False

    def load_class_weights(self, weights_path=None):
        """加载已保存的类别权重"""
        if weights_path is None:
            weights_path = os.path.join(self.config.model_save_dir, 'class_weights.pkl')

        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                self.class_weights = pickle.load(f)
            print(f"类别权重已加载: {weights_path}")
            return True
        else:
            print(f"类别权重文件不存在: {weights_path}")
            return False

    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.vocab_to_idx) if self.vocab_to_idx else 0

    def get_class_names(self):
        """获取类别名称列表"""
        return self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else []


def test_data_processor():
    """测试数据处理器"""

    # 这里需要导入config，但为了避免循环导入，我们创建一个简单的配置类
    class SimpleConfig:
        def __init__(self):
            self.data_path = "D:\\g3\\ML\\TextCategory\\toutiao_cat_data.txt"
            self.output_dir = "test_output"
            self.model_save_dir = os.path.join(self.output_dir, "models")
            self.result_save_dir = os.path.join(self.output_dir, "results")
            self.vocab_size = 10000
            self.max_seq_len = 33
            self.batch_size = 32
            self.test_size = 0.2
            self.val_size = 0.1
            self.random_state = 42
            self.use_class_weights = True
            self.quick_test = True
            self.quick_test_samples = 1000
            self.num_workers = 0

    print("测试数据处理器...")

    config = SimpleConfig()

    try:
        # 创建数据处理器
        processor = DataProcessor(config)

        # 如果数据文件存在，进行测试
        if os.path.exists(config.data_path):
            train_loader, val_loader, test_loader = processor.prepare_data()

            # 测试加载一个批次
            for batch_texts, batch_labels in train_loader:
                print(f"批次形状: 文本 {batch_texts.shape}, 标签 {batch_labels.shape}")
                break

            print("数据处理器测试完成!")
            return True
        else:
            print(f"数据文件 {config.data_path} 不存在，跳过测试")
            return False

    except Exception as e:
        print(f"数据处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    test_data_processor()