#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块
统一管理所有实验参数，基于数据分析结果优化
"""

import torch
import os
import json
import numpy as np


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


class Config:
    """实验配置类"""

    def __init__(self):
        # 基础路径配置
        self.data_path = "D:\\g3\\ML\\TextCategory2\\toutiao_cat_data.txt"
        self.output_dir = "textcnn_experiments"
        self.model_save_dir = os.path.join(self.output_dir, "models")
        self.result_save_dir = os.path.join(self.output_dir, "results")
        self.plot_save_dir = os.path.join(self.output_dir, "plots")

        # 基于数据分析结果的核心参数
        self.vocab_size = 30000
        self.max_seq_len = 33  # 统一使用 max_seq_len
        self.max_length = 33   # 保持兼容性
        self.num_classes = 15

        # TextCNN模型参数
        self.embed_dim = 300
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128
        self.dropout = 0.5

        # 训练参数
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.weight_decay = 1e-4

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0

        # 数据划分
        self.test_size = 0.2
        self.val_size = 0.1
        self.random_state = 42

        # 实验控制
        self.save_model = True
        self.early_stopping_patience = 10
        self.save_predictions = True

        # 数据不平衡处理（基于122.1:1的不平衡比例）
        self.use_class_weights = True
        self.use_focal_loss = False
        self.focal_loss_alpha = 1.0
        self.focal_loss_gamma = 2.0

        # 可视化配置
        self.plot_style = 'seaborn'
        self.figure_size = (12, 8)
        self.font_family = 'SimHei'  # 中文字体

        # 实验设置
        self.ablation_epochs = 15  # 消融实验使用较少epoch节省时间
        self.quick_test = False  # 快速测试模式
        self.quick_test_samples = 10000  # 快速测试样本数

        # 创建必要的目录
        self._create_directories()

        print(f"配置初始化完成")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        print(f"核心参数: vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len}")

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.output_dir,
            self.model_save_dir,
            self.result_save_dir,
            self.plot_save_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_model_config(self):
        """获取模型配置字典"""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'filter_sizes': self.filter_sizes,
            'num_filters': self.num_filters,
            'dropout': self.dropout
        }

    def get_training_config(self):
        """获取训练配置字典"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience
        }

    def get_data_config(self):
        """获取数据配置字典"""
        return {
            'data_path': self.data_path,
            'max_seq_len': self.max_seq_len,
            'vocab_size': self.vocab_size,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state
        }

    def update_config(self, **kwargs):
        """动态更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # 保持max_length和max_seq_len同步
                if key == 'max_seq_len':
                    setattr(self, 'max_length', value)
                elif key == 'max_length':
                    setattr(self, 'max_seq_len', value)
                print(f"更新配置: {key} = {value}")
            else:
                print(f"警告: 配置项 {key} 不存在")

    def save_config(self, filepath=None):
        """保存配置到文件"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'config.json')

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # 处理不能JSON序列化的对象
                if isinstance(value, torch.device):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value

        # 使用convert_numpy确保所有数据都可序列化
        serializable_config = convert_numpy(config_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        print(f"配置已保存到: {filepath}")

    @classmethod
    def load_config(cls, filepath):
        """从文件加载配置"""
        config = cls()

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            if hasattr(config, key):
                if key == 'device':
                    setattr(config, key, torch.device(value))
                else:
                    setattr(config, key, value)

        print(f"配置已从 {filepath} 加载")
        return config


# 预定义的实验配置
class ExperimentConfigs:
    """预定义的实验配置"""

    @staticmethod
    def get_baseline_config():
        """基线实验配置"""
        config = Config()
        config.experiment_name = "baseline"
        return config

    @staticmethod
    def get_ablation_configs():
        """消融实验配置列表"""
        base_config = Config()
        configs = []

        # 实验1: 单一卷积核
        config1 = Config()
        config1.filter_sizes = [3]
        config1.experiment_name = "single_filter"
        config1.num_epochs = base_config.ablation_epochs
        configs.append(config1)

        # 实验2: 更多卷积核
        config2 = Config()
        config2.filter_sizes = [2, 3, 4, 5, 6]
        config2.experiment_name = "more_filters"
        config2.num_epochs = base_config.ablation_epochs
        configs.append(config2)

        # 实验3: 更少特征
        config3 = Config()
        config3.num_filters = 64
        config3.experiment_name = "fewer_features"
        config3.num_epochs = base_config.ablation_epochs
        configs.append(config3)

        # 实验4: 无Dropout
        config4 = Config()
        config4.dropout = 0.0
        config4.experiment_name = "no_dropout"
        config4.num_epochs = base_config.ablation_epochs
        configs.append(config4)

        # 实验5: 更大嵌入维度
        config5 = Config()
        config5.embed_dim = 512
        config5.experiment_name = "large_embedding"
        config5.num_epochs = base_config.ablation_epochs
        configs.append(config5)

        return configs

    @staticmethod
    def get_optimization_configs():
        """优化实验配置列表"""
        base_config = Config()
        configs = []

        # 不同学习率
        for lr in [0.0005, 0.001, 0.002, 0.005]:
            config = Config()
            config.learning_rate = lr
            config.experiment_name = f"lr_{lr}"
            config.num_epochs = base_config.ablation_epochs
            configs.append(config)

        # 不同批次大小
        for bs in [32, 64, 128]:
            config = Config()
            config.batch_size = bs
            config.experiment_name = f"bs_{bs}"
            config.num_epochs = base_config.ablation_epochs
            configs.append(config)

        return configs

    @staticmethod
    def get_quick_test_config():
        """快速测试配置"""
        config = Config()
        config.quick_test = True
        config.num_epochs = 5
        config.ablation_epochs = 3
        config.experiment_name = "quick_test"
        return config


# 全局配置实例
DEFAULT_CONFIG = Config()

if __name__ == "__main__":
    # 测试配置模块
    print("测试配置模块...")

    # 创建默认配置
    config = Config()

    # 保存配置
    config.save_config()

    # 测试实验配置
    ablation_configs = ExperimentConfigs.get_ablation_configs()
    print(f"\n消融实验配置数量: {len(ablation_configs)}")

    for conf in ablation_configs:
        print(f"实验: {conf.experiment_name}, 卷积核: {conf.filter_sizes}")

    # 测试快速配置
    quick_config = ExperimentConfigs.get_quick_test_config()
    print(f"\n快速测试配置: epochs={quick_config.num_epochs}, samples={quick_config.quick_test_samples}")

    print("配置模块测试完成!")