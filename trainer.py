#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块
负责模型训练、验证和保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import time
from tqdm import tqdm
import numpy as np
from models import FocalLoss


def convert_numpy(obj):
    """递归将 NumPy 类型转换为 Python 原生类型"""
    import numpy as np
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


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()


class Trainer:
    """训练器类"""

    def __init__(self, config, model, train_loader, val_loader, test_loader=None, class_weights=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 设置优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )

        # 设置损失函数
        if config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.focal_loss_alpha,
                gamma=config.focal_loss_gamma
            )
            print(f"使用Focal Loss: alpha={config.focal_loss_alpha}, gamma={config.focal_loss_gamma}")
        elif config.use_class_weights and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
            print("使用类别权重的交叉熵损失")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("使用标准交叉熵损失")

        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=0.001,
            restore_best_weights=True
        )

        # 最佳指标
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # 训练时间统计
        self.train_start_time = None
        self.epoch_times = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 进度条
        pbar = tqdm(self.train_loader, desc="训练")

        for batch_idx, (batch_texts, batch_labels) in enumerate(pbar):
            batch_texts = batch_texts.to(self.config.device)
            batch_labels = batch_labels.to(self.config.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_texts)
            loss = self.criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            # 更新进度条
            current_acc = correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return float(avg_loss), float(accuracy)

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_texts, batch_labels in tqdm(self.val_loader, desc="验证"):
                batch_texts = batch_texts.to(self.config.device)
                batch_labels = batch_labels.to(self.config.device)

                outputs = self.model(batch_texts)
                loss = self.criterion(outputs, batch_labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return float(avg_loss), float(accuracy)

    def train(self):
        """完整训练流程"""
        print(f"开始训练 {self.model.__class__.__name__} 模型...")
        print(f"设备: {self.config.device}")
        print(f"训练轮数: {self.config.num_epochs}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")

        # 记录训练开始时间
        self.train_start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史 - 确保所有数值都是Python原生类型
            self.train_history['train_loss'].append(float(train_loss))
            self.train_history['train_acc'].append(float(train_acc))
            self.train_history['val_loss'].append(float(val_loss))
            self.train_history['val_acc'].append(float(val_acc))
            self.train_history['learning_rates'].append(float(current_lr))

            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(float(epoch_time))

            # 打印进度
            print(f"Epoch {epoch + 1:2d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = float(val_acc)
                self.best_epoch = int(epoch + 1)

                if self.config.save_model:
                    self.save_model('best_model.pth')

            # 早停检查
            if self.early_stopping(val_acc, self.model):
                print(f"早停触发！在第 {epoch + 1} 轮停止训练")
                break

            # 定期保存检查点
            if (epoch + 1) % 10 == 0 and self.config.save_model:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')

        # 训练完成
        total_train_time = time.time() - self.train_start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0

        print(f"训练完成！")
        print(f"最佳验证准确率: {self.best_val_acc:.4f} (第 {self.best_epoch} 轮)")
        print(f"总训练时间: {total_train_time / 60:.1f} 分钟")
        print(f"平均每轮时间: {avg_epoch_time:.1f} 秒")

        # 保存训练历史
        self.save_training_history()

        return self.train_history

    def save_model(self, filename):
        """保存模型"""
        # 确保保存目录存在
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        model_path = os.path.join(self.config.model_save_dir, filename)

        # 保存模型状态和相关信息
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': convert_numpy(self.config.__dict__),
            'best_val_acc': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch),
            'train_history': convert_numpy(self.train_history)
        }

        torch.save(save_dict, model_path)

        if 'best' in filename:
            print(f"保存最佳模型: {model_path}")

    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join(self.config.model_save_dir, filename)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.config.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_acc = float(checkpoint.get('best_val_acc', 0))
            self.best_epoch = int(checkpoint.get('best_epoch', 0))
            self.train_history = checkpoint.get('train_history', {})

            print(f"成功加载模型: {model_path}")
            return True
        else:
            print(f"模型文件不存在: {model_path}")
            return False

    def save_training_history(self):
        """保存训练历史"""
        # 确保保存目录存在
        os.makedirs(self.config.result_save_dir, exist_ok=True)
        history_path = os.path.join(self.config.result_save_dir, 'train_history.json')

        # 添加训练统计信息
        training_stats = {
            'train_history': self.train_history,
            'best_val_acc': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch),
            'total_epochs': len(self.train_history['train_loss']),
            'avg_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
            'total_train_time': float(sum(self.epoch_times)) if self.epoch_times else 0.0
        }

        # 使用 convert_numpy 确保所有数据都是可序列化的
        serializable_stats = convert_numpy(training_stats)

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

        print(f"训练历史已保存: {history_path}")

    def get_learning_curve_data(self):
        """获取学习曲线数据"""
        return {
            'epochs': list(range(1, len(self.train_history['train_loss']) + 1)),
            'train_loss': self.train_history['train_loss'],
            'val_loss': self.train_history['val_loss'],
            'train_acc': self.train_history['train_acc'],
            'val_acc': self.train_history['val_acc'],
            'learning_rates': self.train_history['learning_rates']
        }

    def resume_training(self, checkpoint_path, additional_epochs=10):
        """从检查点恢复训练"""
        if self.load_model(checkpoint_path):
            print(f"从检查点恢复训练，将继续训练 {additional_epochs} 轮")

            # 保存当前配置的epoch数
            original_epochs = self.config.num_epochs

            # 设置新的epoch数
            current_epoch = len(self.train_history['train_loss'])
            self.config.num_epochs = current_epoch + additional_epochs

            # 继续训练
            self.train()

            # 恢复原始配置
            self.config.num_epochs = original_epochs
        else:
            print("无法加载检查点，开始新的训练")
            self.train()


class QuickTrainer(Trainer):
    """快速训练器 - 用于消融实验等快速验证"""

    def __init__(self, config, model, train_loader, val_loader, test_loader=None, class_weights=None):
        super().__init__(config, model, train_loader, val_loader, test_loader, class_weights)

        # 快速训练的特殊设置
        self.quick_mode = True

        # 减少输出频率
        self.print_freq = max(1, self.config.num_epochs // 5)

    def train(self):
        """快速训练流程 - 减少输出"""
        print(f"快速训练模式: {self.model.__class__.__name__}")

        self.train_start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # 训练和验证（无进度条）
            train_loss, train_acc = self._quick_train_epoch()
            val_loss, val_acc = self._quick_validate()

            # 记录历史 - 确保数据类型正确
            self.train_history['train_loss'].append(float(train_loss))
            self.train_history['train_acc'].append(float(train_acc))
            self.train_history['val_loss'].append(float(val_loss))
            self.train_history['val_acc'].append(float(val_acc))

            # 减少打印频率
            if (epoch + 1) % self.print_freq == 0 or epoch == self.config.num_epochs - 1:
                print(f"Epoch {epoch + 1:2d}/{self.config.num_epochs} | "
                      f"Val Acc: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = float(val_acc)
                self.best_epoch = int(epoch + 1)

            # 早停检查
            if self.early_stopping(val_acc, self.model):
                break

        total_time = time.time() - self.train_start_time
        print(f"快速训练完成！最佳准确率: {self.best_val_acc:.4f}, 用时: {total_time:.1f}s")

        return self.train_history

    def _quick_train_epoch(self):
        """快速训练一个epoch（无进度条）"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_texts, batch_labels in self.train_loader:
            batch_texts = batch_texts.to(self.config.device)
            batch_labels = batch_labels.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_texts)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        return float(total_loss / len(self.train_loader)), float(correct / total)

    def _quick_validate(self):
        """快速验证（无进度条）"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_texts, batch_labels in self.val_loader:
                batch_texts = batch_texts.to(self.config.device)
                batch_labels = batch_labels.to(self.config.device)

                outputs = self.model(batch_texts)
                loss = self.criterion(outputs, batch_labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        return float(total_loss / len(self.val_loader)), float(correct / total)


def test_trainer():
    """测试训练器"""
    from config import Config
    from data_processor import DataProcessor
    from models import create_model

    print("测试训练器...")

    # 创建配置（快速测试模式）
    config = Config()
    config.quick_test = True
    config.num_epochs = 3
    config.batch_size = 8

    try:
        # 创建数据处理器
        processor = DataProcessor(config)
        train_loader, val_loader, test_loader = processor.prepare_data()

        # 创建模型
        model = create_model(config, 'textcnn')

        # 创建训练器
        trainer = Trainer(config, model, train_loader, val_loader, test_loader, processor.class_weights)

        # 训练模型
        history = trainer.train()

        print("训练器测试完成!")
        print("best_val_acc 类型:", type(trainer.best_val_acc))
        print("train_history 类型:", type(trainer.train_history))
        if trainer.train_history['train_loss']:
            print("train_history['train_loss'] 类型:", type(trainer.train_history['train_loss'][0]))
        print(f"最佳验证准确率: {float(trainer.best_val_acc):.4f}")

        return True

    except Exception as e:
        print(f"训练器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    test_trainer()