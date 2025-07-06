#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
提供各种绘图和可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix',
                          figsize=(10, 8), save_path=None, normalize=False):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
        normalize: 是否归一化
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if class_names and len(class_names) > 5:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return cm


def plot_class_distribution(labels, class_names=None, title='Class Distribution',
                            figsize=(12, 6), save_path=None):
    """
    绘制类别分布图

    Args:
        labels: 标签列表
        class_names: 类别名称列表
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    # 计算类别分布
    class_counts = Counter(labels)

    if class_names:
        classes = class_names
        counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    else:
        classes = sorted(class_counts.keys())
        counts = [class_counts[cls] for cls in classes]

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 条形图
    bars = ax1.bar(range(len(classes)), counts, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} - Bar Chart')
    ax1.set_xticks(range(len(classes)))

    if class_names:
        ax1.set_xticklabels(classes, rotation=45, ha='right')
    else:
        ax1.set_xticklabels(classes)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                 f'{count}', ha='center', va='bottom')

    # 饼图
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{title} - Pie Chart')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")

    plt.show()

    # 打印统计信息
    total = sum(counts)
    print(f"\nClass Distribution Statistics:")
    print(f"Total samples: {total}")
    print(f"Number of classes: {len(classes)}")
    print(f"Most frequent class: {classes[np.argmax(counts)]} ({max(counts)} samples)")
    print(f"Least frequent class: {classes[np.argmin(counts)]} ({min(counts)} samples)")
    print(f"Class imbalance ratio: {max(counts) / min(counts):.2f}")

    return class_counts


def plot_metrics_comparison(results_df, metrics=['accuracy', 'f1_macro'],
                            title='Model Performance Comparison', figsize=(12, 6), save_path=None):
    """
    绘制模型性能对比图

    Args:
        results_df: 包含模型结果的DataFrame
        metrics: 要比较的指标列表
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    n_metrics = len(metrics)

    if n_metrics == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric in results_df.columns:
            # 排序
            sorted_df = results_df.sort_values(metric, ascending=False)

            # 绘制条形图
            bars = ax.bar(range(len(sorted_df)), sorted_df[metric], alpha=0.7)
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} Comparison')
            ax.set_xticks(range(len(sorted_df)))

            # 设置x轴标签
            if 'model_name' in sorted_df.columns:
                labels = sorted_df['model_name']
            elif 'Model' in sorted_df.columns:
                labels = sorted_df['Model']
            else:
                labels = sorted_df.index

            ax.set_xticklabels(labels, rotation=45, ha='right')

            # 添加数值标签
            for bar, value in zip(bars, sorted_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'Metric "{metric}" not found',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{metric.title()} - Not Available')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")

    plt.show()


def plot_training_history(history, metrics=['accuracy', 'loss'], title='Training History',
                          figsize=(12, 4), save_path=None):
    """
    绘制训练历史图

    Args:
        history: 训练历史字典或DataFrame
        metrics: 要绘制的指标
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    if isinstance(history, dict):
        history_df = pd.DataFrame(history)
    else:
        history_df = history

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 训练集指标
        if metric in history_df.columns:
            ax.plot(history_df[metric], label=f'Train {metric}', linewidth=2)

        # 验证集指标
        val_metric = f'val_{metric}'
        if val_metric in history_df.columns:
            ax.plot(history_df[val_metric], label=f'Validation {metric}', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} History')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_roc_curves(y_true, y_proba, class_names, title='ROC Curves',
                    figsize=(10, 8), save_path=None):
    """
    绘制多分类ROC曲线

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        class_names: 类别名称
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    n_classes = len(class_names)

    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=figsize)

    # 计算每个类别的ROC曲线
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    # 计算微平均ROC曲线
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    plt.plot(fpr_micro, tpr_micro, linewidth=2, linestyle='--',
             label=f'Micro-average (AUC = {roc_auc_micro:.2f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves plot saved to {save_path}")

    plt.show()


def plot_feature_importance(feature_names, importance_scores, title='Feature Importance',
                            n_features=20, figsize=(10, 8), save_path=None):
    """
    绘制特征重要性图

    Args:
        feature_names: 特征名称列表
        importance_scores: 重要性分数
        title: 图表标题
        n_features: 显示的特征数量
        figsize: 图像大小
        save_path: 保存路径
    """
    # 排序并选择top n特征
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(n_features)

    plt.figure(figsize=figsize)

    # 水平条形图
    y_pos = np.arange(len(importance_df))
    plt.barh(y_pos, importance_df['importance'], alpha=0.7, color='lightcoral')

    plt.yticks(y_pos, importance_df['feature'])
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()  # 最重要的特征在顶部
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_learning_curve(train_sizes, train_scores, val_scores, title='Learning Curve',
                        figsize=(10, 6), save_path=None):
    """
    绘制学习曲线

    Args:
        train_sizes: 训练集大小
        train_scores: 训练分数
        val_scores: 验证分数
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)

    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 绘制学习曲线
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='blue')

    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plot saved to {save_path}")

    plt.show()


def plot_prediction_distribution(y_proba, y_true, class_names, title='Prediction Distribution',
                                 figsize=(12, 8), save_path=None):
    """
    绘制预测概率分布图

    Args:
        y_proba: 预测概率
        y_true: 真实标签
        class_names: 类别名称
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_classes):
        ax = axes[i]

        # 该类别的正例和负例的预测概率分布
        positive_proba = y_proba[y_true == i, i]
        negative_proba = y_proba[y_true != i, i]

        ax.hist(positive_proba, bins=30, alpha=0.7, label='True Positive', color='green')
        ax.hist(negative_proba, bins=30, alpha=0.7, label='True Negative', color='red')

        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{class_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution plot saved to {save_path}")

    plt.show()


def plot_error_analysis(error_counts, error_types, title='Error Analysis',
                        figsize=(12, 6), save_path=None):
    """
    绘制错误分析图

    Args:
        error_counts: 错误计数
        error_types: 错误类型
        title: 图表标题
        figsize: 图像大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)

    # 排序
    sorted_indices = np.argsort(error_counts)[::-1]
    sorted_counts = [error_counts[i] for i in sorted_indices]
    sorted_types = [error_types[i] for i in sorted_indices]

    # 只显示前20个
    if len(sorted_counts) > 20:
        sorted_counts = sorted_counts[:20]
        sorted_types = sorted_types[:20]

    # 水平条形图
    y_pos = np.arange(len(sorted_types))
    bars = plt.barh(y_pos, sorted_counts, alpha=0.7, color='orange')

    plt.yticks(y_pos, sorted_types)
    plt.xlabel('Error Count')
    plt.title(title)
    plt.gca().invert_yaxis()

    # 添加数值标签
    for bar, count in zip(bars, sorted_counts):
        width = bar.get_width()
        plt.text(width + max(sorted_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{count}', ha='left', va='center')

    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to {save_path}")

    plt.show()


def create_subplots_grid(n_plots, max_cols=3, figsize_per_plot=(5, 4)):
    """
    创建子图网格

    Args:
        n_plots: 子图数量
        max_cols: 最大列数
        figsize_per_plot: 每个子图的大小

    Returns:
        fig, axes: 图形和轴对象
    """
    cols = min(n_plots, max_cols)
    rows = (n_plots - 1) // cols + 1

    figsize = (cols * figsize_per_plot[0], rows * figsize_per_plot[1])
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # 确保axes始终是数组
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    return fig, axes


def save_all_plots(save_dir, prefix="plot", format="png", dpi=300):
    """
    保存所有当前打开的图形

    Args:
        save_dir: 保存目录
        prefix: 文件名前缀
        format: 文件格式
        dpi: 分辨率
    """
    import os
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 获取所有图形
    figs = [plt.figure(i) for i in plt.get_fignums()]

    for i, fig in enumerate(figs):
        filename = f"{prefix}_{i + 1}_{timestamp}.{format}"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved: {filepath}")


# 设置默认样式
def setup_plotting_style():
    """设置绘图样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


# 初始化时设置样式
setup_plotting_style()