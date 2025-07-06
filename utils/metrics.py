#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块
提供各种模型评估指标的计算函数
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    log_loss, matthews_corrcoef, cohen_kappa_score,
    hamming_loss, jaccard_score, zero_one_loss
)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')


def calculate_basic_metrics(y_true, y_pred):
    """
    计算基础分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        dict: 包含各种指标的字典
    """
    metrics = {}

    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return metrics


def calculate_advanced_metrics(y_true, y_pred, y_proba=None):
    """
    计算高级分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选）

    Returns:
        dict: 包含各种高级指标的字典
    """
    metrics = {}

    # MCC和Kappa
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # 错误率相关
    metrics['zero_one_loss'] = zero_one_loss(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)

    # 如果有预测概率，计算相关指标
    if y_proba is not None:
        try:
            # 对于多分类问题，计算多分类AUC
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                # 多分类AUC (one-vs-rest)
                unique_classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=unique_classes)

                if y_true_bin.shape[1] > 1:  # 确保是多分类
                    metrics['auc_macro'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                    metrics['auc_weighted'] = roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')
                else:
                    # 二分类
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2):
                # 二分类AUC
                if y_proba.ndim == 2:
                    y_proba_positive = y_proba[:, 1]
                else:
                    y_proba_positive = y_proba
                metrics['auc'] = roc_auc_score(y_true, y_proba_positive)

            # 对数损失
            metrics['log_loss'] = log_loss(y_true, y_proba)

        except Exception as e:
            print(f"Warning: Could not calculate probability-based metrics: {str(e)}")

    return metrics


def calculate_per_class_metrics(y_true, y_pred, class_names=None):
    """
    计算每个类别的详细指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        pd.DataFrame: 包含每个类别详细指标的DataFrame
    """
    # 生成分类报告
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    # 转换为DataFrame
    df_report = pd.DataFrame(report).transpose()

    # 只保留类别行
    if class_names:
        class_metrics = df_report.loc[class_names].copy()
    else:
        # 自动识别类别
        unique_classes = sorted(set(y_true))
        class_labels = [str(cls) for cls in unique_classes]
        class_metrics = df_report.loc[class_labels].copy()

    # 添加额外的每类别指标
    cm = confusion_matrix(y_true, y_pred)

    if class_names:
        for i, class_name in enumerate(class_names):
            if i < len(cm):
                # 真正例、假正例、假负例、真负例
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn

                # 特异性 (Specificity)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                class_metrics.loc[class_name, 'specificity'] = specificity

                # 假正率 (False Positive Rate)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                class_metrics.loc[class_name, 'false_positive_rate'] = fpr

                # 假负率 (False Negative Rate)
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                class_metrics.loc[class_name, 'false_negative_rate'] = fnr

    return class_metrics


def calculate_confusion_matrix_metrics(y_true, y_pred, class_names=None):
    """
    基于混淆矩阵计算详细指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        dict: 包含混淆矩阵和相关指标的字典
    """
    cm = confusion_matrix(y_true, y_pred)

    results = {
        'confusion_matrix': cm,
        'total_samples': len(y_true),
        'correct_predictions': np.diag(cm).sum(),
        'incorrect_predictions': cm.sum() - np.diag(cm).sum()
    }

    # 按类别分析混淆情况
    if class_names and len(class_names) == cm.shape[0]:
        class_confusion = {}

        for i, class_name in enumerate(class_names):
            tp = cm[i, i]  # 真正例
            fp = cm[:, i].sum() - tp  # 假正例
            fn = cm[i, :].sum() - tp  # 假负例
            tn = cm.sum() - tp - fp - fn  # 真负例

            class_confusion[class_name] = {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_negative': int(tn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            }

        results['per_class_confusion'] = class_confusion

    return results


def calculate_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """
    计算所有相关指标的综合函数

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选）
        class_names: 类别名称列表（可选）

    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {}

    # 基础指标
    basic_metrics = calculate_basic_metrics(y_true, y_pred)
    metrics.update(basic_metrics)

    # 高级指标
    advanced_metrics = calculate_advanced_metrics(y_true, y_pred, y_proba)
    metrics.update(advanced_metrics)

    # 混淆矩阵指标
    confusion_metrics = calculate_confusion_matrix_metrics(y_true, y_pred, class_names)
    metrics.update(confusion_metrics)

    return metrics


def evaluate_model_performance(model, X_test, y_test, class_names=None):
    """
    评估模型在测试集上的性能

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        class_names: 类别名称列表

    Returns:
        dict: 完整的性能评估结果
    """
    # 预测
    y_pred = model.predict(X_test)

    # 预测概率（如果支持）
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
        except:
            pass
    elif hasattr(model, 'decision_function'):
        try:
            y_proba = model.decision_function(X_test)
            # 对于SVM等，转换决策函数值为概率形式
            if y_proba.ndim == 1:
                # 二分类
                y_proba = np.column_stack([1 - y_proba, y_proba])
            else:
                # 多分类，使用softmax转换
                exp_scores = np.exp(y_proba - np.max(y_proba, axis=1, keepdims=True))
                y_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            pass

    # 计算指标
    metrics = calculate_metrics(y_test, y_pred, y_proba, class_names)

    # 添加预测结果
    metrics['predictions'] = y_pred
    metrics['probabilities'] = y_proba

    return metrics


def compare_models(models_results, metrics_to_compare=None):
    """
    比较多个模型的性能

    Args:
        models_results: 模型结果字典，格式为 {model_name: metrics_dict}
        metrics_to_compare: 要比较的指标列表

    Returns:
        pd.DataFrame: 模型比较结果
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

    comparison_data = []

    for model_name, results in models_results.items():
        row = {'Model': model_name}

        for metric in metrics_to_compare:
            if metric in results:
                row[metric] = results[metric]
            else:
                row[metric] = np.nan

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # 按主要指标排序
    if 'accuracy' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    elif 'f1_macro' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('f1_macro', ascending=False)

    return comparison_df


def calculate_confidence_intervals(scores, confidence=0.95):
    """
    计算性能指标的置信区间

    Args:
        scores: 分数列表（来自交叉验证等）
        confidence: 置信度

    Returns:
        dict: 包含均值、标准差和置信区间的字典
    """
    import scipy.stats as stats

    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array, ddof=1)

    # 计算置信区间
    alpha = 1 - confidence
    df = len(scores_array) - 1
    t_value = stats.t.ppf(1 - alpha / 2, df)
    margin_error = t_value * std_score / np.sqrt(len(scores_array))

    ci_lower = mean_score - margin_error
    ci_upper = mean_score + margin_error

    return {
        'mean': mean_score,
        'std': std_score,
        'confidence_interval': (ci_lower, ci_upper),
        'margin_of_error': margin_error
    }


def analyze_prediction_errors(y_true, y_pred, class_names=None):
    """
    分析预测错误模式

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        dict: 错误分析结果
    """
    # 找出错误预测的索引
    error_indices = np.where(y_true != y_pred)[0]

    # 错误类型统计
    error_types = {}
    for idx in error_indices:
        true_label = y_true[idx]
        pred_label = y_pred[idx]

        if class_names:
            error_key = f"{class_names[true_label]} -> {class_names[pred_label]}"
        else:
            error_key = f"{true_label} -> {pred_label}"

        error_types[error_key] = error_types.get(error_key, 0) + 1

    # 按错误频率排序
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

    results = {
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(y_true),
        'error_indices': error_indices.tolist(),
        'error_types': dict(sorted_errors),
        'most_common_errors': sorted_errors[:10]  # 前10种最常见错误
    }

    return results


def calculate_class_imbalance_metrics(y_true, class_names=None):
    """
    计算类别不平衡相关指标

    Args:
        y_true: 真实标签
        class_names: 类别名称列表

    Returns:
        dict: 类别不平衡分析结果
    """
    from collections import Counter

    # 类别分布
    class_counts = Counter(y_true)
    total_samples = len(y_true)

    # 类别频率
    class_frequencies = {cls: count / total_samples for cls, count in class_counts.items()}

    # 不平衡指标
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    results = {
        'total_samples': total_samples,
        'num_classes': len(class_counts),
        'class_counts': dict(class_counts),
        'class_frequencies': class_frequencies,
        'max_class_count': max_count,
        'min_class_count': min_count,
        'imbalance_ratio': max_count / min_count,
        'entropy': -sum(freq * np.log2(freq) for freq in class_frequencies.values()),
        'gini_index': 1 - sum(freq ** 2 for freq in class_frequencies.values())
    }

    # 如果有类别名称，添加具体类别信息
    if class_names:
        results['class_names'] = class_names
        results['most_frequent_class'] = class_names[max(class_counts, key=class_counts.get)]
        results['least_frequent_class'] = class_names[min(class_counts, key=class_counts.get)]

    return results


def generate_classification_report_dict(y_true, y_pred, class_names=None):
    """
    生成完整的分类报告字典

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        dict: 完整的分类报告
    """
    # 基础分类报告
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    # 添加额外信息
    report['total_samples'] = len(y_true)
    report['total_correct'] = np.sum(y_true == y_pred)
    report['total_incorrect'] = np.sum(y_true != y_pred)
    report['error_rate'] = report['total_incorrect'] / report['total_samples']

    # 类别不平衡信息
    imbalance_info = calculate_class_imbalance_metrics(y_true, class_names)
    report['class_imbalance'] = imbalance_info

    # 错误分析
    error_analysis = analyze_prediction_errors(y_true, y_pred, class_names)
    report['error_analysis'] = error_analysis

    return report


# 便捷函数：打印格式化的指标报告
def print_metrics_report(metrics, title="Model Performance Report"):
    """
    打印格式化的指标报告

    Args:
        metrics: 指标字典
        title: 报告标题
    """
    print("=" * 60)
    print(title.center(60))
    print("=" * 60)

    # 基础性能指标
    print("\nBasic Performance Metrics:")
    print("-" * 30)
    basic_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']

    for metric in basic_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title():<20}: {metrics[metric]:.4f}")

    # 高级指标
    print("\nAdvanced Metrics:")
    print("-" * 30)
    advanced_metrics = ['matthews_corrcoef', 'cohen_kappa', 'auc_macro', 'log_loss']

    for metric in advanced_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title():<20}: {metrics[metric]:.4f}")

    # 样本统计
    if 'total_samples' in metrics:
        print(f"\nSample Statistics:")
        print("-" * 30)
        print(f"{'Total Samples':<20}: {metrics['total_samples']}")
        if 'correct_predictions' in metrics:
            print(f"{'Correct Predictions':<20}: {metrics['correct_predictions']}")
            print(f"{'Incorrect Predictions':<20}: {metrics['incorrect_predictions']}")

    print("=" * 60)