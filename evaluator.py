#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估器模块
负责模型评估、指标计算和结果分析
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def convert_numpy(obj):
    """递归将 NumPy 类型转换为 Python 原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, config, model, test_loader, label_encoder):
        self.config = config
        self.model = model.to(config.device)
        self.test_loader = test_loader
        self.label_encoder = label_encoder
        self.class_names = label_encoder.classes_
        self.num_classes = len(self.class_names)

        # 评估结果存储
        self.predictions = []
        self.true_labels = []
        self.prediction_probs = []
        self.evaluation_results = {}

    def load_best_model(self, model_path=None):
        """加载最佳模型"""
        if model_path is None:
            model_path = os.path.join(self.config.model_save_dir, 'best_model.pth')

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {model_path}")
            return True
        else:
            print(f"模型文件不存在: {model_path}")
            return False

    def predict(self, return_probs=True):
        """对测试集进行预测"""
        print("开始预测...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_texts, batch_labels in tqdm(self.test_loader, desc="预测中"):
                batch_texts = batch_texts.to(self.config.device)
                batch_labels = batch_labels.to(self.config.device)

                # 前向传播
                outputs = self.model(batch_texts)

                # 获取预测结果
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                # 收集结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                if return_probs:
                    all_probs.extend(probs.cpu().numpy())

        self.predictions = np.array(all_predictions)
        self.true_labels = np.array(all_labels)
        if return_probs:
            self.prediction_probs = np.array(all_probs)

        print(f"预测完成！共处理 {len(self.predictions)} 个样本")
        return self.predictions, self.true_labels, self.prediction_probs

    def calculate_metrics(self):
        """计算各种评估指标"""
        print("计算评估指标...")

        if len(self.predictions) == 0:
            self.predict()

        # 基础分类指标
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision_macro = precision_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        recall_macro = recall_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        f1_macro = f1_score(self.true_labels, self.predictions, average='macro', zero_division=0)

        precision_weighted = precision_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)

        # 每个类别的详细指标
        precision_per_class = precision_score(self.true_labels, self.predictions, average=None, zero_division=0)
        recall_per_class = recall_score(self.true_labels, self.predictions, average=None, zero_division=0)
        f1_per_class = f1_score(self.true_labels, self.predictions, average=None, zero_division=0)

        # AUC指标（如果有概率预测）
        auc_macro = None
        auc_weighted = None
        if len(self.prediction_probs) > 0:
            try:
                auc_macro = roc_auc_score(
                    self.true_labels, self.prediction_probs,
                    multi_class='ovr', average='macro'
                )
                auc_weighted = roc_auc_score(
                    self.true_labels, self.prediction_probs,
                    multi_class='ovr', average='weighted'
                )
            except ValueError as e:
                print(f"AUC计算失败: {e}")

        # 存储结果
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc_macro': auc_macro,
            'auc_weighted': auc_weighted,
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
                'class_names': self.class_names.tolist()
            }
        }

        return self.evaluation_results

    def generate_classification_report(self):
        """生成详细的分类报告"""
        if len(self.predictions) == 0:
            self.predict()

        # sklearn的分类报告
        report_dict = classification_report(
            self.true_labels, self.predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        # 转换为DataFrame便于展示
        report_df = pd.DataFrame(report_dict).transpose()

        # 保存报告
        report_path = os.path.join(self.config.result_save_dir, 'classification_report.csv')
        report_df.to_csv(report_path, encoding='utf-8-sig')

        # 打印报告
        print("\n分类报告:")
        print("=" * 80)
        print(classification_report(
            self.true_labels, self.predictions,
            target_names=self.class_names,
            zero_division=0
        ))

        return report_dict

    def analyze_confusion_matrix(self):
        """分析混淆矩阵"""
        if len(self.predictions) == 0:
            self.predict()

        # 计算混淆矩阵
        cm = confusion_matrix(self.true_labels, self.predictions)

        # 计算百分比矩阵
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 分析最容易混淆的类别对
        confusion_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': cm_percent[i, j]
                    })

        # 按混淆数量排序
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

        print("\n最容易混淆的类别对（Top 10）:")
        print("=" * 60)
        for i, pair in enumerate(confusion_pairs[:10], 1):
            print(f"{i:2d}. {pair['true_class']} -> {pair['predicted_class']}: "
                  f"{pair['count']} 次 ({pair['percentage']:.1f}%)")

        # 保存混淆矩阵数据
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_percent': cm_percent.tolist(),
            'class_names': self.class_names.tolist(),
            'confusion_pairs': confusion_pairs
        }

        cm_path = os.path.join(self.config.result_save_dir, 'confusion_matrix.json')
        with open(cm_path, 'w', encoding='utf-8') as f:
            # json.dump(cm_data, f, indent=2, ensure_ascii=False)
            json.dump(convert_numpy(cm_data), f, indent=2, ensure_ascii=False)

        return cm, cm_percent, confusion_pairs

    def analyze_per_class_performance(self):
        """分析各类别性能"""
        if 'per_class_metrics' not in self.evaluation_results:
            self.calculate_metrics()

        per_class = self.evaluation_results['per_class_metrics']

        # 创建类别性能DataFrame
        performance_df = pd.DataFrame({
            'class_name': per_class['class_names'],
            'precision': per_class['precision'],
            'recall': per_class['recall'],
            'f1_score': per_class['f1']
        })

        # 添加支持度（每个类别的样本数）
        class_support = np.bincount(self.true_labels)
        performance_df['support'] = class_support

        # 按F1分数排序
        performance_df = performance_df.sort_values('f1_score', ascending=False)

        print("\n各类别性能排序（按F1分数）:")
        print("=" * 80)
        for _, row in performance_df.iterrows():
            print(f"{row['class_name']:<12}: "
                  f"P={row['precision']:.3f} "
                  f"R={row['recall']:.3f} "
                  f"F1={row['f1_score']:.3f} "
                  f"Support={row['support']:>5}")

        # 识别表现最好和最差的类别
        best_class = performance_df.iloc[0]
        worst_class = performance_df.iloc[-1]

        print(f"\n表现最好的类别: {best_class['class_name']} (F1={best_class['f1_score']:.3f})")
        print(f"表现最差的类别: {worst_class['class_name']} (F1={worst_class['f1_score']:.3f})")

        # 保存类别性能分析
        performance_path = os.path.join(self.config.result_save_dir, 'per_class_performance.csv')
        performance_df.to_csv(performance_path, index=False, encoding='utf-8-sig')

        return performance_df

    def analyze_errors(self, num_samples=50):
        """错误案例分析"""
        if len(self.predictions) == 0:
            self.predict()

        # 找出所有错误预测
        error_mask = self.true_labels != self.predictions
        error_indices = np.where(error_mask)[0]

        print(f"\n错误分析:")
        print(f"总错误数: {len(error_indices)} / {len(self.predictions)}")
        print(f"错误率: {len(error_indices) / len(self.predictions) * 100:.2f}%")

        if len(error_indices) == 0:
            print("没有预测错误！")
            return []

        # 分析错误类型
        error_analysis = []
        for idx in error_indices[:num_samples]:  # 只分析前num_samples个错误
            true_class = self.class_names[self.true_labels[idx]]
            pred_class = self.class_names[self.predictions[idx]]

            confidence = None
            if len(self.prediction_probs) > 0:
                confidence = self.prediction_probs[idx].max()

            error_analysis.append({
                'sample_index': int(idx),
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': float(confidence) if confidence is not None else None
            })

        # 保存错误分析
        error_path = os.path.join(self.config.result_save_dir, 'error_analysis.json')
        with open(error_path, 'w', encoding='utf-8') as f:
            # json.dump(error_analysis, f, indent=2, ensure_ascii=False)
            json.dump(convert_numpy(error_analysis), f, indent=2, ensure_ascii=False)

        return error_analysis

    def save_predictions(self):
        """保存详细的预测结果"""
        if len(self.predictions) == 0:
            self.predict()

        # 创建预测结果DataFrame
        results_data = {
            'sample_index': range(len(self.predictions)),
            'true_label': self.true_labels,
            'predicted_label': self.predictions,
            'true_class': [self.class_names[i] for i in self.true_labels],
            'predicted_class': [self.class_names[i] for i in self.predictions],
            'is_correct': self.true_labels == self.predictions
        }

        # 添加预测置信度
        if len(self.prediction_probs) > 0:
            results_data['max_confidence'] = self.prediction_probs.max(axis=1)
            results_data['predicted_confidence'] = [
                self.prediction_probs[i, self.predictions[i]]
                for i in range(len(self.predictions))
            ]

            # 添加各类别的概率
            for i, class_name in enumerate(self.class_names):
                results_data[f'prob_{class_name}'] = self.prediction_probs[:, i]

        # 创建DataFrame并保存
        results_df = pd.DataFrame(results_data)

        predictions_path = os.path.join(self.config.result_save_dir, 'detailed_predictions.csv')
        results_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')

        print(f"详细预测结果已保存: {predictions_path}")

        return results_df

    def comprehensive_evaluation(self):
        """综合评估 - 执行所有评估步骤"""
        print("开始综合评估...")
        print("=" * 60)

        # 1. 加载最佳模型
        self.load_best_model()

        # 2. 进行预测
        self.predict()

        # 3. 计算基础指标
        metrics = self.calculate_metrics()

        # 4. 生成分类报告
        classification_report = self.generate_classification_report()

        # 5. 分析混淆矩阵
        cm, cm_percent, confusion_pairs = self.analyze_confusion_matrix()

        # 6. 分析各类别性能
        performance_df = self.analyze_per_class_performance()

        # 7. 错误案例分析
        error_analysis = self.analyze_errors()

        # 8. 保存预测结果
        predictions_df = self.save_predictions()

        # 9. 生成评估总结
        evaluation_summary = self._generate_evaluation_summary(metrics, performance_df)

        print("=" * 60)
        print("综合评估完成！")
        print(f"总体准确率: {metrics['accuracy']:.4f}")
        print(f"宏平均F1: {metrics['f1_macro']:.4f}")
        print(f"加权平均F1: {metrics['f1_weighted']:.4f}")

        return {
            'metrics': metrics,
            'classification_report': classification_report,
            'confusion_matrix': cm,
            'performance_df': performance_df,
            'error_analysis': error_analysis,
            'predictions_df': predictions_df,
            'evaluation_summary': evaluation_summary
        }

    def _generate_evaluation_summary(self, metrics, performance_df):
        """生成评估总结"""
        summary = {
            'model_performance': {
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro']
            },
            'best_performing_classes': performance_df.head(3)[['class_name', 'f1_score']].to_dict('records'),
            'worst_performing_classes': performance_df.tail(3)[['class_name', 'f1_score']].to_dict('records'),
            'total_samples': len(self.predictions),
            'total_errors': np.sum(self.true_labels != self.predictions),
            'error_rate': np.mean(self.true_labels != self.predictions)
        }

        # 保存评估总结
        summary_path = os.path.join(self.config.result_save_dir, 'evaluation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            # json.dump(summary, f, indent=2, ensure_ascii=False)
            json.dump(convert_numpy(summary), f, indent=2, ensure_ascii=False)

        return summary


class ComparisonEvaluator:
    """模型对比评估器"""

    def __init__(self, config, results_dir=None):
        self.config = config
        self.results_dir = results_dir or config.result_save_dir
        self.comparison_results = {}

    def add_model_results(self, model_name, metrics, predictions=None):
        """添加模型结果"""
        self.comparison_results[model_name] = {
            'metrics': metrics,
            'predictions': predictions
        }

    def load_results_from_files(self, model_results_mapping):
        """从文件加载多个模型的结果

        Args:
            model_results_mapping: dict, {model_name: results_file_path}
        """
        for model_name, file_path in model_results_mapping.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                self.comparison_results[model_name] = results
                print(f"加载模型结果: {model_name}")
            except Exception as e:
                print(f"加载模型结果失败 {model_name}: {e}")

    def generate_comparison_table(self):
        """生成模型对比表"""
        if not self.comparison_results:
            print("没有模型结果可对比")
            return None

        # 提取对比指标
        comparison_data = []
        for model_name, results in self.comparison_results.items():
            metrics = results.get('metrics', {})
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'F1_Macro': metrics.get('f1_macro', 0),
                'F1_Weighted': metrics.get('f1_weighted', 0),
                'Precision_Macro': metrics.get('precision_macro', 0),
                'Recall_Macro': metrics.get('recall_macro', 0)
            }
            comparison_data.append(row)

        # 创建对比表
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)

        # 按F1_Macro排序
        comparison_df = comparison_df.sort_values('F1_Macro', ascending=False)

        print("\n模型性能对比:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))

        # 保存对比表
        comparison_path = os.path.join(self.results_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')

        return comparison_df

    def analyze_model_differences(self, primary_model, comparison_models):
        """分析模型间的差异"""
        if primary_model not in self.comparison_results:
            print(f"主模型 {primary_model} 结果不存在")
            return

        primary_metrics = self.comparison_results[primary_model]['metrics']

        print(f"\n以 {primary_model} 为基准的性能差异:")
        print("=" * 60)

        for model_name in comparison_models:
            if model_name not in self.comparison_results:
                continue

            model_metrics = self.comparison_results[model_name]['metrics']

            acc_diff = model_metrics.get('accuracy', 0) - primary_metrics.get('accuracy', 0)
            f1_diff = model_metrics.get('f1_macro', 0) - primary_metrics.get('f1_macro', 0)

            print(f"{model_name}:")
            print(f"  准确率差异: {acc_diff:+.4f}")
            print(f"  F1宏平均差异: {f1_diff:+.4f}")


def create_visualization_plots(config, evaluator, save_plots=True):
    """创建评估可视化图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 混淆矩阵热图
    def plot_confusion_matrix():
        cm, _, _ = evaluator.analyze_confusion_matrix()

        plt.figure(figsize=(12, 10))
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=evaluator.class_names,
                    yticklabels=evaluator.class_names,
                    cbar_kws={'label': '百分比 (%)'})

        plt.title('混淆矩阵热图', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(config.plot_save_dir, 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        plt.show()

    # 2. 各类别性能对比图
    def plot_class_performance():
        performance_df = evaluator.analyze_per_class_performance()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # F1分数对比
        ax1.barh(performance_df['class_name'], performance_df['f1_score'])
        ax1.set_xlabel('F1分数')
        ax1.set_title('各类别F1分数对比')
        ax1.grid(True, alpha=0.3)

        # 精确率vs召回率散点图
        ax2.scatter(performance_df['precision'], performance_df['recall'],
                    s=100, alpha=0.7)

        for i, class_name in enumerate(performance_df['class_name']):
            ax2.annotate(class_name,
                         (performance_df['precision'].iloc[i],
                          performance_df['recall'].iloc[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)

        ax2.set_xlabel('精确率')
        ax2.set_ylabel('召回率')
        ax2.set_title('精确率 vs 召回率')
        ax2.grid(True, alpha=0.3)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(config.plot_save_dir, 'class_performance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        plt.show()

    # 执行绘图
    plot_confusion_matrix()
    plot_class_performance()


def test_evaluator():
    """测试评估器"""
    from config import Config
    from data_processor import DataProcessor
    from models import create_model
    from trainer import Trainer

    print("测试评估器...")

    # 创建配置（快速测试模式）
    config = Config()
    config.quick_test = True
    config.num_epochs = 2

    try:
        # 创建数据和模型
        processor = DataProcessor(config)
        train_loader, val_loader, test_loader = processor.prepare_data()

        model = create_model(config, 'textcnn')

        # 快速训练
        trainer = Trainer(config, model, train_loader, val_loader, test_loader)
        trainer.train()

        # 评估模型
        evaluator = ModelEvaluator(config, model, test_loader, processor.label_encoder)
        results = evaluator.comprehensive_evaluation()

        print("评估器测试完成!")
        print(f"测试准确率: {results['metrics']['accuracy']:.4f}")

        return True

    except Exception as e:
        print(f"评估器测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    test_evaluator()