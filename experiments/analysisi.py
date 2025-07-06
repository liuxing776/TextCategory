#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果分析实验 - 深入分析模型性能和错误模式
包括错误分析、特征重要性分析、类别混淆分析等
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from datetime import datetime
import json
import re
from collections import Counter, defaultdict

from config import Config
from data_processor import DataProcessor
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_class_distribution


class AnalysisExperiment:
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.trained_model = None
        self.vectorizer = None
        self.class_names = None

    def load_data_and_train_best_model(self):
        """加载数据并训练最佳模型"""
        print("Loading data and training best model...")
        self.data_processor.load_data()
        self.data_processor.preprocess()

        self.train_texts, self.train_labels = self.data_processor.get_train_data()
        self.val_texts, self.val_labels = self.data_processor.get_val_data()
        self.test_texts, self.test_labels = self.data_processor.get_test_data()

        # 获取类别名称
        self.class_names = self.data_processor.get_class_names()

        # 训练最佳模型（基于之前实验的最佳配置）
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        X_train = self.vectorizer.fit_transform(self.train_texts)
        X_val = self.vectorizer.transform(self.val_texts)
        X_test = self.vectorizer.transform(self.test_texts)

        self.trained_model = LogisticRegression(
            C=1,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        )

        self.trained_model.fit(X_train, self.train_labels)

        # 获取预测结果
        self.train_pred = self.trained_model.predict(X_train)
        self.val_pred = self.trained_model.predict(X_val)
        self.test_pred = self.trained_model.predict(X_test)

        # 获取预测概率
        self.train_proba = self.trained_model.predict_proba(X_train)
        self.val_proba = self.trained_model.predict_proba(X_val)
        self.test_proba = self.trained_model.predict_proba(X_test)

        print("Model training completed!")

    def analyze_overall_performance(self):
        """分析整体性能"""
        print("\n" + "=" * 50)
        print("Overall Performance Analysis")
        print("=" * 50)

        datasets = {
            'Train': (self.train_labels, self.train_pred, self.train_proba),
            'Validation': (self.val_labels, self.val_pred, self.val_proba),
            'Test': (self.test_labels, self.test_pred, self.test_proba)
        }

        performance_summary = []

        for dataset_name, (true_labels, pred_labels, pred_proba) in datasets.items():
            metrics = calculate_metrics(true_labels, pred_labels, pred_proba)

            performance_summary.append({
                'Dataset': dataset_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1 (Macro)': metrics['f1_macro'],
                'F1 (Weighted)': metrics['f1_weighted']
            })

            print(f"\n{dataset_name} Performance:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['f1_macro']:.4f}")
            print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")

        return pd.DataFrame(performance_summary)

    def analyze_class_performance(self):
        """分析各类别性能"""
        print("\n" + "=" * 50)
        print("Per-Class Performance Analysis")
        print("=" * 50)

        # 详细的分类报告
        report = classification_report(
            self.test_labels,
            self.test_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # 转换为DataFrame
        df_report = pd.DataFrame(report).transpose()

        # 只保留类别行（排除macro avg, weighted avg等）
        class_report = df_report[df_report.index.isin(self.class_names)].copy()

        # 添加支持度百分比
        total_samples = len(self.test_labels)
        class_report['support_pct'] = class_report['support'] / total_samples * 100

        # 排序
        class_report = class_report.sort_values('f1-score', ascending=False)

        print("\nPer-Class Performance (sorted by F1-score):")
        print(class_report[['precision', 'recall', 'f1-score', 'support', 'support_pct']].round(4))

        # 分析表现最好和最差的类别
        best_classes = class_report.head(3)
        worst_classes = class_report.tail(3)

        print(f"\nTop 3 performing classes:")
        for idx, row in best_classes.iterrows():
            print(f"  {idx}: F1={row['f1-score']:.4f}, Support={int(row['support'])}")

        print(f"\nWorst 3 performing classes:")
        for idx, row in worst_classes.iterrows():
            print(f"  {idx}: F1={row['f1-score']:.4f}, Support={int(row['support'])}")

        return class_report

    def analyze_confusion_patterns(self):
        """分析混淆模式"""
        print("\n" + "=" * 50)
        print("Confusion Pattern Analysis")
        print("=" * 50)

        # 计算混淆矩阵
        cm = confusion_matrix(self.test_labels, self.test_pred)

        # 转换为DataFrame便于分析
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)

        # 分析最容易混淆的类别对
        confusion_pairs = []

        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append({
                        'True Class': self.class_names[i],
                        'Predicted Class': self.class_names[j],
                        'Count': cm[i][j],
                        'True Class Total': cm[i].sum(),
                        'Error Rate': cm[i][j] / cm[i].sum()
                    })

        # 排序找出最大的混淆
        confusion_df = pd.DataFrame(confusion_pairs)
        if not confusion_df.empty:
            confusion_df = confusion_df.sort_values('Count', ascending=False)

            print("\nTop 10 confusion pairs:")
            print(confusion_df[['True Class', 'Predicted Class', 'Count', 'Error Rate']].head(10))

            # 分析最容易被误分类的类别
            error_by_class = confusion_df.groupby('True Class').agg({
                'Count': 'sum',
                'Error Rate': 'mean'
            }).sort_values('Count', ascending=False)

            print("\nClasses with most misclassifications:")
            print(error_by_class.head())

        return cm_df, confusion_df if not confusion_df.empty else pd.DataFrame()

    def analyze_prediction_confidence(self):
        """分析预测置信度"""
        print("\n" + "=" * 50)
        print("Prediction Confidence Analysis")
        print("=" * 50)

        # 计算最大预测概率（置信度）
        max_proba = np.max(self.test_proba, axis=1)
        predicted_classes = np.argmax(self.test_proba, axis=1)

        # 分析正确和错误预测的置信度分布
        correct_mask = (predicted_classes == self.test_labels)

        correct_confidence = max_proba[correct_mask]
        incorrect_confidence = max_proba[~correct_mask]

        print(
            f"Correct predictions confidence - Mean: {correct_confidence.mean():.4f}, Std: {correct_confidence.std():.4f}")
        print(
            f"Incorrect predictions confidence - Mean: {incorrect_confidence.mean():.4f}, Std: {incorrect_confidence.std():.4f}")

        # 分析低置信度预测
        low_confidence_threshold = 0.5
        low_confidence_mask = max_proba < low_confidence_threshold

        print(f"\nLow confidence predictions (< {low_confidence_threshold}): {low_confidence_mask.sum()}")
        print(f"Accuracy for low confidence predictions: {correct_mask[low_confidence_mask].mean():.4f}")

        # 分析高置信度预测
        high_confidence_threshold = 0.9
        high_confidence_mask = max_proba > high_confidence_threshold

        print(f"High confidence predictions (> {high_confidence_threshold}): {high_confidence_mask.sum()}")
        print(f"Accuracy for high confidence predictions: {correct_mask[high_confidence_mask].mean():.4f}")

        confidence_analysis = {
            'total_predictions': len(self.test_labels),
            'correct_predictions': correct_mask.sum(),
            'overall_accuracy': correct_mask.mean(),
            'mean_confidence_correct': correct_confidence.mean(),
            'mean_confidence_incorrect': incorrect_confidence.mean(),
            'low_confidence_count': low_confidence_mask.sum(),
            'low_confidence_accuracy': correct_mask[low_confidence_mask].mean() if low_confidence_mask.sum() > 0 else 0,
            'high_confidence_count': high_confidence_mask.sum(),
            'high_confidence_accuracy': correct_mask[
                high_confidence_mask].mean() if high_confidence_mask.sum() > 0 else 0
        }

        return confidence_analysis

    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n" + "=" * 50)
        print("Feature Importance Analysis")
        print("=" * 50)

        # 获取特征名称
        feature_names = self.vectorizer.get_feature_names_out()

        # 分析每个类别的重要特征（基于系数）
        class_features = {}

        for i, class_name in enumerate(self.class_names):
            # 获取该类别的系数
            coef = self.trained_model.coef_[i]

            # 获取最重要的正负特征
            top_positive_idx = np.argsort(coef)[-20:][::-1]
            top_negative_idx = np.argsort(coef)[:20]

            class_features[class_name] = {
                'positive_features': [(feature_names[idx], coef[idx]) for idx in top_positive_idx],
                'negative_features': [(feature_names[idx], coef[idx]) for idx in top_negative_idx]
            }

            print(f"\n{class_name} - Top positive features:")
            for feature, weight in class_features[class_name]['positive_features'][:10]:
                print(f"  {feature}: {weight:.4f}")

        # 使用置换重要性分析
        print("\nCalculating permutation importance...")
        X_test = self.vectorizer.transform(self.test_texts)

        # 只对少量特征计算置换重要性（计算量大）
        perm_importance = permutation_importance(
            self.trained_model, X_test, self.test_labels,
            n_repeats=5, random_state=42, n_jobs=-1
        )

        # 获取最重要的特征
        top_important_idx = np.argsort(perm_importance.importances_mean)[-20:][::-1]

        print("\nTop features by permutation importance:")
        for idx in top_important_idx[:10]:
            feature = feature_names[idx]
            importance = perm_importance.importances_mean[idx]
            print(f"  {feature}: {importance:.4f}")

        return class_features, perm_importance

    def analyze_error_cases(self, n_examples=5):
        """分析错误案例"""
        print("\n" + "=" * 50)
        print("Error Case Analysis")
        print("=" * 50)

        # 找出错误预测
        incorrect_mask = self.test_pred != self.test_labels
        incorrect_indices = np.where(incorrect_mask)[0]

        print(f"Total incorrect predictions: {len(incorrect_indices)}")

        # 分析每种错误类型的案例
        error_analysis = defaultdict(list)

        for idx in incorrect_indices:
            true_class = self.class_names[self.test_labels[idx]]
            pred_class = self.class_names[self.test_pred[idx]]
            confidence = np.max(self.test_proba[idx])
            text = self.test_texts[idx]

            error_key = f"{true_class} -> {pred_class}"
            error_analysis[error_key].append({
                'index': idx,
                'text': text,
                'confidence': confidence,
                'true_class': true_class,
                'pred_class': pred_class
            })

        # 显示每种错误的案例
        print(f"\nShowing up to {n_examples} examples for each error type:")

        for error_type, cases in sorted(error_analysis.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"\n{error_type} ({len(cases)} cases):")

            # 按置信度排序，显示高置信度的错误案例
            cases_sorted = sorted(cases, key=lambda x: x['confidence'], reverse=True)

            for i, case in enumerate(cases_sorted[:n_examples]):
                print(f"  Example {i + 1} (confidence: {case['confidence']:.3f}):")
                print(f"    Text: {case['text'][:100]}...")
                print()

        return error_analysis

    def analyze_text_characteristics(self):
        """分析文本特征"""
        print("\n" + "=" * 50)
        print("Text Characteristics Analysis")
        print("=" * 50)

        # 分析文本长度分布
        text_lengths = {
            'train': [len(text) for text in self.train_texts],
            'test': [len(text) for text in self.test_texts]
        }

        print("Text length statistics:")
        for dataset, lengths in text_lengths.items():
            print(f"  {dataset}: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}, "
                  f"min={np.min(lengths)}, max={np.max(lengths)}")

        # 分析词频分布
        print("\nAnalyzing word frequency distribution...")

        # 合并所有文本进行分词
        all_texts = self.train_texts + self.test_texts
        all_words = []

        for text in all_texts[:1000]:  # 只取前1000个文本避免计算量过大
            words = jieba.cut(text)
            all_words.extend([word for word in words if len(word.strip()) > 1])

        word_freq = Counter(all_words)

        print(f"Total unique words: {len(word_freq)}")
        print("Most frequent words:")
        for word, freq in word_freq.most_common(20):
            print(f"  {word}: {freq}")

        # 分析各类别的文本长度
        class_text_lengths = defaultdict(list)

        for text, label in zip(self.test_texts, self.test_labels):
            class_name = self.class_names[label]
            class_text_lengths[class_name].append(len(text))

        print("\nText length by class:")
        for class_name, lengths in class_text_lengths.items():
            if lengths:
                print(f"  {class_name}: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")

        return {
            'text_lengths': text_lengths,
            'word_freq': word_freq,
            'class_text_lengths': class_text_lengths
        }

    def generate_comprehensive_analysis_report(self):
        """生成综合分析报告"""
        print("=" * 60)
        print("Running Comprehensive Analysis")
        print("=" * 60)

        self.load_data_and_train_best_model()

        # 运行各项分析
        results = {}

        # 1. 整体性能分析
        results['overall_performance'] = self.analyze_overall_performance()

        # 2. 类别性能分析
        results['class_performance'] = self.analyze_class_performance()

        # 3. 混淆模式分析
        results['confusion_matrix'], results['confusion_pairs'] = self.analyze_confusion_patterns()

        # 4. 置信度分析
        results['confidence_analysis'] = self.analyze_prediction_confidence()

        # 5. 特征重要性分析
        results['class_features'], results['perm_importance'] = self.analyze_feature_importance()

        # 6. 错误案例分析
        results['error_analysis'] = self.analyze_error_cases()

        # 7. 文本特征分析
        results['text_characteristics'] = self.analyze_text_characteristics()

        # 保存结果和生成可视化
        self.save_analysis_results(results)

        return results

    def save_analysis_results(self, results):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results/analysis"
        os.makedirs(results_dir, exist_ok=True)

        # 保存各项结果
        if 'overall_performance' in results:
            results['overall_performance'].to_csv(
                f"{results_dir}/overall_performance_{timestamp}.csv", index=False
            )

        if 'class_performance' in results:
            results['class_performance'].to_csv(
                f"{results_dir}/class_performance_{timestamp}.csv"
            )

        if 'confusion_matrix' in results:
            results['confusion_matrix'].to_csv(
                f"{results_dir}/confusion_matrix_{timestamp}.csv"
            )

        if 'confusion_pairs' in results and not results['confusion_pairs'].empty:
            results['confusion_pairs'].to_csv(
                f"{results_dir}/confusion_pairs_{timestamp}.csv", index=False
            )

        # 保存置信度分析
        if 'confidence_analysis' in results:
            with open(f"{results_dir}/confidence_analysis_{timestamp}.json", 'w') as f:
                json.dump(results['confidence_analysis'], f, indent=2, default=str)

        # 生成可视化
        self.create_analysis_visualizations(results, results_dir, timestamp)

        print(f"\nAnalysis results saved to {results_dir}/")

    def create_analysis_visualizations(self, results, results_dir, timestamp):
        """创建分析可视化"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(20, 24))

        # 1. 整体性能对比
        ax1 = plt.subplot(4, 3, 1)
        if 'overall_performance' in results:
            perf_df = results['overall_performance']
            metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']

            x = np.arange(len(perf_df))
            width = 0.2

            for i, metric in enumerate(metrics):
                values = perf_df[metric]
                ax1.bar(x + i * width, values, width, label=metric, alpha=0.8)

            ax1.set_xlabel('Dataset')
            ax1.set_ylabel('Score')
            ax1.set_title('Overall Performance by Dataset')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(perf_df['Dataset'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 类别性能分布
        ax2 = plt.subplot(4, 3, 2)
        if 'class_performance' in results:
            class_perf = results['class_performance']
            ax2.hist(class_perf['f1-score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('F1 Score')
            ax2.set_ylabel('Number of Classes')
            ax2.set_title('Distribution of F1 Scores Across Classes')
            ax2.grid(True, alpha=0.3)

        # 3. 混淆矩阵热力图
        ax3 = plt.subplot(4, 3, 3)
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            # 只显示部分类别避免图太密集
            if len(cm) > 10:
                top_classes = results['class_performance'].head(10).index
                cm_subset = cm.loc[top_classes, top_classes]
            else:
                cm_subset = cm

            sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', ax=ax3,
                        cbar_kws={'shrink': 0.8})
            ax3.set_title('Confusion Matrix (Top Classes)')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax3.get_yticklabels(), rotation=0)

        # 4. 置信度分布
        ax4 = plt.subplot(4, 3, 4)
        max_proba = np.max(self.test_proba, axis=1)
        correct_mask = (self.test_pred == self.test_labels)

        ax4.hist(max_proba[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
        ax4.hist(max_proba[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
        ax4.set_xlabel('Prediction Confidence')
        ax4.set_ylabel('Count')
        ax4.set_title('Prediction Confidence Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 类别支持度分布
        ax5 = plt.subplot(4, 3, 5)
        if 'class_performance' in results:
            class_perf = results['class_performance']
            ax5.bar(range(len(class_perf)), class_perf['support'], alpha=0.7)
            ax5.set_xlabel('Class Index')
            ax5.set_ylabel('Support Count')
            ax5.set_title('Class Support Distribution')
            ax5.grid(True, alpha=0.3)

        # 6. 错误率 vs 支持度
        ax6 = plt.subplot(4, 3, 6)
        if 'class_performance' in results:
            class_perf = results['class_performance']
            error_rate = 1 - class_perf['f1-score']
            ax6.scatter(class_perf['support'], error_rate, alpha=0.7)
            ax6.set_xlabel('Support Count')
            ax6.set_ylabel('Error Rate (1 - F1)')
            ax6.set_title('Error Rate vs Class Support')
            ax6.grid(True, alpha=0.3)

        # 7. 文本长度分布
        ax7 = plt.subplot(4, 3, 7)
        if 'text_characteristics' in results:
            text_chars = results['text_characteristics']
            for dataset, lengths in text_chars['text_lengths'].items():
                ax7.hist(lengths, bins=50, alpha=0.7, label=dataset)
            ax7.set_xlabel('Text Length (characters)')
            ax7.set_ylabel('Count')
            ax7.set_title('Text Length Distribution')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. 置信度 vs 准确率
        ax8 = plt.subplot(4, 3, 8)
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []

        for i in range(len(confidence_bins) - 1):
            mask = (max_proba >= confidence_bins[i]) & (max_proba < confidence_bins[i + 1])
            if mask.sum() > 0:
                accuracy = correct_mask[mask].mean()
                bin_accuracies.append(accuracy)
            else:
                bin_accuracies.append(0)

        ax8.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=6)
        ax8.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        ax8.set_xlabel('Prediction Confidence')
        ax8.set_ylabel('Accuracy')
        ax8.set_title('Model Calibration')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. 特征重要性（如果可用）
        ax9 = plt.subplot(4, 3, 9)
        if 'perm_importance' in results:
            perm_imp = results['perm_importance']
            feature_names = self.vectorizer.get_feature_names_out()
            top_idx = np.argsort(perm_imp.importances_mean)[-15:]

            y_pos = np.arange(len(top_idx))
            ax9.barh(y_pos, perm_imp.importances_mean[top_idx])
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
            ax9.set_xlabel('Permutation Importance')
            ax9.set_title('Top Feature Importance')

        # 10. 错误类型分布
        ax10 = plt.subplot(4, 3, 10)
        if 'confusion_pairs' in results and not results['confusion_pairs'].empty:
            confusion_pairs = results['confusion_pairs']
            top_confusions = confusion_pairs.head(10)

            y_pos = np.arange(len(top_confusions))
            ax10.barh(y_pos, top_confusions['Count'])
            ax10.set_yticks(y_pos)
            labels = [f"{row['True Class'][:8]} → {row['Predicted Class'][:8]}"
                      for _, row in top_confusions.iterrows()]
            ax10.set_yticklabels(labels, fontsize=8)
            ax10.set_xlabel('Error Count')
            ax10.set_title('Top Confusion Pairs')

        # 11. 性能趋势（准确率、精确率、召回率、F1）
        ax11 = plt.subplot(4, 3, 11)
        if 'overall_performance' in results:
            perf_df = results['overall_performance']
            datasets = perf_df['Dataset']

            ax11.plot(datasets, perf_df['Accuracy'], 'o-', label='Accuracy', linewidth=2)
            ax11.plot(datasets, perf_df['Precision (Macro)'], 's-', label='Precision', linewidth=2)
            ax11.plot(datasets, perf_df['Recall (Macro)'], '^-', label='Recall', linewidth=2)
            ax11.plot(datasets, perf_df['F1 (Macro)'], 'd-', label='F1', linewidth=2)

            ax11.set_ylabel('Score')
            ax11.set_title('Performance Across Datasets')
            ax11.legend()
            ax11.grid(True, alpha=0.3)

        # 12. 模型总结
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')

        summary_text = "Model Analysis Summary\n\n"

        if 'confidence_analysis' in results:
            conf_analysis = results['confidence_analysis']
            summary_text += f"Overall Accuracy: {conf_analysis['overall_accuracy']:.4f}\n"
            summary_text += f"Total Predictions: {conf_analysis['total_predictions']}\n"
            summary_text += f"Correct Predictions: {conf_analysis['correct_predictions']}\n\n"

            summary_text += f"High Confidence (>0.9): {conf_analysis['high_confidence_count']}\n"
            summary_text += f"High Conf. Accuracy: {conf_analysis['high_confidence_accuracy']:.4f}\n\n"

            summary_text += f"Low Confidence (<0.5): {conf_analysis['low_confidence_count']}\n"
            summary_text += f"Low Conf. Accuracy: {conf_analysis['low_confidence_accuracy']:.4f}\n\n"

        if 'class_performance' in results:
            class_perf = results['class_performance']
            best_class = class_perf.index[0]
            worst_class = class_perf.index[-1]

            summary_text += f"Best Class: {best_class}\n"
            summary_text += f"F1: {class_perf.loc[best_class, 'f1-score']:.4f}\n\n"

            summary_text += f"Worst Class: {worst_class}\n"
            summary_text += f"F1: {class_perf.loc[worst_class, 'f1-score']:.4f}\n"

        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=11,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{results_dir}/comprehensive_analysis_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Analysis visualization saved to {results_dir}/comprehensive_analysis_{timestamp}.png")


def main():
    config = Config()
    experiment = AnalysisExperiment(config)

    print("Analysis Options:")
    print("1. Overall performance analysis")
    print("2. Class performance analysis")
    print("3. Confusion pattern analysis")
    print("4. Prediction confidence analysis")
    print("5. Feature importance analysis")
    print("6. Error case analysis")
    print("7. Text characteristics analysis")
    print("8. Comprehensive analysis (all above)")

    choice = input("Choose analysis type (1-8): ")

    experiment.load_data_and_train_best_model()

    if choice == "1":
        experiment.analyze_overall_performance()
    elif choice == "2":
        experiment.analyze_class_performance()
    elif choice == "3":
        experiment.analyze_confusion_patterns()
    elif choice == "4":
        experiment.analyze_prediction_confidence()
    elif choice == "5":
        experiment.analyze_feature_importance()
    elif choice == "6":
        experiment.analyze_error_cases()
    elif choice == "7":
        experiment.analyze_text_characteristics()
    else:
        experiment.generate_comprehensive_analysis_report()


if __name__ == "__main__":
    main()