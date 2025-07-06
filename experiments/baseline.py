#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线实验 - 使用简单模型建立性能基准
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from config import Config
from data_processor import DataProcessor
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_class_distribution


class BaselineExperiment:
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.results = {}

    def load_data(self):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")
        self.data_processor.load_data()
        self.data_processor.preprocess()

        # 获取处理后的数据
        self.train_texts, self.train_labels = self.data_processor.get_train_data()
        self.val_texts, self.val_labels = self.data_processor.get_val_data()
        self.test_texts, self.test_labels = self.data_processor.get_test_data()

        print(f"Train samples: {len(self.train_texts)}")
        print(f"Validation samples: {len(self.val_texts)}")
        print(f"Test samples: {len(self.test_texts)}")

    def extract_features(self, method='tfidf'):
        """特征提取"""
        print(f"Extracting features using {method}...")

        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError("Unsupported feature extraction method")

        # 训练向量化器并转换训练数据
        X_train = vectorizer.fit_transform(self.train_texts)
        X_val = vectorizer.transform(self.val_texts)
        X_test = vectorizer.transform(self.test_texts)

        return X_train, X_val, X_test, vectorizer

    def train_baseline_models(self, X_train, X_val, X_test):
        """训练基线模型"""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Linear SVM': LinearSVC(
                random_state=42,
                max_iter=2000,
                class_weight='balanced'
            )
        }

        model_results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # 训练模型
            model.fit(X_train, self.train_labels)

            # 预测
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

            # 计算指标
            val_metrics = calculate_metrics(self.val_labels, val_pred)
            test_metrics = calculate_metrics(self.test_labels, test_pred)

            model_results[name] = {
                'model': model,
                'val_pred': val_pred,
                'test_pred': test_pred,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }

            print(f"{name} - Validation Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"{name} - Test Accuracy: {test_metrics['accuracy']:.4f}")

        return model_results

    def run_experiment(self):
        """运行完整的基线实验"""
        print("=" * 50)
        print("Running Baseline Experiment")
        print("=" * 50)

        # 加载数据
        self.load_data()

        # 尝试不同的特征提取方法
        feature_methods = ['tfidf', 'count']

        for method in feature_methods:
            print(f"\n{'=' * 30}")
            print(f"Feature Method: {method.upper()}")
            print(f"{'=' * 30}")

            # 特征提取
            X_train, X_val, X_test, vectorizer = self.extract_features(method)

            # 训练模型
            model_results = self.train_baseline_models(X_train, X_val, X_test)

            # 保存结果
            self.results[method] = model_results

            # 保存最佳模型
            best_model_name = max(model_results.keys(),
                                  key=lambda k: model_results[k]['val_metrics']['accuracy'])
            best_model = model_results[best_model_name]

            print(f"\nBest model for {method}: {best_model_name}")
            print(f"Validation Accuracy: {best_model['val_metrics']['accuracy']:.4f}")
            print(f"Test Accuracy: {best_model['test_metrics']['accuracy']:.4f}")

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 50)
        print("BASELINE EXPERIMENT RESULTS")
        print("=" * 50)

        # 汇总所有结果
        summary_results = []

        for feature_method in self.results:
            for model_name in self.results[feature_method]:
                result = self.results[feature_method][model_name]
                summary_results.append({
                    'Feature Method': feature_method,
                    'Model': model_name,
                    'Val Accuracy': result['val_metrics']['accuracy'],
                    'Test Accuracy': result['test_metrics']['accuracy'],
                    'Val F1': result['val_metrics']['f1_macro'],
                    'Test F1': result['test_metrics']['f1_macro']
                })

        # 转换为DataFrame并排序
        df_results = pd.DataFrame(summary_results)
        df_results = df_results.sort_values('Test Accuracy', ascending=False)

        print("\nSummary Results:")
        print(df_results.to_string(index=False))

        # 找出最佳配置
        best_idx = df_results['Test Accuracy'].idxmax()
        best_config = df_results.loc[best_idx]

        print(f"\nBest Configuration:")
        print(f"Feature Method: {best_config['Feature Method']}")
        print(f"Model: {best_config['Model']}")
        print(f"Test Accuracy: {best_config['Test Accuracy']:.4f}")
        print(f"Test F1: {best_config['Test F1']:.4f}")

        # 保存结果到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results/baseline"
        os.makedirs(results_dir, exist_ok=True)

        # 保存详细结果
        with open(f"{results_dir}/baseline_results_{timestamp}.json", 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python类型以便JSON序列化
            json_results = {}
            for feature_method in self.results:
                json_results[feature_method] = {}
                for model_name in self.results[feature_method]:
                    result = self.results[feature_method][model_name]
                    json_results[feature_method][model_name] = {
                        'val_metrics': {k: float(v) if isinstance(v, np.floating) else v
                                        for k, v in result['val_metrics'].items()},
                        'test_metrics': {k: float(v) if isinstance(v, np.floating) else v
                                         for k, v in result['test_metrics'].items()}
                    }
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        # 保存CSV格式的汇总结果
        df_results.to_csv(f"{results_dir}/baseline_summary_{timestamp}.csv", index=False)

        print(f"\nResults saved to {results_dir}/")

        # 绘制结果对比图
        self.plot_results_comparison(df_results, results_dir, timestamp)

        return df_results

    def plot_results_comparison(self, df_results, results_dir, timestamp):
        """绘制结果对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 不同模型在不同特征方法下的准确率对比
        pivot_acc = df_results.pivot(index='Model', columns='Feature Method', values='Test Accuracy')
        pivot_acc.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.legend(title='Feature Method')

        # 2. F1分数对比
        pivot_f1 = df_results.pivot(index='Model', columns='Feature Method', values='Test F1')
        pivot_f1.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title('Test F1 Score Comparison')
        ax2.set_ylabel('F1 Score')
        ax2.legend(title='Feature Method')

        # 3. 验证集vs测试集性能对比
        df_melted = pd.melt(df_results,
                            id_vars=['Feature Method', 'Model'],
                            value_vars=['Val Accuracy', 'Test Accuracy'],
                            var_name='Dataset',
                            value_name='Accuracy')

        sns.boxplot(data=df_melted, x='Dataset', y='Accuracy', ax=ax3)
        ax3.set_title('Validation vs Test Accuracy Distribution')

        # 4. 最佳模型的类别分布（如果有的话）
        ax4.text(0.1, 0.5, 'Baseline Experiment Summary:\n\n' +
                 f"Total configurations tested: {len(df_results)}\n" +
                 f"Best accuracy: {df_results['Test Accuracy'].max():.4f}\n" +
                 f"Worst accuracy: {df_results['Test Accuracy'].min():.4f}\n" +
                 f"Average accuracy: {df_results['Test Accuracy'].mean():.4f}\n" +
                 f"Std accuracy: {df_results['Test Accuracy'].std():.4f}",
                 transform=ax4.transAxes, fontsize=12,
                 verticalalignment='center')
        ax4.set_title('Experiment Summary')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(f"{results_dir}/baseline_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison plots saved to {results_dir}/baseline_comparison_{timestamp}.png")


def main():
    config = Config()
    experiment = BaselineExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()