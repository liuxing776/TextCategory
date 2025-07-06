#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 - 分析不同组件对模型性能的影响
包括预处理步骤、特征选择、数据平衡等
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time

from config import Config
from data_processor import DataProcessor
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_class_distribution


class AblationExperiment:
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.results = {}

    def load_data(self):
        """加载原始数据"""
        print("Loading raw data...")
        self.data_processor.load_data()

        # 获取原始文本数据（未预处理）
        self.raw_train_texts, self.train_labels = self.data_processor.get_train_data(preprocessed=False)
        self.raw_val_texts, self.val_labels = self.data_processor.get_val_data(preprocessed=False)
        self.raw_test_texts, self.test_labels = self.data_processor.get_test_data(preprocessed=False)

        print(f"Train samples: {len(self.raw_train_texts)}")
        print(f"Validation samples: {len(self.raw_val_texts)}")
        print(f"Test samples: {len(self.raw_test_texts)}")

    def preprocess_text(self, texts, steps):
        """根据指定步骤预处理文本"""
        processed_texts = texts.copy()

        for step in steps:
            if step == 'remove_punctuation':
                processed_texts = [re.sub(r'[^\w\s]', '', text) for text in processed_texts]
            elif step == 'remove_numbers':
                processed_texts = [re.sub(r'\d+', '', text) for text in processed_texts]
            elif step == 'remove_english':
                processed_texts = [re.sub(r'[a-zA-Z]+', '', text) for text in processed_texts]
            elif step == 'remove_spaces':
                processed_texts = [re.sub(r'\s+', ' ', text).strip() for text in processed_texts]
            elif step == 'jieba_cut':
                processed_texts = [' '.join(jieba.cut(text)) for text in processed_texts]
            elif step == 'lower_case':
                processed_texts = [text.lower() for text in processed_texts]

        return processed_texts

    def experiment_preprocessing_steps(self):
        """实验不同预处理步骤的影响"""
        print("\n" + "=" * 50)
        print("Ablation Study: Preprocessing Steps")
        print("=" * 50)

        # 定义不同的预处理组合
        preprocessing_combinations = {
            'No Preprocessing': [],
            'Remove Punctuation': ['remove_punctuation'],
            'Remove Numbers': ['remove_numbers'],
            'Remove English': ['remove_english'],
            'Basic Clean': ['remove_punctuation', 'remove_numbers', 'remove_spaces'],
            'With Jieba': ['remove_punctuation', 'remove_numbers', 'jieba_cut'],
            'Full Preprocessing': ['remove_punctuation', 'remove_numbers',
                                   'remove_english', 'remove_spaces', 'jieba_cut'],
        }

        results = []

        for combo_name, steps in preprocessing_combinations.items():
            print(f"\nTesting: {combo_name}")

            # 预处理文本
            train_texts = self.preprocess_text(self.raw_train_texts, steps)
            val_texts = self.preprocess_text(self.raw_val_texts, steps)
            test_texts = self.preprocess_text(self.raw_test_texts, steps)

            # 特征提取
            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            X_train = vectorizer.fit_transform(train_texts)
            X_val = vectorizer.transform(val_texts)
            X_test = vectorizer.transform(test_texts)

            # 训练模型
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train, self.train_labels)

            # 评估
            val_pred = classifier.predict(X_val)
            test_pred = classifier.predict(X_test)

            val_metrics = calculate_metrics(self.val_labels, val_pred)
            test_metrics = calculate_metrics(self.test_labels, test_pred)

            result = {
                'Preprocessing': combo_name,
                'Steps': ', '.join(steps) if steps else 'None',
                'Val Accuracy': val_metrics['accuracy'],
                'Test Accuracy': test_metrics['accuracy'],
                'Val F1': val_metrics['f1_macro'],
                'Test F1': test_metrics['f1_macro']
            }

            results.append(result)
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

        return pd.DataFrame(results)

    def experiment_feature_selection(self):
        """实验不同特征选择方法的影响"""
        print("\n" + "=" * 50)
        print("Ablation Study: Feature Selection")
        print("=" * 50)

        # 使用最佳预处理方法
        train_texts = self.preprocess_text(self.raw_train_texts,
                                           ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        val_texts = self.preprocess_text(self.raw_val_texts,
                                         ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        test_texts = self.preprocess_text(self.raw_test_texts,
                                          ['remove_punctuation', 'remove_numbers', 'jieba_cut'])

        # 基础特征提取
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_full = vectorizer.fit_transform(train_texts)
        X_val_full = vectorizer.transform(val_texts)
        X_test_full = vectorizer.transform(test_texts)

        # 不同的特征选择方法
        feature_selection_methods = {
            'No Selection': None,
            'Chi2 (5000)': SelectKBest(chi2, k=5000),
            'Chi2 (10000)': SelectKBest(chi2, k=10000),
            'Mutual Info (5000)': SelectKBest(mutual_info_classif, k=5000),
            'Mutual Info (10000)': SelectKBest(mutual_info_classif, k=10000),
        }

        results = []

        for method_name, selector in feature_selection_methods.items():
            print(f"\nTesting: {method_name}")

            if selector is None:
                X_train = X_train_full
                X_val = X_val_full
                X_test = X_test_full
            else:
                X_train = selector.fit_transform(X_train_full, self.train_labels)
                X_val = selector.transform(X_val_full)
                X_test = selector.transform(X_test_full)

            # 训练模型
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train, self.train_labels)

            # 评估
            val_pred = classifier.predict(X_val)
            test_pred = classifier.predict(X_test)

            val_metrics = calculate_metrics(self.val_labels, val_pred)
            test_metrics = calculate_metrics(self.test_labels, test_pred)

            result = {
                'Feature Selection': method_name,
                'Features Count': X_train.shape[1],
                'Val Accuracy': val_metrics['accuracy'],
                'Test Accuracy': test_metrics['accuracy'],
                'Val F1': val_metrics['f1_macro'],
                'Test F1': test_metrics['f1_macro']
            }

            results.append(result)
            print(f"  Features: {X_train.shape[1]}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

        return pd.DataFrame(results)

    def experiment_class_balancing(self):
        """实验不同类别平衡方法的影响"""
        print("\n" + "=" * 50)
        print("Ablation Study: Class Balancing")
        print("=" * 50)

        # 使用最佳预处理方法
        train_texts = self.preprocess_text(self.raw_train_texts,
                                           ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        val_texts = self.preprocess_text(self.raw_val_texts,
                                         ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        test_texts = self.preprocess_text(self.raw_test_texts,
                                          ['remove_punctuation', 'remove_numbers', 'jieba_cut'])

        # 特征提取
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)

        # 分析类别分布
        print("\nOriginal class distribution:")
        class_counts = pd.Series(self.train_labels).value_counts().sort_index()
        print(class_counts)

        # 不同的类别平衡方法
        balancing_methods = {
            'No Balancing': None,
            'Class Weight': 'class_weight',
            'Random Oversampling': RandomOverSampler(random_state=42),
            'Random Undersampling': RandomUnderSampler(random_state=42),
            'SMOTE': SMOTE(random_state=42, k_neighbors=3),
            'SMOTE + ENN': SMOTEENN(random_state=42),
            'SMOTE + Tomek': SMOTETomek(random_state=42)
        }

        results = []

        for method_name, method in balancing_methods.items():
            print(f"\nTesting: {method_name}")

            try:
                if method is None:
                    # 无平衡
                    X_train_balanced = X_train
                    y_train_balanced = self.train_labels
                    classifier = LogisticRegression(max_iter=1000, random_state=42)

                elif method == 'class_weight':
                    # 类别权重
                    X_train_balanced = X_train
                    y_train_balanced = self.train_labels
                    classifier = LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight='balanced'
                    )

                else:
                    # 重采样方法
                    X_train_balanced, y_train_balanced = method.fit_resample(X_train, self.train_labels)
                    classifier = LogisticRegression(max_iter=1000, random_state=42)

                # 训练模型
                classifier.fit(X_train_balanced, y_train_balanced)

                # 评估
                val_pred = classifier.predict(X_val)
                test_pred = classifier.predict(X_test)

                val_metrics = calculate_metrics(self.val_labels, val_pred)
                test_metrics = calculate_metrics(self.test_labels, test_pred)

                result = {
                    'Balancing Method': method_name,
                    'Train Samples': len(y_train_balanced),
                    'Val Accuracy': val_metrics['accuracy'],
                    'Test Accuracy': test_metrics['accuracy'],
                    'Val F1': val_metrics['f1_macro'],
                    'Test F1': test_metrics['f1_macro'],
                    'Val F1 Weighted': val_metrics['f1_weighted'],
                    'Test F1 Weighted': test_metrics['f1_weighted']
                }

                results.append(result)

                print(f"  Train samples: {len(y_train_balanced)}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")

            except Exception as e:
                print(f"  Error with {method_name}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def experiment_ngram_ranges(self):
        """实验不同n-gram范围的影响"""
        print("\n" + "=" * 50)
        print("Ablation Study: N-gram Ranges")
        print("=" * 50)

        # 使用最佳预处理方法
        train_texts = self.preprocess_text(self.raw_train_texts,
                                           ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        val_texts = self.preprocess_text(self.raw_val_texts,
                                         ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        test_texts = self.preprocess_text(self.raw_test_texts,
                                          ['remove_punctuation', 'remove_numbers', 'jieba_cut'])

        # 不同的n-gram范围
        ngram_ranges = {
            'Unigram (1,1)': (1, 1),
            'Bigram (2,2)': (2, 2),
            'Trigram (3,3)': (3, 3),
            'Uni+Bi (1,2)': (1, 2),
            'Uni+Bi+Tri (1,3)': (1, 3),
            'Bi+Tri (2,3)': (2, 3)
        }

        results = []

        for range_name, ngram_range in ngram_ranges.items():
            print(f"\nTesting: {range_name}")

            # 特征提取
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=ngram_range,
                min_df=2
            )

            try:
                X_train = vectorizer.fit_transform(train_texts)
                X_val = vectorizer.transform(val_texts)
                X_test = vectorizer.transform(test_texts)

                # 训练模型
                classifier = LogisticRegression(max_iter=1000, random_state=42)
                classifier.fit(X_train, self.train_labels)

                # 评估
                val_pred = classifier.predict(X_val)
                test_pred = classifier.predict(X_test)

                val_metrics = calculate_metrics(self.val_labels, val_pred)
                test_metrics = calculate_metrics(self.test_labels, test_pred)

                result = {
                    'N-gram Range': range_name,
                    'Features Count': X_train.shape[1],
                    'Val Accuracy': val_metrics['accuracy'],
                    'Test Accuracy': test_metrics['accuracy'],
                    'Val F1': val_metrics['f1_macro'],
                    'Test F1': test_metrics['f1_macro']
                }

                results.append(result)

                print(f"  Features: {X_train.shape[1]}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

            except Exception as e:
                print(f"  Error with {range_name}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def experiment_max_features(self):
        """实验不同最大特征数的影响"""
        print("\n" + "=" * 50)
        print("Ablation Study: Maximum Features")
        print("=" * 50)

        # 使用最佳预处理方法
        train_texts = self.preprocess_text(self.raw_train_texts,
                                           ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        val_texts = self.preprocess_text(self.raw_val_texts,
                                         ['remove_punctuation', 'remove_numbers', 'jieba_cut'])
        test_texts = self.preprocess_text(self.raw_test_texts,
                                          ['remove_punctuation', 'remove_numbers', 'jieba_cut'])

        # 不同的最大特征数
        max_features_list = [1000, 2000, 5000, 10000, 20000, 50000, None]

        results = []

        for max_features in max_features_list:
            print(f"\nTesting max_features: {max_features}")

            # 特征提取
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )

            X_train = vectorizer.fit_transform(train_texts)
            X_val = vectorizer.transform(val_texts)
            X_test = vectorizer.transform(test_texts)

            # 训练模型
            start_time = time.time()
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train, self.train_labels)
            training_time = time.time() - start_time

            # 评估
            val_pred = classifier.predict(X_val)
            test_pred = classifier.predict(X_test)

            val_metrics = calculate_metrics(self.val_labels, val_pred)
            test_metrics = calculate_metrics(self.test_labels, test_pred)

            result = {
                'Max Features': max_features if max_features else 'All',
                'Actual Features': X_train.shape[1],
                'Training Time': training_time,
                'Val Accuracy': val_metrics['accuracy'],
                'Test Accuracy': test_metrics['accuracy'],
                'Val F1': val_metrics['f1_macro'],
                'Test F1': test_metrics['f1_macro']
            }

            results.append(result)

            print(f"  Actual features: {X_train.shape[1]}")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

        return pd.DataFrame(results)

    def run_all_ablation_studies(self):
        """运行所有消融实验"""
        print("=" * 60)
        print("Running Complete Ablation Studies")
        print("=" * 60)

        self.load_data()

        # 运行各项消融实验
        experiments = {
            'preprocessing': self.experiment_preprocessing_steps,
            'feature_selection': self.experiment_feature_selection,
            'class_balancing': self.experiment_class_balancing,
            'ngram_ranges': self.experiment_ngram_ranges,
            'max_features': self.experiment_max_features
        }

        all_results = {}

        for exp_name, exp_func in experiments.items():
            print(f"\n{'=' * 60}")
            print(f"Running {exp_name} experiment...")
            print(f"{'=' * 60}")

            try:
                results = exp_func()
                all_results[exp_name] = results

                print(f"\n{exp_name.title()} Results Summary:")
                print(results.sort_values('Test Accuracy', ascending=False).head())

            except Exception as e:
                print(f"Error in {exp_name} experiment: {str(e)}")
                continue

        # 生成综合报告
        self.generate_ablation_report(all_results)

        return all_results

    def generate_ablation_report(self, all_results):
        """生成消融实验综合报告"""
        print("\n" + "=" * 60)
        print("ABLATION STUDY COMPREHENSIVE REPORT")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results/ablation"
        os.makedirs(results_dir, exist_ok=True)

        # 保存所有结果到文件
        for exp_name, results in all_results.items():
            results.to_csv(f"{results_dir}/{exp_name}_results_{timestamp}.csv", index=False)

            print(f"\n{exp_name.upper()} - Best Configuration:")
            best_result = results.sort_values('Test Accuracy', ascending=False).iloc[0]
            print(best_result.to_string())

        # 绘制综合可视化
        self.plot_ablation_results(all_results, results_dir, timestamp)

        # 生成最佳配置建议
        best_configs = {}
        for exp_name, results in all_results.items():
            best_result = results.sort_values('Test Accuracy', ascending=False).iloc[0]
            best_configs[exp_name] = best_result.to_dict()

        # 保存最佳配置
        with open(f"{results_dir}/best_configurations_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(best_configs, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nAll ablation results saved to {results_dir}/")

    def plot_ablation_results(self, all_results, results_dir, timestamp):
        """绘制消融实验结果可视化"""
        # 创建大图包含所有实验结果
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        plot_idx = 0

        for exp_name, results in all_results.items():
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            if exp_name == 'preprocessing':
                # 预处理步骤对比
                results_sorted = results.sort_values('Test Accuracy')
                ax.barh(range(len(results_sorted)), results_sorted['Test Accuracy'])
                ax.set_yticks(range(len(results_sorted)))
                ax.set_yticklabels(results_sorted['Preprocessing'], fontsize=8)
                ax.set_xlabel('Test Accuracy')
                ax.set_title('Preprocessing Steps Impact')

            elif exp_name == 'feature_selection':
                # 特征选择对比
                ax.scatter(results['Features Count'], results['Test Accuracy'])
                for i, row in results.iterrows():
                    ax.annotate(row['Feature Selection'],
                                (row['Features Count'], row['Test Accuracy']),
                                fontsize=8, rotation=45)
                ax.set_xlabel('Number of Features')
                ax.set_ylabel('Test Accuracy')
                ax.set_title('Feature Selection Impact')

            elif exp_name == 'class_balancing':
                # 类别平衡对比
                results_sorted = results.sort_values('Test Accuracy')
                bars = ax.bar(range(len(results_sorted)), results_sorted['Test Accuracy'])
                ax.set_xticks(range(len(results_sorted)))
                ax.set_xticklabels(results_sorted['Balancing Method'], rotation=45, fontsize=8)
                ax.set_ylabel('Test Accuracy')
                ax.set_title('Class Balancing Impact')

                # 添加F1分数作为次要信息
                ax2 = ax.twinx()
                ax2.plot(range(len(results_sorted)), results_sorted['Test F1'], 'ro-', alpha=0.7)
                ax2.set_ylabel('Test F1', color='red')

            elif exp_name == 'ngram_ranges':
                # N-gram范围对比
                results_sorted = results.sort_values('Test Accuracy')
                ax.barh(range(len(results_sorted)), results_sorted['Test Accuracy'])
                ax.set_yticks(range(len(results_sorted)))
                ax.set_yticklabels(results_sorted['N-gram Range'], fontsize=8)
                ax.set_xlabel('Test Accuracy')
                ax.set_title('N-gram Range Impact')

            elif exp_name == 'max_features':
                # 最大特征数对比
                # 处理'All'值
                x_values = []
                for val in results['Max Features']:
                    if val == 'All':
                        x_values.append(results['Actual Features'].max())
                    else:
                        x_values.append(val)

                ax.plot(x_values, results['Test Accuracy'], 'o-')
                ax.set_xlabel('Max Features')
                ax.set_ylabel('Test Accuracy')
                ax.set_title('Max Features Impact')
                ax.set_xscale('log')

                # 添加训练时间作为次要信息
                ax2 = ax.twinx()
                ax2.plot(x_values, results['Training Time'], 'rs-', alpha=0.7)
                ax2.set_ylabel('Training Time (s)', color='red')

            plot_idx += 1

        # 隐藏多余的子图
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/ablation_comprehensive_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Ablation visualization saved to {results_dir}/ablation_comprehensive_{timestamp}.png")


def main():
    config = Config()
    experiment = AblationExperiment(config)

    # 选择运行模式
    print("Ablation Study Options:")
    print("1. Preprocessing steps")
    print("2. Feature selection")
    print("3. Class balancing")
    print("4. N-gram ranges")
    print("5. Max features")
    print("6. All ablation studies")

    choice = input("Choose experiment (1-6): ")

    experiment.load_data()

    if choice == "1":
        results = experiment.experiment_preprocessing_steps()
        print("\nPreprocessing Results:")
        print(results.sort_values('Test Accuracy', ascending=False))
    elif choice == "2":
        results = experiment.experiment_feature_selection()
        print("\nFeature Selection Results:")
        print(results.sort_values('Test Accuracy', ascending=False))
    elif choice == "3":
        results = experiment.experiment_class_balancing()
        print("\nClass Balancing Results:")
        print(results.sort_values('Test Accuracy', ascending=False))
    elif choice == "4":
        results = experiment.experiment_ngram_ranges()
        print("\nN-gram Range Results:")
        print(results.sort_values('Test Accuracy', ascending=False))
    elif choice == "5":
        results = experiment.experiment_max_features()
        print("\nMax Features Results:")
        print(results.sort_values('Test Accuracy', ascending=False))
    else:
        experiment.run_all_ablation_studies()


if __name__ == "__main__":
    main()