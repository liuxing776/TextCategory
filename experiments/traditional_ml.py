#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统机器学习模型对比实验
包括多种经典ML算法和特征工程技术
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time

from config import Config
from data_processor import DataProcessor
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_class_distribution


class TraditionalMLExperiment:
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.results = {}

    def load_data(self):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")
        self.data_processor.load_data()
        self.data_processor.preprocess()

        self.train_texts, self.train_labels = self.data_processor.get_train_data()
        self.val_texts, self.val_labels = self.data_processor.get_val_data()
        self.test_texts, self.test_labels = self.data_processor.get_test_data()

        print(f"Train samples: {len(self.train_texts)}")
        print(f"Validation samples: {len(self.val_texts)}")
        print(f"Test samples: {len(self.test_texts)}")

    def get_feature_extractors(self):
        """定义多种特征提取器"""
        extractors = {
            'TF-IDF (1-2gram)': TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            ),
            'TF-IDF (1-3gram)': TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            ),
            'Count Vectorizer': CountVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            ),
            'TF-IDF + SVD': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True
                )),
                ('svd', TruncatedSVD(n_components=1000, random_state=42))
            ]),
            'Hashing Vectorizer': HashingVectorizer(
                n_features=2 ** 18,  # 262144 features
                ngram_range=(1, 2),
                alternate_sign=False
            )
        }
        return extractors

    def get_classifiers(self):
        """定义多种分类器"""
        classifiers = {
            'Logistic Regression': LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight='balanced'
            ),
            'SGD Classifier': SGDClassifier(
                random_state=42,
                max_iter=2000,
                class_weight='balanced'
            ),
            'Multinomial NB': MultinomialNB(alpha=1.0),
            'Complement NB': ComplementNB(alpha=1.0),
            'Linear SVM': LinearSVC(
                random_state=42,
                max_iter=3000,
                class_weight='balanced',
                dual=False
            ),
            'RBF SVM': SVC(
                kernel='rbf',
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        }
        return classifiers

    def train_and_evaluate(self, extractor_name, extractor, classifier_name, classifier):
        """训练和评估单个模型组合"""
        start_time = time.time()

        # 特征提取
        if hasattr(extractor, 'fit_transform'):
            X_train = extractor.fit_transform(self.train_texts)
            X_val = extractor.transform(self.val_texts)
            X_test = extractor.transform(self.test_texts)
        else:  # Pipeline
            X_train = extractor.fit_transform(self.train_texts, self.train_labels)
            X_val = extractor.transform(self.val_texts)
            X_test = extractor.transform(self.test_texts)

        # 训练分类器
        classifier.fit(X_train, self.train_labels)

        # 预测
        val_pred = classifier.predict(X_val)
        test_pred = classifier.predict(self.test_texts)

        # 获取预测概率（如果支持）
        val_proba = None
        test_proba = None
        if hasattr(classifier, 'predict_proba'):
            try:
                val_proba = classifier.predict_proba(X_val)
                test_proba = classifier.predict_proba(X_test)
            except:
                pass
        elif hasattr(classifier, 'decision_function'):
            try:
                val_proba = classifier.decision_function(X_val)
                test_proba = classifier.decision_function(X_test)
            except:
                pass

        # 计算指标
        val_metrics = calculate_metrics(self.val_labels, val_pred, val_proba)
        test_metrics = calculate_metrics(self.test_labels, test_pred, test_proba)

        training_time = time.time() - start_time

        return {
            'val_pred': val_pred,
            'test_pred': test_pred,
            'val_proba': val_proba,
            'test_proba': test_proba,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'model': classifier,
            'extractor': extractor
        }

    def run_comprehensive_experiment(self):
        """运行全面的传统ML实验"""
        print("=" * 60)
        print("Running Comprehensive Traditional ML Experiment")
        print("=" * 60)

        self.load_data()

        extractors = self.get_feature_extractors()
        classifiers = self.get_classifiers()

        all_results = []

        total_combinations = len(extractors) * len(classifiers)
        current_combination = 0

        for extractor_name, extractor in extractors.items():
            for classifier_name, classifier in classifiers.items():
                current_combination += 1

                print(f"\n[{current_combination}/{total_combinations}] Training {extractor_name} + {classifier_name}")

                try:
                    result = self.train_and_evaluate(
                        extractor_name, extractor, classifier_name, classifier
                    )

                    result_summary = {
                        'Feature Extractor': extractor_name,
                        'Classifier': classifier_name,
                        'Val Accuracy': result['val_metrics']['accuracy'],
                        'Test Accuracy': result['test_metrics']['accuracy'],
                        'Val F1': result['val_metrics']['f1_macro'],
                        'Test F1': result['test_metrics']['f1_macro'],
                        'Val Precision': result['val_metrics']['precision_macro'],
                        'Test Precision': result['test_metrics']['precision_macro'],
                        'Val Recall': result['val_metrics']['recall_macro'],
                        'Test Recall': result['test_metrics']['recall_macro'],
                        'Training Time': result['training_time']
                    }

                    all_results.append(result_summary)

                    # 保存详细结果
                    self.results[f"{extractor_name}_{classifier_name}"] = result

                    print(f"  Validation Accuracy: {result['val_metrics']['accuracy']:.4f}")
                    print(f"  Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
                    print(f"  Training Time: {result['training_time']:.2f}s")

                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue

        # 转换为DataFrame并排序
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values('Test Accuracy', ascending=False)

        self.generate_comprehensive_report(df_results)

        return df_results

    def run_optimized_experiment(self):
        """运行优化的实验（只选择最佳组合进行超参数调优）"""
        print("\n" + "=" * 60)
        print("Running Optimized Traditional ML Experiment")
        print("=" * 60)

        # 首先运行快速筛选
        quick_results = self.run_quick_screening()

        # 选择前3个最佳组合进行详细优化
        top_combinations = quick_results.head(3)

        optimized_results = []

        for idx, row in top_combinations.iterrows():
            extractor_name = row['Feature Extractor']
            classifier_name = row['Classifier']

            print(f"\nOptimizing {extractor_name} + {classifier_name}...")

            # 进行超参数优化
            optimized_result = self.optimize_hyperparameters(extractor_name, classifier_name)

            if optimized_result:
                optimized_results.append({
                    'Feature Extractor': extractor_name,
                    'Classifier': classifier_name,
                    'Best Val Accuracy': optimized_result['best_val_accuracy'],
                    'Best Test Accuracy': optimized_result['best_test_accuracy'],
                    'Best Params': optimized_result['best_params'],
                    'CV Score': optimized_result['cv_score']
                })

        return pd.DataFrame(optimized_results)

    def run_quick_screening(self):
        """快速筛选最佳特征提取器和分类器组合"""
        print("Running quick screening with subset of combinations...")

        # 选择最有效的特征提取器和分类器进行快速测试
        quick_extractors = {
            'TF-IDF (1-2gram)': TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2),
                min_df=2, max_df=0.95, sublinear_tf=True
            ),
            'TF-IDF + SVD': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=20000, ngram_range=(1, 2),
                    min_df=2, max_df=0.95, sublinear_tf=True
                )),
                ('svd', TruncatedSVD(n_components=500, random_state=42))
            ])
        }

        quick_classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM': LinearSVC(random_state=42, max_iter=2000),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Multinomial NB': MultinomialNB()
        }

        results = []

        for extractor_name, extractor in quick_extractors.items():
            for classifier_name, classifier in quick_classifiers.items():
                print(f"Testing {extractor_name} + {classifier_name}")

                try:
                    result = self.train_and_evaluate(
                        extractor_name, extractor, classifier_name, classifier
                    )

                    results.append({
                        'Feature Extractor': extractor_name,
                        'Classifier': classifier_name,
                        'Val Accuracy': result['val_metrics']['accuracy'],
                        'Test Accuracy': result['test_metrics']['accuracy'],
                        'Training Time': result['training_time']
                    })

                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        return pd.DataFrame(results).sort_values('Val Accuracy', ascending=False)

    def optimize_hyperparameters(self, extractor_name, classifier_name):
        """为特定组合优化超参数"""
        # 根据分类器类型定义超参数空间
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'Linear SVM': {
                'C': [0.1, 1, 10]
            },
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            },
            'Multinomial NB': {
                'alpha': [0.1, 1.0, 10.0]
            }
        }

        if classifier_name not in param_grids:
            return None

        # 重新创建特征提取器和分类器
        extractors = self.get_feature_extractors()
        classifiers = self.get_classifiers()

        extractor = extractors[extractor_name]
        classifier = classifiers[classifier_name]

        # 特征提取
        if hasattr(extractor, 'fit_transform'):
            X_train = extractor.fit_transform(self.train_texts)
            X_val = extractor.transform(self.val_texts)
            X_test = extractor.transform(self.test_texts)
        else:
            X_train = extractor.fit_transform(self.train_texts, self.train_labels)
            X_val = extractor.transform(self.val_texts)
            X_test = extractor.transform(self.test_texts)

        # 网格搜索
        grid_search = GridSearchCV(
            classifier,
            param_grids[classifier_name],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, self.train_labels)

        # 使用最佳参数预测
        best_model = grid_search.best_estimator_
        val_pred = best_model.predict(X_val)
        test_pred = best_model.predict(X_test)

        val_accuracy = accuracy_score(self.val_labels, val_pred)
        test_accuracy = accuracy_score(self.test_labels, test_pred)

        return {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'best_val_accuracy': val_accuracy,
            'best_test_accuracy': test_accuracy,
            'best_model': best_model
        }

    def generate_comprehensive_report(self, df_results):
        """生成全面的实验报告"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TRADITIONAL ML EXPERIMENT RESULTS")
        print("=" * 60)

        # 显示Top 10结果
        print("\nTop 10 Results:")
        print(df_results[['Feature Extractor', 'Classifier', 'Test Accuracy',
                          'Test F1', 'Training Time']].head(10).to_string(index=False))

        # 按特征提取器分组分析
        print("\nResults by Feature Extractor:")
        extractor_summary = df_results.groupby('Feature Extractor').agg({
            'Test Accuracy': ['mean', 'max', 'std'],
            'Training Time': ['mean', 'std']
        }).round(4)
        print(extractor_summary)

        # 按分类器分组分析
        print("\nResults by Classifier:")
        classifier_summary = df_results.groupby('Classifier').agg({
            'Test Accuracy': ['mean', 'max', 'std'],
            'Training Time': ['mean', 'std']
        }).round(4)
        print(classifier_summary)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results/traditional_ml"
        os.makedirs(results_dir, exist_ok=True)

        # 保存详细结果
        df_results.to_csv(f"{results_dir}/traditional_ml_results_{timestamp}.csv", index=False)

        # 保存分组统计
        extractor_summary.to_csv(f"{results_dir}/extractor_summary_{timestamp}.csv")
        classifier_summary.to_csv(f"{results_dir}/classifier_summary_{timestamp}.csv")

        # 绘制可视化结果
        self.plot_comprehensive_results(df_results, results_dir, timestamp)

        print(f"\nResults saved to {results_dir}/")

    def plot_comprehensive_results(self, df_results, results_dir, timestamp):
        """绘制全面的结果可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 热力图：特征提取器 vs 分类器的准确率
        pivot_acc = df_results.pivot(index='Classifier',
                                     columns='Feature Extractor',
                                     values='Test Accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Test Accuracy Heatmap: Classifier vs Feature Extractor')

        # 2. 箱线图：不同特征提取器的性能分布
        df_results.boxplot(column='Test Accuracy', by='Feature Extractor', ax=ax2)
        ax2.set_title('Test Accuracy Distribution by Feature Extractor')
        ax2.set_xlabel('Feature Extractor')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. 散点图：准确率 vs 训练时间
        for extractor in df_results['Feature Extractor'].unique():
            subset = df_results[df_results['Feature Extractor'] == extractor]
            ax3.scatter(subset['Training Time'], subset['Test Accuracy'],
                        label=extractor, alpha=0.7, s=60)
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_title('Accuracy vs Training Time Trade-off')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. 条形图：Top 10 模型
        top_10 = df_results.head(10)
        model_names = [f"{row['Feature Extractor'][:15]}+{row['Classifier'][:10]}"
                       for _, row in top_10.iterrows()]

        ax4.barh(range(len(top_10)), top_10['Test Accuracy'])
        ax4.set_yticks(range(len(top_10)))
        ax4.set_yticklabels(model_names, fontsize=8)
        ax4.set_xlabel('Test Accuracy')
        ax4.set_title('Top 10 Model Combinations')
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"{results_dir}/comprehensive_results_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 额外的可视化：分类器性能对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 分类器平均性能
        classifier_avg = df_results.groupby('Classifier')['Test Accuracy'].mean().sort_values(ascending=True)
        classifier_avg.plot(kind='barh', ax=ax1)
        ax1.set_title('Average Test Accuracy by Classifier')
        ax1.set_xlabel('Average Test Accuracy')

        # F1分数 vs 准确率
        ax2.scatter(df_results['Test Accuracy'], df_results['Test F1'], alpha=0.6)
        ax2.set_xlabel('Test Accuracy')
        ax2.set_ylabel('Test F1 Score')
        ax2.set_title('F1 Score vs Accuracy')

        # 添加对角线
        min_val = min(df_results['Test Accuracy'].min(), df_results['Test F1'].min())
        max_val = max(df_results['Test Accuracy'].max(), df_results['Test F1'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/classifier_analysis_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    config = Config()
    experiment = TraditionalMLExperiment(config)

    # 选择运行模式
    mode = input("Choose experiment mode (1: Quick screening, 2: Comprehensive, 3: Optimized): ")

    if mode == "1":
        results = experiment.run_quick_screening()
        print("\nQuick Screening Results:")
        print(results)
    elif mode == "2":
        experiment.run_comprehensive_experiment()
    elif mode == "3":
        experiment.run_optimized_experiment()
    else:
        print("Running comprehensive experiment by default...")
        experiment.run_comprehensive_experiment()


if __name__ == "__main__":
    main()