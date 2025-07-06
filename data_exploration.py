#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
今日头条文本分类数据集分析 - 精简版
专注于TextCNN实验所需的核心信息
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import re
import os
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SimpleDataExplorer:
    """精简版数据探索器 - 专注于TextCNN实验核心需求"""

    def __init__(self, data_path, output_dir="results"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        self.category_mapping = {
            '100': '民生故事', '101': '文化', '102': '娱乐', '103': '体育',
            '104': '财经', '106': '房产', '107': '汽车', '108': '教育',
            '109': '科技', '110': '军事', '112': '旅游', '113': '国际',
            '114': '证券', '115': '农业', '116': '游戏'
        }

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")

        data_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('_!_')
                    if len(parts) >= 4:
                        data_list.append({
                            'category_code': parts[1],
                            'category_name': parts[2],
                            'title': parts[3]
                        })

        self.df = pd.DataFrame(data_list)
        self.df = self.df.dropna(subset=['title'])
        self.df = self.df[self.df['title'].str.len() > 5]

        # 添加中文类别名
        self.df['category_chinese'] = self.df['category_code'].map(self.category_mapping)

        print(f"✅ 数据加载完成，共 {len(self.df):,} 条记录")
        return self.df

    def analyze_for_textcnn(self):
        """专门为TextCNN实验分析关键信息"""
        print("\n" + "=" * 50)
        print("🎯 TextCNN实验关键信息分析")
        print("=" * 50)

        results = {}

        # 1. 基础统计
        total_samples = len(self.df)
        num_classes = self.df['category_code'].nunique()

        print(f"📊 基础信息:")
        print(f"  总样本数: {total_samples:,}")
        print(f"  类别数: {num_classes}")

        results['total_samples'] = total_samples
        results['num_classes'] = num_classes

        # 2. 类别分布 - 核心：数据不平衡问题
        category_counts = self.df['category_chinese'].value_counts()
        max_samples = category_counts.max()
        min_samples = category_counts.min()
        imbalance_ratio = max_samples / min_samples

        print(f"\n📈 类别分布（影响采样和损失函数设计）:")
        print(f"  数据不平衡比例: {imbalance_ratio:.1f}:1")
        print(f"  最多类别: {category_counts.index[0]} ({max_samples:,} 样本)")
        print(f"  最少类别: {category_counts.index[-1]} ({min_samples:,} 样本)")

        # 保存类别分布 - 用于后续数据划分
        category_dist = pd.DataFrame({
            '类别名称': category_counts.index,
            '样本数量': category_counts.values,
            '占比百分比': category_counts.values / total_samples * 100
        })
        category_dist.to_csv(f"{self.output_dir}/category_distribution.csv", index=False, encoding='utf-8-sig')
        print(f"  ✅ 保存类别分布: {self.output_dir}/category_distribution.csv")

        results['imbalance_ratio'] = imbalance_ratio
        results['max_category'] = category_counts.index[0]
        results['min_category'] = category_counts.index[-1]

        # 3. 文本长度分析 - 核心：确定max_length参数
        self.df['title_length'] = self.df['title'].str.len()
        length_stats = self.df['title_length'].describe()

        # 关键百分位数
        p90 = int(self.df['title_length'].quantile(0.90))
        p95 = int(self.df['title_length'].quantile(0.95))
        p99 = int(self.df['title_length'].quantile(0.99))

        print(f"\n📏 文本长度分析（确定max_length参数）:")
        print(f"  平均长度: {length_stats['mean']:.1f}")
        print(f"  90%文本长度 ≤ {p90} 字符  [推荐用于快速实验]")
        print(f"  95%文本长度 ≤ {p95} 字符  [推荐用于正式实验]")
        print(f"  99%文本长度 ≤ {p99} 字符  [完整覆盖但可能过长]")

        # 保存长度统计
        length_summary = pd.DataFrame({
            '统计指标': ['平均值', '中位数', '90%分位数', '95%分位数', '99%分位数', '最大值'],
            '数值': [length_stats['mean'], length_stats['50%'], p90, p95, p99, length_stats['max']],
            '说明': ['所有标题的平均长度', '中位数长度', '90%标题在此长度以下', '推荐最大长度', '几乎全覆盖长度', '最长标题长度']
        })
        length_summary.to_csv(f"{self.output_dir}/length_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"  ✅ 保存长度分析: {self.output_dir}/length_analysis.csv")

        results['recommended_max_length'] = p95
        results['avg_length'] = length_stats['mean']

        # 4. 词汇分析 - 核心：确定vocab_size参数
        print(f"\n🔤 词汇分析（确定vocab_size参数）:")

        # 分词并统计
        all_words = []
        for title in self.df['title'].sample(min(50000, len(self.df))):  # 采样加速
            words = jieba.lcut(title)
            words = [w for w in words if len(w) >= 2 and not re.match(r'^[\d\W]+$', w)]
            all_words.extend(words)

        word_counter = Counter(all_words)
        total_words = len(all_words)
        unique_words = len(word_counter)

        print(f"  总词数: {total_words:,}")
        print(f"  唯一词数: {unique_words:,}")

        # 计算不同词汇表大小的覆盖率
        vocab_sizes = [5000, 10000, 20000, 30000, 50000]
        coverage_info = []

        cumsum = 0
        for i, (word, count) in enumerate(word_counter.most_common(), 1):
            cumsum += count
            if i in vocab_sizes:
                coverage = cumsum / total_words * 100
                coverage_info.append({'词汇表大小': i, '覆盖率百分比': coverage})
                print(f"  词汇表 {i:,}: 覆盖 {coverage:.1f}% 词汇")

        # 保存词汇覆盖率分析
        coverage_df = pd.DataFrame(coverage_info)
        coverage_df.to_csv(f"{self.output_dir}/vocab_coverage.csv", index=False, encoding='utf-8-sig')
        print(f"  ✅ 保存词汇分析: {self.output_dir}/vocab_coverage.csv")

        # 推荐词汇表大小
        recommended_vocab = 30000 if unique_words > 30000 else min(unique_words, 20000)
        results['recommended_vocab_size'] = recommended_vocab
        results['unique_words'] = unique_words

        print(f"  💡 推荐词汇表大小: {recommended_vocab:,}")

        # 5. 生成TextCNN推荐参数
        print(f"\n🚀 TextCNN推荐参数:")
        textcnn_params = {
            'vocab_size': recommended_vocab,
            'embed_dim': 300,
            'max_length': p95,
            'num_classes': num_classes,
            'filter_sizes': [3, 4, 5],
            'num_filters': 128,
            'dropout': 0.5,
            'batch_size': 64 if total_samples > 100000 else 32,
            'learning_rate': 0.001
        }

        for param, value in textcnn_params.items():
            print(f"  {param}: {value}")

        # 保存推荐参数
        params_df = pd.DataFrame([
            {'参数名称': k, '参数值': v, '参数说明': self._get_param_description(k)}
            for k, v in textcnn_params.items()
        ])
        params_df.to_csv(f"{self.output_dir}/textcnn_params.csv", index=False, encoding='utf-8-sig')
        print(f"  ✅ 保存推荐参数: {self.output_dir}/textcnn_params.csv")

        # 6. 数据处理建议
        print(f"\n💡 关键建议:")
        suggestions = []

        if imbalance_ratio > 10:
            suggestions.append("严重数据不平衡，建议使用类别权重或focal loss")
        elif imbalance_ratio > 5:
            suggestions.append("中等数据不平衡，建议调整类别权重")
        else:
            suggestions.append("数据分布相对均衡")

        if p95 > 100:
            suggestions.append(f"文本较长，建议max_length={p95}并考虑截断策略")
        else:
            suggestions.append(f"文本长度适中，建议max_length={p95}")

        suggestions.append("使用分层采样确保训练/验证/测试集类别分布一致")
        suggestions.append("重点关注少数类别的F1-score")

        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

        # 保存分析结果摘要
        results.update(textcnn_params)
        results['suggestions'] = suggestions

        summary_df = pd.DataFrame([
            {'统计指标': '总样本数', '数值': total_samples, '说明': '数据集中的总新闻标题数量'},
            {'统计指标': '类别数量', '数值': num_classes, '说明': '新闻分类的类别总数'},
            {'统计指标': '数据不平衡比例', '数值': f"{imbalance_ratio:.1f}:1", '说明': '最多类别与最少类别的样本数比例'},
            {'统计指标': '推荐最大长度', '数值': p95, '说明': 'TextCNN模型建议的max_length参数'},
            {'统计指标': '推荐词汇表大小', '数值': recommended_vocab, '说明': 'TextCNN模型建议的vocab_size参数'},
            {'统计指标': '平均标题长度', '数值': f"{length_stats['mean']:.1f}", '说明': '所有新闻标题的平均字符长度'}
        ])
        summary_df.to_csv(f"{self.output_dir}/analysis_summary.csv", index=False, encoding='utf-8-sig')
        print(f"  ✅ 保存分析摘要: {self.output_dir}/analysis_summary.csv")

        return results

    def _get_param_description(self, param):
        """参数说明 - 中文版"""
        descriptions = {
            'vocab_size': '词汇表大小，基于词汇覆盖率分析确定',
            'embed_dim': '词向量维度，300是预训练词向量的常用维度',
            'max_length': '最大序列长度，基于95%分位数确定，平衡覆盖率和效率',
            'num_classes': '分类类别数量，等于数据集中的新闻类别总数',
            'filter_sizes': '卷积核尺寸列表，用于捕获不同长度的文本特征',
            'num_filters': '每种尺寸卷积核的数量，影响模型特征提取能力',
            'dropout': '随机失活比例，防止模型过拟合',
            'batch_size': '批次大小，基于数据集规模和内存限制调整',
            'learning_rate': '学习率，Adam优化器的常用初始值'
        }
        return descriptions.get(param, '模型参数配置')

    def create_essential_plots(self):
        """生成3个核心图表"""
        print(f"\n📊 生成核心可视化图表...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 类别分布
        category_counts = self.df['category_chinese'].value_counts()
        axes[0, 0].bar(range(len(category_counts)), category_counts.values)
        axes[0, 0].set_title('类别分布 - 评估数据不平衡', fontweight='bold')
        axes[0, 0].set_xticks(range(len(category_counts)))
        axes[0, 0].set_xticklabels(category_counts.index, rotation=45)
        axes[0, 0].set_ylabel('样本数量')

        # 2. 文本长度分布
        axes[0, 1].hist(self.df['title_length'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.df['title_length'].quantile(0.95), color='red', linestyle='--',
                           label=f'95%分位数: {int(self.df["title_length"].quantile(0.95))}')
        axes[0, 1].set_title('文本长度分布 - 确定max_length', fontweight='bold')
        axes[0, 1].set_xlabel('标题长度（字符）')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()

        # 3. 类别样本数对比（水平条形图）
        sorted_counts = category_counts.sort_values()
        axes[1, 0].barh(range(len(sorted_counts)), sorted_counts.values)
        axes[1, 0].set_title('类别不平衡程度', fontweight='bold')
        axes[1, 0].set_yticks(range(len(sorted_counts)))
        axes[1, 0].set_yticklabels(sorted_counts.index)
        axes[1, 0].set_xlabel('样本数量')

        # 4. 长度分布箱线图
        length_data = [self.df[self.df['category_chinese'] == cat]['title_length'].values
                       for cat in category_counts.index[:8]]  # 只显示前8个类别
        axes[1, 1].boxplot(length_data, labels=category_counts.index[:8])
        axes[1, 1].set_title('各类别长度分布', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylabel('标题长度（字符）')

        plt.tight_layout()
        plot_path = f"{self.output_dir}/essential_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 保存核心图表: {plot_path}")


def main():
    """主函数"""
    data_path = "D:\\g3\\ML\\TextCategory2\\toutiao_cat_data.txt"  # 请修改为实际路径

    try:
        print("🚀 开始今日头条数据集分析 - TextCNN专用版")
        print("=" * 50)

        # 创建分析器
        explorer = SimpleDataExplorer(data_path)

        # 加载数据
        df = explorer.load_data()

        # 核心分析
        results = explorer.analyze_for_textcnn()

        # 生成图表
        explorer.create_essential_plots()

        print(f"\n✅ 分析完成！生成文件:")
        print(f"📊 category_distribution.csv - 类别分布统计（用于数据划分）")
        print(f"📏 length_analysis.csv - 文本长度分析（确定max_length参数）")
        print(f"🔤 vocab_coverage.csv - 词汇覆盖率分析（确定vocab_size参数）")
        print(f"⚙️  textcnn_params.csv - TextCNN推荐参数配置")
        print(f"📋 analysis_summary.csv - 数据分析摘要报告")
        print(f"📈 essential_analysis.png - 核心可视化图表")

        print(f"\n🎯 接下来可以:")
        print(f"1. 使用推荐参数构建TextCNN模型")
        print(f"2. 基于类别分布进行数据划分")
        print(f"3. 根据不平衡程度设计损失函数")
        print(f"4. 开始模型训练和实验")

        return explorer

    except FileNotFoundError:
        print(f"❌ 找不到数据文件: {data_path}")
        print(f"请确认文件路径正确")
        return None
    except Exception as e:
        print(f"❌ 分析出错: {str(e)}")
        return None


if __name__ == "__main__":
    explorer = main()