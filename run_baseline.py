#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线实验运行示例
展示如何使用模块化框架进行TextCNN基线实验
"""

import os
import sys
import torch
from config import Config
from data_processor import DataProcessor
from models import create_model
from trainer import Trainer
from evaluator import ModelEvaluator, create_visualization_plots


def run_baseline_experiment(quick_test=False):
    """运行TextCNN基线实验"""

    print("=" * 60)
    print("TextCNN基线实验开始")
    print("=" * 60)

    # 1. 创建配置
    config = Config()
    if quick_test:
        config.quick_test = True
        config.num_epochs = 5
        print("快速测试模式已启用")

    config.save_config()  # 保存配置

    # 2. 数据预处理
    print("\n步骤1: 数据预处理")
    print("-" * 30)

    processor = DataProcessor(config)

    # 加载数据
    df = processor.load_data()

    # 准备训练数据
    train_loader, val_loader, test_loader = processor.prepare_data(df)

    # 3. 创建模型
    print("\n步骤2: 模型创建")
    print("-" * 30)

    model = create_model(config, 'textcnn')

    # 打印模型信息
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"模型类型: {model_info['model_name']}")
        print(f"参数数量: {model_info['total_params']:,}")
        print(f"词汇表大小: {model_info['vocab_size']:,}")
        print(f"嵌入维度: {model_info['embed_dim']}")
        print(f"卷积核尺寸: {model_info['filter_sizes']}")

    # 4. 训练模型
    print("\n步骤3: 模型训练")
    print("-" * 30)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=processor.class_weights
    )

    # 开始训练
    train_history = trainer.train()

    # 5. 评估模型
    print("\n步骤4: 模型评估")
    print("-" * 30)

    evaluator = ModelEvaluator(
        config=config,
        model=model,
        test_loader=test_loader,
        label_encoder=processor.label_encoder
    )

    # 综合评估
    evaluation_results = evaluator.comprehensive_evaluation()

    # 6. 可视化结果
    print("\n步骤5: 结果可视化")
    print("-" * 30)

    try:
        create_visualization_plots(config, evaluator, save_plots=True)
        print("可视化图表已生成")
    except Exception as e:
        print(f"可视化生成失败: {e}")

    # 7. 保存实验总结
    print("\n步骤6: 保存实验总结")
    print("-" * 30)

    experiment_summary = {
        'experiment_name': 'TextCNN_Baseline',
        'config': config.__dict__,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
        'train_history': train_history,
        'evaluation_results': evaluation_results['metrics'],
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch
    }

    # 保存总结
    import json
    summary_path = os.path.join(config.result_save_dir, 'baseline_experiment_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_summary, f, indent=2, ensure_ascii=False, default=str)

    # 8. 打印最终结果
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)

    metrics = evaluation_results['metrics']
    print(f"最终结果:")
    print(f"  训练最佳验证准确率: {trainer.best_val_acc:.4f} (第{trainer.best_epoch}轮)")
    print(f"  测试集准确率: {metrics['accuracy']:.4f}")
    print(f"  宏平均F1分数: {metrics['f1_macro']:.4f}")
    print(f"  加权平均F1分数: {metrics['f1_weighted']:.4f}")

    if metrics['auc_macro']:
        print(f"  宏平均AUC: {metrics['auc_macro']:.4f}")

    print(f"\n文件保存位置:")
    print(f"  模型文件: {config.model_save_dir}/")
    print(f"  结果文件: {config.result_save_dir}/")
    print(f"  图表文件: {config.plot_save_dir}/")

    print(f"\n主要输出文件:")
    print(f"  - best_model.pth: 最佳训练模型")
    print(f"  - train_history.json: 训练历史")
    print(f"  - evaluation_summary.json: 评估总结")
    print(f"  - detailed_predictions.csv: 详细预测结果")
    print(f"  - classification_report.csv: 分类报告")
    print(f"  - confusion_matrix.png: 混淆矩阵图")
    print(f"  - class_performance.png: 类别性能图")

    return experiment_summary


def check_requirements():
    """检查运行环境和数据文件"""
    print("检查运行环境...")

    # 检查数据文件
    data_file = "D:\\g3\\ML\\TextCategory2\\toutiao_cat_data.txt"
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'")
        print("请确保数据文件在当前目录下")
        return False

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，将使用CPU")

    # 检查Python包
    required_packages = ['torch', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'jieba', 'tqdm']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"缺少包: {missing_packages}")
        print("请安装: pip install " + " ".join(missing_packages))
        return False

    print("环境检查通过!")
    return True


def main():
    """主函数"""
    print("TextCNN文本分类实验 - 模块化版本")
    print("基于今日头条数据集的深度研究")
    print("=" * 60)

    # 检查环境
    if not check_requirements():
        return

    # 询问运行模式
    print("\n选择运行模式:")
    print("1. 完整实验 (推荐，约20-30分钟)")
    print("2. 快速测试 (用于验证代码，约5分钟)")

    choice = input("请输入选择 (1/2): ").strip()

    if choice == '2':
        print("选择快速测试模式")
        quick_test = True
    else:
        print("选择完整实验模式")
        quick_test = False

    try:
        # 运行实验
        summary = run_baseline_experiment(quick_test=quick_test)

        print("\n实验成功完成!")
        print("您可以:")
        print("1. 查看生成的图表和结果文件")
        print("2. 运行其他实验模块 (消融实验、对比实验等)")
        print("3. 修改config.py中的参数进行调优")

    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"\n实验执行出错: {e}")
        print("请检查错误信息并重试")


if __name__ == "__main__":
    main()