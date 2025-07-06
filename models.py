#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型定义模块
包含TextCNN及其变体模型的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss实现 - 用于处理数据不平衡"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class TextCNN(nn.Module):
    """标准TextCNN模型实现"""

    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多尺度卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])

        # 分类层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        # 嵌入层初始化
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # 卷积层初始化
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

        # 全连接层初始化
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # 嵌入层
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 特征拼接
        concat_output = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes)*num_filters)

        # Dropout和分类
        output = self.dropout(concat_output)
        logits = self.fc(output)

        return logits

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'TextCNN',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'filter_sizes': self.filter_sizes,
            'num_filters': self.num_filters,
            'dropout_rate': self.dropout_rate
        }


class MultiChannelTextCNN(nn.Module):
    """多通道TextCNN - 可以使用不同的词向量作为不同通道"""

    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters,
                 num_channels=2, dropout=0.5):
        super(MultiChannelTextCNN, self).__init__()

        self.num_channels = num_channels

        # 多个嵌入层（不同通道）
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for _ in range(num_channels)
        ])

        # 卷积层（需要考虑多通道）
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim * num_channels, num_filters, kernel_size=k)
            for k in filter_sizes
        ])

        # 分类层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -0.1, 0.1)

        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # 多通道嵌入
        embedded_channels = []
        for embedding in self.embeddings:
            embedded = embedding(x)  # (batch_size, seq_len, embed_dim)
            embedded_channels.append(embedded)

        # 拼接所有通道
        multi_embedded = torch.cat(embedded_channels, dim=2)  # (batch_size, seq_len, embed_dim*num_channels)
        multi_embedded = multi_embedded.transpose(1, 2)  # (batch_size, embed_dim*num_channels, seq_len)

        # 卷积和池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(multi_embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 拼接和分类
        concat_output = torch.cat(conv_outputs, dim=1)
        output = self.dropout(concat_output)
        logits = self.fc(output)

        return logits


class AttentionTextCNN(nn.Module):
    """带注意力机制的TextCNN"""

    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout=0.5):
        super(AttentionTextCNN, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])

        # 注意力层
        self.attention = nn.Linear(len(filter_sizes) * num_filters, len(filter_sizes) * num_filters)

        # 分类层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

        nn.init.xavier_normal_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)

        # 卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 特征拼接
        concat_output = torch.cat(conv_outputs, dim=1)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(concat_output), dim=1)
        attended_output = concat_output * attention_weights

        # 分类
        output = self.dropout(attended_output)
        logits = self.fc(output)

        return logits


class DeepTextCNN(nn.Module):
    """深层TextCNN - 多层卷积"""

    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters,
                 num_layers=2, dropout=0.5):
        super(DeepTextCNN, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多层卷积
        self.conv_layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_convs = nn.ModuleList([
                nn.Conv1d(embed_dim if layer == 0 else num_filters,
                          num_filters, kernel_size=k, padding=k // 2)
                for k in filter_sizes
            ])
            self.conv_layers.append(layer_convs)

        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters * len(filter_sizes))
            for _ in range(num_layers)
        ])

        # 分类层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        for layer_convs in self.conv_layers:
            for conv in layer_convs:
                nn.init.kaiming_normal_(conv.weight)
                nn.init.constant_(conv.bias, 0)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)
        current_input = embedded.transpose(1, 2)

        # 多层卷积
        for layer, layer_convs in enumerate(self.conv_layers):
            layer_outputs = []
            for conv in layer_convs:
                conv_out = F.relu(conv(current_input))
                layer_outputs.append(conv_out)

            # 拼接当前层的所有卷积输出
            layer_concat = torch.cat(layer_outputs, dim=1)

            # 批归一化
            if layer < len(self.batch_norms):
                layer_concat = self.batch_norms[layer](layer_concat)

            current_input = layer_concat

        # 全局最大池化
        pooled = F.max_pool1d(current_input, current_input.size(2)).squeeze(2)

        # 分类
        output = self.dropout(pooled)
        logits = self.fc(output)

        return logits


def create_model(config, model_type='textcnn'):
    """模型工厂函数"""
    model_config = config.get_model_config()

    if model_type.lower() == 'textcnn':
        model = TextCNN(**model_config)
    elif model_type.lower() == 'multichannel':
        model = MultiChannelTextCNN(**model_config, num_channels=2)
    elif model_type.lower() == 'attention':
        model = AttentionTextCNN(**model_config)
    elif model_type.lower() == 'deep':
        model = DeepTextCNN(**model_config, num_layers=2)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


def load_pretrained_embeddings(embedding_layer, pretrained_path, vocab_to_idx):
    """加载预训练词向量"""
    try:
        print(f"加载预训练词向量: {pretrained_path}")

        # 这里需要根据具体的词向量格式进行调整
        # 示例：假设是word2vec格式
        pretrained_embeddings = {}
        with open(pretrained_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 2:
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    pretrained_embeddings[word] = vector

        # 更新嵌入层权重
        embed_dim = embedding_layer.weight.size(1)
        updated_count = 0

        with torch.no_grad():
            for word, idx in vocab_to_idx.items():
                if word in pretrained_embeddings:
                    embedding_layer.weight[idx] = torch.from_numpy(pretrained_embeddings[word])
                    updated_count += 1

        print(f"成功更新 {updated_count}/{len(vocab_to_idx)} 个词的预训练向量")
        return True

    except Exception as e:
        print(f"加载预训练词向量失败: {e}")
        return False


def get_model_summary(model, input_size):
    """获取模型摘要信息"""

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)

    # 计算模型大小（MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024

    summary = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'model_size_mb': size_mb,
        'input_size': input_size
    }

    return summary


def test_models():
    """测试所有模型"""
    from config import Config

    print("测试模型模块...")

    config = Config()
    batch_size = 4
    seq_len = config.max_length

    # 创建测试输入
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 测试所有模型类型
    model_types = ['textcnn', 'multichannel', 'attention', 'deep']

    for model_type in model_types:
        print(f"\n测试 {model_type.upper()} 模型:")

        try:
            model = create_model(config, model_type)
            model.eval()

            with torch.no_grad():
                output = model(test_input)

            print(f"  输入形状: {test_input.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")

            # 测试模型信息获取
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                print(f"  模型信息: {info['model_name']}, {info['total_params']:,} 参数")

        except Exception as e:
            print(f"  模型 {model_type} 测试失败: {e}")

    print("\n模型测试完成!")


if __name__ == "__main__":
    # 运行测试
    test_models()