# simple_cnn_anomaly_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 1. CNN-Based BinaryClassifier
# ----------------------------
class CNNBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.feature_extractor = CNN1D(input_dim, hidden_dim, dropout)

        # Pool + Classifier head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        h = self.feature_extractor(x)         # [B, L, hidden_dim]
        h = h.permute(0, 2, 1)                # [B, hidden_dim, L]
        h_pooled = self.pool(h)               # [B, hidden_dim, 1]
        out = self.classifier(h_pooled)       # [B, 1]
        return out

# ----------------------------
# 2. CNN Feature Extractor
# ----------------------------

class CNN1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)       # [B, F, L]
        x = self.conv_layers(x)      # [B, hidden_dim, L]
        x = x.mean(dim=2)            # [B, hidden_dim]
        return x                     # 提取后的特征向量


class ContrastiveClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=128, num_classes=2, dropout=0):
        """
        input_dim: 每个时间步的输入特征维度
        embedding_dim: 用于对比学习的投影维度
        """
        super().__init__()
        self.feature_extractor = CNN1D(input_dim, hidden_dim, dropout)

        # 对比学习的 projection head（2 层 MLP）
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # 分类任务的 head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)            # [B, hidden_dim]

        # 对比学习表征
        projected = self.projection_head(features)      # [B, embedding_dim]
        projected= F.normalize(projected, dim=1)  # [B, D]

        # 分类输出
        logits = self.classifier(features)              # [B, num_classes]

        return projected, logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha  # 控制少数类权重
        self.gamma = gamma  # 控制难样本聚焦程度

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()
