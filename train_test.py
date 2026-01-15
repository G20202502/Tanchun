import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from collections import Counter
from tqdm import tqdm


from utils import generate_training_samples, generate_test_samples,  set_seed, read_dataset_from_fold, read_dataset_from_csv, normalize_fs_samples, save_debug_logs,  save_behavior_samples_to_txt, get_concept_labels,get_test_behavior_sequence_samples,  TrainingBehaviorSequenceDataset, TestBehaviorSequenceDataset, evaluate_cls_model,visualize_embeddings, generate_text_anchored_prototypes

from concept_manager import ConceptManager

from models import ContrastiveClassificationModel, FocalLoss
from text_projection import TextPrototypeProjector

from supcon_loss import SupervisedContrastiveLoss, prototype_contrastive_loss, prototype_contrastive_loss_v2

# 指定 CSV 文件路径
training_data_json = 'llm_labeled_data/HOIC_train.json'
test_csv = "datasets/CIC18-DDoS-LOIC-HTTP/test_set.csv"

concept_manager = ConceptManager(["is high request rate", "is continuous requesting", "is fixed behavior pattern"])

seed = 1226
set_seed(seed)

test_df = pd.read_csv(test_csv)

time_slot_interval = 1 # 以多少秒为一个时间步分析源IP的行为
behavior_time_window_len = 15 # 每个源ip记录多少个时间步用于检测

# 打开并读取 JSON 文件, 加载标有概念label的样本
with open(training_data_json, 'r', encoding='utf-8') as file:
    training_behavior_sequence_samples = json.load(file)

training_samples, training_labels, training_concept_labels, training_ip_pair_list, training_start_time_slot_index_list = generate_training_samples(training_behavior_sequence_samples, behavior_time_window_len, concept_manager) # (num_samples,fs_len,features)

training_samples_norm, training_scale = normalize_fs_samples(training_samples, scaler=None)

training_dataset = TrainingBehaviorSequenceDataset(training_samples_norm, training_labels, training_concept_labels)

# 设置设备（GPU如果可用，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练参数
learning_rate = 1e-4
num_epochs = 32
input_dim = 11
shared_embedding_dim = 512
batch_size = 64

sbert_model = "all-mpnet-base-v2"

# 可以使用 DataLoader 对数据集进行批量加载
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True) # (batch_size,fs_len,features)

fixed_prototypes, text_prototype_emb_dim = generate_text_anchored_prototypes(concept_manager, device, sbert_model)
fixed_prototypes = fixed_prototypes.to(device)

# 初始化 模型
model = ContrastiveClassificationModel(input_dim=input_dim, embedding_dim=shared_embedding_dim, hidden_dim=64, dropout=0).to(device)

text_prototype_projector = TextPrototypeProjector(text_prototype_emb_dim, shared_embedding_dim).to(device)

# 设置权重
optimizer = torch.optim.AdamW(list(model.parameters()) + list(text_prototype_projector.parameters()), lr=1e-4)

# 设置权重
alpha = 0.6  # SupCon Loss 权重
beta = 1-alpha   # 分类损失权重，若不训练分类器可设为 0

# 准备工作
model.to(device)

cls_loss_fn = nn.CrossEntropyLoss()
# cls_loss_fn = FocalLoss(alpha=0.75, gamma=2)

for epoch in range(num_epochs):
    model.train()
    text_prototype_projector.train()
    
    total_loss = 0
    total_proto_loss = 0
    total_cls_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_x, class_labels, concept_ids = batch
        batch_x = batch_x.to(device)           # [B, L, F]
        class_labels = class_labels.to(device) # [B]
        concept_ids = concept_ids.to(device)   # [B]

        # 前向传播
        features, logits = model(batch_x)      # features: [B, D], logits: [B, num_classes]
        projected_text_prototypes = text_prototype_projector(fixed_prototypes)

        # 使用固定 prototype 的对比损失
        loss_proto = prototype_contrastive_loss(features, class_labels, concept_ids, projected_text_prototypes)

        # 分类损失（可选）
        if beta > 0:
            loss_cls = cls_loss_fn(logits, class_labels)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # 总损失
        loss = alpha * loss_proto + beta * loss_cls

        # 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        total_proto_loss += loss_proto.item()
        total_cls_loss += loss_cls.item() if beta > 0 else 0.0

    # 平均损失输出
    avg_loss = total_loss / len(train_dataloader)
    avg_proto_loss = total_proto_loss / len(train_dataloader)
    avg_cls = total_cls_loss / len(train_dataloader) if beta > 0 else 0.0

    print(f"[Epoch {epoch+1}/{num_epochs}] Total Loss: {avg_loss:.4f} | total_proto_loss: {avg_proto_loss:.4f} | CLS: {avg_cls:.4f}")

test_behavior_sequence_samples = get_test_behavior_sequence_samples(
    test_df, 
    time_slot_interval=time_slot_interval, 
    behavior_time_window_len=behavior_time_window_len
)

save_behavior_samples_to_txt(test_behavior_sequence_samples, "CIC17-DoS-slowhttptest_behaviour_log_new", "CIC17-DoS-slowhttptest_test")

test_samples, test_labels, test_flow_indices_list, test_ip_pair_list, test_start_time_slot_index_list = generate_test_samples(test_behavior_sequence_samples, behavior_time_window_len) # (num_samples,fs_len,features)

test_samples_norm, _ = normalize_fs_samples(test_samples, scaler=training_scale)

test_dataset = TestBehaviorSequenceDataset(test_samples_norm)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

ts_metrics, flow_metrics, all_preds, all_probs, all_labels , flow_detection_test_df,_= evaluate_cls_model(
    model, test_loader, test_labels, test_flow_indices_list, test_df, device
)

print("✅ 时间序列评估结果：")
for key, value in ts_metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")

print("✅ flow评估结果：")
for key, value in flow_metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")
