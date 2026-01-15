import pandas as pd
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import torch
import os
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from collections import Counter
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unify_column_names(df):
    """
    统一DataFrame中的列名。
    """
    column_mapping = {
        'start_time': 'timestamp',
        'packets': 'total_fwd_packet',
        'bytes': 'total_length_of_fwd_packet'
    }
    return df.rename(columns=column_mapping)

def read_dataset_from_fold(folder_path):
    
    # 处理的时候，Flow的时间戳可能是乱序的
    
    """
    从指定文件夹中读取训练集和测试集CSV，转换时间戳为相对时间。

    参数:
    folder_path (str): 包含train.csv和test.csv的文件夹路径

    返回:
    tuple: (训练集DataFrame, 测试集DataFrame, 训练集所有原始数据（含恶意）DataFrame)
    """
    train_path = os.path.join(folder_path, "train_set.csv")
    test_path = os.path.join(folder_path, "test_set.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    # 读取 CSV 文件
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 确保包含 timestamp 列
    if 'timestamp' not in train_df.columns or 'timestamp' not in test_df.columns:
        raise ValueError("Both train and test CSVs must contain 'timestamp' column.")

    # 转换为 datetime
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    # 使用第一个时间戳作为基准
    train_base_time = train_df['timestamp'].iloc[0]
    test_base_time = test_df['timestamp'].iloc[0]

    # 相对时间戳（单位：秒）
    train_df['relative_time'] = (train_df['timestamp'] - train_base_time).dt.total_seconds().astype(int)
    test_df['relative_time'] = (test_df['timestamp'] - test_base_time).dt.total_seconds().astype(int)

    # 删除原始 timestamp，重命名 relative_time
    train_df = train_df.drop(columns=['timestamp']).rename(columns={'relative_time': 'timestamp'})
    test_df = test_df.drop(columns=['timestamp']).rename(columns={'relative_time': 'timestamp'})
    
    # 计算持续时间
    train_duration = train_df['timestamp'].max() - train_df['timestamp'].min()
    test_duration = test_df['timestamp'].max() - test_df['timestamp'].min()

    return train_df, test_df, train_duration, test_duration

def read_dataset_from_csv(csv_file_name):
    """
    从指定文件夹中读取训练集和测试集CSV，转换时间戳为相对时间。

    参数:
    folder_path (str): 包含train.csv和test.csv的文件夹路径

    返回:
    tuple: (训练集DataFrame, 测试集DataFrame, 训练集所有原始数据（含恶意）DataFrame)
    """
    test_path = csv_file_name
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    # 读取 CSV 文件
    test_df = pd.read_csv(test_path)
    
    test_df = unify_column_names(test_df)

    # 确保包含 timestamp 列
    if 'timestamp' not in test_df.columns:
        raise ValueError("test CSVs must contain 'timestamp' column.")

    # 转换为 datetime
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    # 使用第一个时间戳作为基准
    test_base_time = test_df['timestamp'].iloc[0]

    # 相对时间戳（单位：秒）
    test_df['relative_time'] = (test_df['timestamp'] - test_base_time).dt.total_seconds().astype(int)

    # 删除原始 timestamp，重命名 relative_time
    test_df = test_df.drop(columns=['timestamp']).rename(columns={'relative_time': 'timestamp'})
    
    # 计算持续时间
    test_duration = test_df['timestamp'].max() - test_df['timestamp'].min()

    return test_df, test_duration

def build_ip_pair_behavior_sequences_fast(flow_df, time_slot_interval, if_filter_dst_ip=False):
    
    # 是否开启按目的IP的分层过滤
    if if_filter_dst_ip:
        dst_ip_counts = flow_df['dst_ip'].value_counts()
        dst_ip_to_keep = set(dst_ip_counts[dst_ip_counts > 200].index)
        flow_df = flow_df[flow_df['dst_ip'].isin(dst_ip_to_keep)]
    
    ip_pair_info = defaultdict(lambda: {
        "behavior_sequence": [],
        "ip_pair_label": 0,
        "last_time_slot": -1  
    })

    flow_df = flow_df.copy()
    flow_df["flow_index"] = flow_df.index # 把待检测数据集中flow的原始index记录下来，使得后面计算指标时可以索引的到
    
    flow_df = flow_df.sort_values(by='start_time') # flow已经有序
    
    flow_df['time_slot'] = (flow_df['start_time'] // time_slot_interval).astype('uint16')

    for flow_item in flow_df.itertuples(index=False):
        flow_index = flow_item.flow_index
        
        src_ip = flow_item.src_ip
        dst_ip = flow_item.dst_ip

        # 建立ip-pair的哈希表
        ip_pair = f"{src_ip}>{dst_ip}"
        
        time_slot_index = flow_item.time_slot

        current_info = ip_pair_info[ip_pair]

        if flow_item.label != 'BENIGN':
            current_info['ip_pair_label'] = 1

        if current_info["last_time_slot"] != time_slot_index:
            # 新时间片：新建行为记录
            new_behavior = {
                "time_slot_index": time_slot_index,
                "flow_count": 1,
                "total_packets": flow_item.total_fwd_packet,
                "total_bytes": flow_item.total_length_of_fwd_packet,
                "ave_pkts_per_flow": flow_item.total_fwd_packet,
                "max_pkts_per_flow": flow_item.total_fwd_packet,
                "min_pkts_per_flow": flow_item.total_fwd_packet,
                "std_pkts_per_flow": 0,
                "ave_bytes_per_flow": flow_item.total_length_of_fwd_packet,
                "max_bytes_per_flow": flow_item.total_length_of_fwd_packet,
                "min_bytes_per_flow": flow_item.total_length_of_fwd_packet,
                "std_bytes_per_flow": 0,
                "m2_pkts": 0,
                "m2_bytes": 0,
                "flow_indices": [flow_index] 
            }
            current_info["behavior_sequence"].append(new_behavior)
            current_info["last_time_slot"] = time_slot_index
        else:
            b = current_info["behavior_sequence"][-1]
            b["flow_count"] += 1
            b["total_packets"] += flow_item.total_fwd_packet
            b["total_bytes"] += flow_item.total_length_of_fwd_packet
            b["max_pkts_per_flow"] = max(b["max_pkts_per_flow"], flow_item.total_fwd_packet)
            b["min_pkts_per_flow"] = min(b["min_pkts_per_flow"], flow_item.total_fwd_packet)
            b["max_bytes_per_flow"] = max(b["max_bytes_per_flow"], flow_item.total_length_of_fwd_packet)
            b["min_bytes_per_flow"] = min(b["min_bytes_per_flow"], flow_item.total_length_of_fwd_packet)
            b["flow_indices"].append(flow_index)

            # 优化后的 Welford 计算
            fc = b["flow_count"]
            inv_fc = 1.0 / fc

            delta_pkts = flow_item.total_fwd_packet - b["ave_pkts_per_flow"]
            b["ave_pkts_per_flow"] += delta_pkts * inv_fc
            delta2_pkts = flow_item.total_fwd_packet - b["ave_pkts_per_flow"]
            b["m2_pkts"] += delta_pkts * delta2_pkts

            delta_bytes = flow_item.total_length_of_fwd_packet - b["ave_bytes_per_flow"]
            b["ave_bytes_per_flow"] += delta_bytes * inv_fc
            delta2_bytes = flow_item.total_length_of_fwd_packet - b["ave_bytes_per_flow"]
            b["m2_bytes"] += delta_bytes * delta2_bytes

            if fc > 1:
                b["std_pkts_per_flow"] = (b["m2_pkts"] / (fc - 1)) ** 0.5
                b["std_bytes_per_flow"] = (b["m2_bytes"] / (fc - 1)) ** 0.5

    return ip_pair_info

def get_test_behavior_sequence_samples(flow_df, time_slot_interval, behavior_time_window_len=15):
    ip_pair_info = build_ip_pair_behavior_sequences_fast(flow_df, time_slot_interval)
    all_test_samples = []

    for ip_pair, info in ip_pair_info.items():
        
        # 按 time_slot_index 排序
        behaviors = sorted(info["behavior_sequence"], key=lambda x: x["time_slot_index"])
        idx = 0

        while idx < len(behaviors):
            sample_behaviors = []
            sample_flow_indices = []

            first_slot = behaviors[idx]["time_slot_index"]
            current_sample_slots = {first_slot}

            # 拷贝第一个行为并处理相对时间
            first_behavior = behaviors[idx].copy()
            first_behavior["time_slot_index"] = 0
            sample_behaviors.append(first_behavior)
            sample_flow_indices.extend(first_behavior["flow_indices"])
            idx += 1

            while idx < len(behaviors):
                current_slot = behaviors[idx]["time_slot_index"]

                if (current_slot - first_slot >= behavior_time_window_len) or (current_slot in current_sample_slots):
                    break

                behavior = behaviors[idx].copy()
                behavior["time_slot_index"] -= first_slot  # 转为相对时间
                sample_behaviors.append(behavior)
                sample_flow_indices.extend(behavior["flow_indices"])
                current_sample_slots.add(current_slot)
                idx += 1

            if sample_behaviors:
                clean_behaviors = [{k: b[k] for k in feature_names} for b in sample_behaviors]
                all_test_samples.append({
                    "ip_pair": ip_pair,
                    "start_time_slot_index": first_slot,
                    "behaviour_sequence": clean_behaviors,
                    "flow_indices": sample_flow_indices,
                    "label": info["ip_pair_label"]
                })

    return all_test_samples

def save_behavior_samples_to_txt(samples, output_dir, filename_prefix):
    os.makedirs(output_dir, exist_ok=True)

    benign_path = os.path.join(output_dir, f"{filename_prefix}_benign_sample.txt")
    attack_path = os.path.join(output_dir, f"{filename_prefix}_attack_sample.txt")

    benign_file = open(benign_path, "w")
    attack_file = open(attack_path, "w")
    
    for sample in samples:
        target_file = attack_file if sample['label'] == 1 else benign_file

        # 写入样本头信息
        target_file.write(f"# ip_pair: {sample['ip_pair']}\n")
        target_file.write(f"# start_time_slot_index: {sample['start_time_slot_index']}\n")
        target_file.write(f"# label: {sample['label']}\n")
        target_file.write(f"# flow_indices: {sample['flow_indices']}\n")

        # 写入每个时间步的行为信息（每行一个时间步）
        for step in sample['behaviour_sequence']:
            ordered_step = OrderedDict((key, step[key]) for key in feature_names)
            target_file.write(json.dumps(ordered_step) + "\n")

        target_file.write("\n")  # 用空行分隔样本

    benign_file.close()
    attack_file.close()

    print(f"Saved benign samples to: {benign_path}")
    print(f"Saved attack samples to: {attack_path}")


# 定义 behavior 具有的feature结构
feature_names = [
    "time_slot_index",
    "flow_count",
    "total_packets",
    "total_bytes",
    "ave_pkts_per_flow",
    "max_pkts_per_flow",
    "min_pkts_per_flow",
    "std_pkts_per_flow",
    "ave_bytes_per_flow",
    "max_bytes_per_flow",
    "min_bytes_per_flow",
    "std_bytes_per_flow"
]

# 提取对应的behavior特征组成序列
def extract_features(behavior):
    return [behavior[feature] for feature in feature_names if feature != "time_slot_index"]

def concept_id_adapt(sample):
    concept_label = sample["concept_label"]

    behaviour_sequence = sample["behaviour_sequence"]
        
    if len(behaviour_sequence)==1:
        concept_label["is fixed behavior pattern"] = False

    return concept_label

def generate_training_samples(behavior_sequence_samples, behavior_time_window_len, concept_manager):
    training_samples = []
    training_labels = []
    training_concept_labels = []
    ip_pair_list = []
    start_time_slot_index_list = []

    for sample in behavior_sequence_samples:
        
        behaviour_sequence = sample["behaviour_sequence"]
        label = sample["label"]
        concept_label = sample["concept_label"]
        
        concept_id = concept_manager.dict_to_id(concept_label)
        
        ip_pair = sample["ip_pair"]
        start_time_slot_index = sample["start_time_slot_index"]

        # 构建 time_slot_index -> feature 的映射（相对位置从 0 开始）
        base_time_slot = behaviour_sequence[0]["time_slot_index"]
        time_slot_to_features = {
            b["time_slot_index"] - base_time_slot: extract_features(b)
            for b in behaviour_sequence
        }

        # 构造定长序列并填补缺失时间步为全零向量
        feature_len = len(next(iter(time_slot_to_features.values())))
        padded_sequence = [time_slot_to_features.get(i, [0] * feature_len) for i in range(behavior_time_window_len)]

        training_samples.append(np.array(padded_sequence))
        training_labels.append(label)
        training_concept_labels.append(concept_id)
        ip_pair_list.append(ip_pair)
        start_time_slot_index_list.append(start_time_slot_index)

    return (
        np.array(training_samples),
        np.array(training_labels),
        np.array(training_concept_labels),
        ip_pair_list,
        start_time_slot_index_list
    )


def generate_test_samples(behavior_sequence_samples, behavior_time_window_len):
    training_samples = []
    training_labels = []
    flow_indices_list = []
    ip_pair_list = []
    start_time_slot_index_list = []

    for sample in behavior_sequence_samples:
        behaviour_sequence = sample["behaviour_sequence"]
        label = sample["label"]
        flow_indices = sample["flow_indices"]
        
        ip_pair = sample["ip_pair"]
        start_time_slot_index = sample["start_time_slot_index"]

        # 构建 time_slot_index -> feature 的映射（相对位置从 0 开始）
        base_time_slot = behaviour_sequence[0]["time_slot_index"]
        time_slot_to_features = {
            b["time_slot_index"] - base_time_slot: extract_features(b)
            for b in behaviour_sequence
        }

        # 构造定长序列并填补缺失时间步为全零向量
        feature_len = len(next(iter(time_slot_to_features.values())))
        padded_sequence = [time_slot_to_features.get(i, [0] * feature_len) for i in range(behavior_time_window_len)]

        training_samples.append(np.array(padded_sequence))
        training_labels.append(label)
        flow_indices_list.append(flow_indices)
        ip_pair_list.append(ip_pair)
        start_time_slot_index_list.append(start_time_slot_index)

    return (
        np.array(training_samples),
        np.array(training_labels),
        flow_indices_list,
        ip_pair_list,
        start_time_slot_index_list
    )

def get_concept_labels(training_concept_labels, concept_manager):
    # 生成概念标签，后面用大模型替换
    concept_labels = []
    for concept in training_concept_labels:
        concept_id = concept_manager.dict_to_id(concept)
        concept_labels.append(concept_id)
    
    return np.array(concept_labels)

def normalize_fs_samples(samples, scaler=None):
    """
    对形状为 (num_samples, L, N) 的多元时间序列数据进行归一化（MaxAbsScaler）。
    如果传入了 scaler，则使用已有的 scaler 对数据进行 transform；
    否则，新建一个 scaler 并 fit_transform。
    
    返回归一化后的数据和 scaler。
    """
    num_samples, L, N = samples.shape

    # reshape 为 (num_samples * L, N)
    reshaped = samples.reshape(-1, N)

    # 如果已有 scaler 就 transform，否则 fit_transform
    if scaler is not None:
        normalized = scaler.transform(reshaped)
    else:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped)

    # reshape 回原始形状
    normalized_samples = normalized.reshape(num_samples, L, N)

    return normalized_samples, scaler


class TrainingBehaviorSequenceDataset(Dataset):
    def __init__(self, input_samples, training_labels, concept_labels):
        self.input_samples = input_samples
        self.training_labels = training_labels
        self.concept_labels = concept_labels

    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        input_sample = torch.tensor(self.input_samples[idx], dtype=torch.float32)
        training_label = torch.tensor(self.training_labels[idx], dtype=torch.long)
        concept_label = torch.tensor(self.concept_labels[idx], dtype=torch.long)
        return input_sample, training_label, concept_label
    
class TestBehaviorSequenceDataset(Dataset):
    def __init__(self, input_samples):
        self.input_samples = input_samples

    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        input_sample = torch.tensor(self.input_samples[idx], dtype=torch.float32)
        return input_sample, idx

def evaluate_cls_model(model, dataloader, test_labels, test_flow_indices_list, test_df, device):
    
    # 以下为time series级别的相关指标
    all_ts_probs = []
    all_ts_preds = []
    all_ts_indices = []
    all_ts_latent_vectors = []

    model.eval()
    
    eval_start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            
            x, idx = batch
            x = x.to(device)

            latent_vector, logits = model(x)
            probs = F.softmax(logits, dim=1)         # [B, 2]
            y_prob = probs[:, 1].cpu().numpy()       # 第1类（恶意）概率
            y_pred_label = torch.argmax(logits, dim=1).cpu().numpy()  # [B]

            all_ts_probs.extend(y_prob)
            all_ts_preds.extend(y_pred_label)
            all_ts_indices.extend(idx)
            all_ts_latent_vectors.extend(latent_vector.cpu().numpy())
    
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    print(f"评估耗时: {eval_time:.4f} 秒")
    
    # 由索引查出真实标签
    all_ts_true_labels = [test_labels[i] for i in all_ts_indices]

    # 样本级别指标
    ts_acc = accuracy_score(all_ts_true_labels, all_ts_preds)
    ts_auc = roc_auc_score(all_ts_true_labels, all_ts_preds)
    ts_precision = precision_score(all_ts_true_labels, all_ts_preds, zero_division=0)
    ts_recall = recall_score(all_ts_true_labels, all_ts_preds, zero_division=0)
    ts_f1 = f1_score(all_ts_true_labels, all_ts_preds, zero_division=0)
    cm = confusion_matrix(all_ts_true_labels, all_ts_preds)
    tn, fp, fn, tp = cm.ravel()
    ts_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    ts_metrics = {
        "ts_accuracy": ts_acc,
        "ts_auc": ts_auc,
        "ts_precision": ts_precision,
        "ts_recall": ts_recall,
        "ts_f1": ts_f1,
        "ts_false_positive_rate": ts_fpr
    }

    # 计算flow级别检测结果
    num_flows = len(test_df)
    flow_preds = [-1]*num_flows
    
    num_pred_flows = 0
    
    for ts_idx, ts_pred in zip(all_ts_indices, all_ts_preds):
        flow_indices = test_flow_indices_list[ts_idx]
        for flow_idx in flow_indices:
            flow_preds[flow_idx] = ts_pred
            num_pred_flows += 1

    # 4. 获取真实标签：BENIGN → 0，其它 → 1
    flow_true_labels = [0 if label == "BENIGN" else 1 for label in test_df["label"]]
    
    print("num_pred_flows:",num_pred_flows)
    print("len_flow_true_labels:",len(flow_true_labels))

    # 5. 计算评估指标
    flow_acc = accuracy_score(flow_true_labels, flow_preds)
    flow_auc = roc_auc_score(flow_true_labels, flow_preds)
    flow_precision = precision_score(flow_true_labels, flow_preds, zero_division=0)
    flow_recall = recall_score(flow_true_labels, flow_preds, zero_division=0)
    flow_f1 = f1_score(flow_true_labels, flow_preds, zero_division=0)
    flow_cm = confusion_matrix(flow_true_labels, flow_preds)
    tn, fp, fn, tp = flow_cm.ravel()
    flow_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # 6. 植入预测标签
    test_df["predicted_label"] = flow_preds

    # 7. 打包结果
    flow_metrics = {
        "flow_accuracy": flow_acc,
        "flow_auc": flow_auc,
        "flow_precision": flow_precision,
        "flow_recall": flow_recall,
        "flow_f1": flow_f1,
        "flow_false_positive_rate": flow_fpr
    }

    return ts_metrics, flow_metrics, all_ts_preds, all_ts_probs, all_ts_true_labels, test_df, all_ts_latent_vectors


def interpret_eva(model, dataloader, device):
    
    # 以下为time series级别的相关指标
    all_ts_probs = []
    all_ts_preds = []
    all_ts_indices = []
    
    # 存储每个样本投影后的隐向量
    all_ts_latent_vectors = []

    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            
            x, idx = batch
            x = x.to(device)

            latent_vector, logits = model(x)
            probs = F.softmax(logits, dim=1)         # [B, 2]
            y_prob = probs[:, 1].cpu().numpy()       # 第1类（恶意）概率
            y_pred_label = torch.argmax(logits, dim=1).cpu().numpy()  # [B]

            all_ts_probs.extend(y_prob)
            all_ts_preds.extend(y_pred_label)
            all_ts_indices.extend(idx)
            
            all_ts_latent_vectors.extend(latent_vector.cpu().numpy())
            

    return all_ts_preds, all_ts_probs, all_ts_latent_vectors



def save_debug_logs(
    all_preds, all_labels, all_concepts, flow_sample_index_list,
    test_ip_pair_list, test_start_time_slot_index_list, test_samples,
    concept_manager, output_dir="debug_logs_txt"
):

    os.makedirs(output_dir, exist_ok=True)

    logs = {
        "benign_correct": [],
        "benign_misclassified": [],
        "malicious_correct": [],
        "malicious_misclassified": [],
    }

    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        index = flow_sample_index_list[i]
        ip_pair = test_ip_pair_list[index]
        time_slot = test_start_time_slot_index_list[index]
        sample = test_samples[index]  # shape: (seq_len, feature_dim)
        concept_vector = all_concepts[i].numpy()

        # 概念解释文本
        concept_text = "\n".join([
            f"  - {concept_manager.get_name(j)}: {concept_vector[j]:.4f}"
            for j in range(len(concept_vector))
        ])

        # 样本行为序列文本（展示部分维度）
        sample_text = "\n".join([
            f"  Timestep {t}: " + ", ".join([f"{v:.4f}" for v in sample[t]])
            for t in range(len(sample))
        ])

        block = (
            f"Sample Index: {index}\n"
            f"IP Pair: {ip_pair}\n"
            f"Start Time Slot: {time_slot}\n"
            f"True Label: {'BENIGN' if label == 0 else 'MALICIOUS'}\n"
            f"Predicted Label: {'BENIGN' if pred == 0 else 'MALICIOUS'}\n"
            f"Concept Vector:\n{concept_text}\n"
            f"Sample Sequence:\n{sample_text}\n"
            f"{'-'*60}\n"
        )

        if label == 0 and pred == 0:
            logs["benign_correct"].append(block)
        elif label == 0 and pred == 1:
            logs["benign_misclassified"].append(block)
        elif label == 1 and pred == 1:
            logs["malicious_correct"].append(block)
        elif label == 1 and pred == 0:
            logs["malicious_misclassified"].append(block)

    # 写入 txt 文件
    for key, entries in logs.items():
        file_path = os.path.join(output_dir, f"{key}.txt")
        with open(file_path, "w") as f:
            f.writelines(entries)

def evaluate_ae_model(model, dataloader, test_flow_indices_list, test_df, device, threshold):
    model.eval()
    all_errors = []
    all_preds = []
    all_labels = []

    all_flow_pred_labels = []
    all_flow_true_labels = []

    # 初始化每个 flow 的预测标签容器
    flow_pred_map = [-1] * len(test_df)

    with torch.no_grad():
        for batch in dataloader:
            x, y_true, flow_sample_index = batch
            x = x.to(device)
            y_true = y_true.to(device)

            reconstructed, _ = model(x)
            mse = ((reconstructed - x) ** 2).mean(dim=(1, 2))  # 每个样本的重建误差
            preds = (mse > threshold).long().cpu().numpy()

            all_errors.extend(mse.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(y_true.cpu().numpy())

            for pred_label, flow_sample_idx in zip(preds, flow_sample_index):
                flow_indices = test_flow_indices_list[flow_sample_idx]

                all_flow_pred_labels.extend([pred_label] * len(flow_indices))

                flow_true_labels = test_df.loc[flow_indices, "label"].apply(lambda x: 0 if x == "BENIGN" else 1).tolist()
                all_flow_true_labels.extend(flow_true_labels)

                for idx in flow_indices:
                    flow_pred_map[idx] = pred_label

    # 构造流级别 DataFrame
    flow_result_df = test_df.copy()
    flow_result_df["predicted_label"] = flow_pred_map
    flow_result_df["true_label"] = flow_result_df["label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    flow_misclassified_df = flow_result_df[flow_result_df["predicted_label"] != flow_result_df["true_label"]]

    # 样本级别指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_errors)  # 使用重建误差进行 AUC 计算
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Flow 级别指标
    flow_acc = accuracy_score(all_flow_true_labels, all_flow_pred_labels)
    flow_precision = precision_score(all_flow_true_labels, all_flow_pred_labels, zero_division=0)
    flow_recall = recall_score(all_flow_true_labels, all_flow_pred_labels, zero_division=0)
    flow_f1 = f1_score(all_flow_true_labels, all_flow_pred_labels, zero_division=0)
    flow_cm = confusion_matrix(all_flow_true_labels, all_flow_pred_labels)
    flow_tn, flow_fp, flow_fn, flow_tp = flow_cm.ravel()
    flow_fpr = flow_fp / (flow_fp + flow_tn) if (flow_fp + flow_tn) > 0 else 0.0

    metrics = {
        "sample_accuracy": acc,
        "sample_auc": auc,
        "sample_precision": precision,
        "sample_recall": recall,
        "sample_f1": f1,
        "sample_false_positive_rate": fpr,
        "flow_accuracy": flow_acc,
        "flow_precision": flow_precision,
        "flow_recall": flow_recall,
        "flow_f1": flow_f1,
        "flow_false_positive_rate": flow_fpr
    }

    return metrics, all_preds, all_errors, all_labels, all_flow_pred_labels, all_flow_true_labels, flow_result_df, flow_misclassified_df

def visualize_embeddings(model, text_prototype_projector, dataloader, device, epoch,
                         save_dir="./visualizations", use_umap=True,
                         label_type='class', fixed_prototypes=None):
    """
    可视化样本和 prototype 的嵌入。
    fixed_prototypes: [2, num_concepts, D] 文本原型嵌入（来自 SBERT）
    concept_manager: 提供 id_to_dict()
    sbert_model: sentence-transformers 模型（用于 encode 描述）
    """

    model.eval()
    text_prototype_projector.eval()
    
    all_features, all_class_labels, all_concept_ids = [], [], []

    # ========== 1. 收集样本的特征和标签 ==========
    with torch.no_grad():
        for batch in dataloader:
            batch_x, class_labels, concept_ids = batch
            batch_x = batch_x.to(device)
            features, _ = model(batch_x)
            features = F.normalize(features, dim=1)
            all_features.append(features.cpu())
            all_class_labels.append(class_labels)
            all_concept_ids.append(concept_ids)
            
        projected_prototypes = text_prototype_projector(fixed_prototypes)
        projected_prototypes = F.normalize(projected_prototypes, dim=2)  # [2, num_concepts, D]

    features_np = torch.cat(all_features, dim=0).numpy()
    class_np = torch.cat(all_class_labels, dim=0).numpy()
    concept_np = torch.cat(all_concept_ids, dim=0).numpy()

    # ========== 2. 构建可视化标签 ==========
    if label_type == "class":
        labels = class_np
    elif label_type == "concept":
        labels = concept_np
    elif label_type == "both":
        labels = [f"Cls{c}_Cpt{g}" for c, g in zip(class_np, concept_np)]
    else:
        raise ValueError("label_type must be one of: 'class', 'concept', 'both'")

    # ========== 3. 构建 prototype 子集 ==========
    prototype_points, prototype_labels = [], []
    used_keys = set(zip(class_np.tolist(), concept_np.tolist()))
    
    print(f"Number of unique (class, concept) pairs: {len(used_keys)}")
    
    for class_label, concept_id in used_keys:
        proto_vec = projected_prototypes[class_label, int(concept_id)].cpu().numpy()
        prototype_points.append(proto_vec)

        if label_type == "class":
            prototype_labels.append(class_label)
        elif label_type == "concept":
            prototype_labels.append(concept_id)
        elif label_type == "both":
            prototype_labels.append(f"Cls{class_label}_Cpt{concept_id}")
            

    # ========== 4. 降维 ==========
    all_points = np.concatenate([features_np] + ([np.stack(prototype_points)] if prototype_points else []), axis=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine') if use_umap \
        else TSNE(n_components=2, perplexity=30, n_iter=1000)
    embedding_2d = reducer.fit_transform(all_points)

    sample_embeds = embedding_2d[:len(features_np)]
    proto_embeds = embedding_2d[len(features_np):] if prototype_points else []

    # ========== 5. 绘图 ==========
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("hls", len(unique_labels))
    label_to_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    # 绘制样本点
    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        plt.scatter(sample_embeds[idxs, 0], sample_embeds[idxs, 1],
                    c=[label_to_color[lbl]] * len(idxs), label=str(lbl),
                    s=40, alpha=0.8, edgecolor='k', marker='o')

    # 绘制 prototype 点
    for i, proto_vec in enumerate(proto_embeds):
        lbl = prototype_labels[i]
        color = label_to_color.get(lbl, "black")  # fallback in case of mismatch
        
        # 背景（模拟边框）
        plt.scatter(proto_vec[0], proto_vec[1],
                    c="black", marker='x', s=200, linewidths=5, alpha=0.6)

        # 前景（实际颜色）
        plt.scatter(proto_vec[0], proto_vec[1],
                    c=[color], marker='x', s=100, linewidths=2.5,
                    label=f"{lbl}_proto" if label_type == 'both' else "prototype")
        
    
    # for i, proto_vec in enumerate(proto_embeds):
    #     print(f"Prototype {i} embed position: {proto_vec}")


    # 图例和保存
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), title=label_type, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Embedding Visualization ({'UMAP' if use_umap else 't-SNE'}) by {label_type}, Epoch {epoch}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"epoch_{epoch}_{label_type}_{'umap' if use_umap else 'tsne'}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def generate_concept_description(concept_dict, class_label):
    label_str = "benign" if class_label == 0 else "DDoS"
    rate_str = "high" if concept_dict["is high request rate"] else "low"
    activity_str = "continuous" if concept_dict["is continuous requesting"] else "intermittent"
    pattern_str = "fixed" if concept_dict["is fixed behavior pattern"] else "variable"

    return (
        f"This is a {label_str} sample that exhibits a {rate_str} request rate, "
        f"{activity_str} request activity, and {pattern_str} request behavior pattern."
    )
    
    # return (
    #     f"This is a {label_str} sample."
    # )


def generate_text_anchored_prototypes(concept_manager, device, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    为每个概念 + 类别组合生成文本描述，编码成 prototype 向量。
    - 模型名称可配置
    - 自动检测 embedding_dim 并返回
    """
    # 加载指定的 sentence-BERT 模型
    sentence_model = SentenceTransformer(model_name)

    # 先编码一个样例文本，获取输出维度
    dummy_desc = "This is a dummy sentence."
    with torch.no_grad():
        dummy_emb = sentence_model.encode(dummy_desc, convert_to_tensor=True).to(device)
    embedding_dim = dummy_emb.shape[0]

    # 初始化 prototype tensor
    prototype_tensor = torch.zeros((2, concept_manager.num_concepts(), embedding_dim), device=device)

    # 生成每个 (class_label, concept_id) 对应的文本 prototype 向量
    for concept_id in range(concept_manager.num_concepts()):
        for class_label in [0, 1]:
            concept_dict = concept_manager.id_to_dict(concept_id)
            desc = generate_concept_description(concept_dict, class_label)
            with torch.no_grad():
                emb = sentence_model.encode(desc, convert_to_tensor=True).to(device)
                prototype_tensor[class_label, concept_id] = emb

    return prototype_tensor, embedding_dim

import torch
from torch.nn.functional import cosine_similarity

def find_nearest_text_prototypes(all_ts_latent_vectors, final_projected_text_prototypes, device):
    """
    计算每个时间序列样本的隐向量与文本锚点之间的余弦相似度，并找到最接近的文本锚点的索引
    
    参数:
    all_ts_latent_vectors: 所有时间序列样本的隐向量列表，形状为[样本数, 512]
    final_projected_text_prototypes: 文本锚点的表征，形状为[2, 8, 512]
    device: 计算设备
    
    返回:
    nearest_indices: 每个样本最接近的文本锚点的索引列表，每个元素为二元组(i, j)
    max_similarities: 对应的最大余弦相似度值列表
    """
    # 将隐向量转换为PyTorch张量并移至指定设备
    all_ts_latent_vectors = torch.tensor(all_ts_latent_vectors, dtype=torch.float32).to(device)
    
    # 确保文本锚点在同一设备上
    final_projected_text_prototypes = final_projected_text_prototypes.to(device)
    
    # 初始化结果列表
    nearest_indices = []
    max_similarities = []
    
    # 对每个样本计算与所有文本锚点的余弦相似度
    for latent_vector in all_ts_latent_vectors:
        # 初始化最大相似度和对应的索引
        max_sim = -1
        best_i, best_j = -1, -1
        
        # 遍历所有文本锚点
        for i in range(final_projected_text_prototypes.shape[0]):
            for j in range(final_projected_text_prototypes.shape[1]):
                # 计算余弦相似度
                similarity = cosine_similarity(
                    latent_vector.unsqueeze(0),
                    final_projected_text_prototypes[i, j].unsqueeze(0)
                ).item()
                
                # 更新最大值和索引
                if similarity > max_sim:
                    max_sim = similarity
                    best_i, best_j = i, j
        
        # 保存结果
        nearest_indices.append((best_i, best_j))
        max_similarities.append(max_sim)
    
    return nearest_indices, max_similarities

def find_all_text_similarities(all_ts_latent_vectors, final_projected_text_prototypes, device):
    """
    计算每个时间序列样本与所有文本锚点的余弦相似度，并返回完整相似度信息
    
    参数:
    all_ts_latent_vectors: 所有时间序列样本的隐向量，形状为[样本数, 512]
    final_projected_text_prototypes: 文本锚点表示，形状为[2, 8, 512]
    device: 计算设备
    
    返回:
    all_similarities: 每个样本对应一个列表，列表中是所有文本锚点的 (i, j, similarity) 元组
    nearest_indices: 每个样本最相近的文本锚点索引 (i, j)
    max_similarities: 每个样本最大相似度值
    """
    all_ts_latent_vectors = torch.tensor(all_ts_latent_vectors, dtype=torch.float32).to(device)
    final_projected_text_prototypes = final_projected_text_prototypes.to(device)

    all_similarities = []
    nearest_indices = []
    max_similarities = []

    for latent_vector in all_ts_latent_vectors:
        similarities = []
        max_sim = -1
        best_i, best_j = -1, -1

        for i in range(final_projected_text_prototypes.shape[0]):
            for j in range(final_projected_text_prototypes.shape[1]):
                sim = cosine_similarity(
                    latent_vector.unsqueeze(0),
                    final_projected_text_prototypes[i, j].unsqueeze(0)
                ).item()
                
                similarities.append((i, j, sim))

                if sim > max_sim:
                    max_sim = sim
                    best_i, best_j = i, j

        all_similarities.append(similarities)
        nearest_indices.append((best_i, best_j))
        max_similarities.append(max_sim)

    return all_similarities, nearest_indices, max_similarities







