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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dataset_from_csv(csv_file_name):
    """
    从指定文件夹中读取训练集和测试集CSV，转换时间戳为相对时间。

    参数:
    folder_path (str): 包含train.csv和test.csv的文件夹路径

    返回:
    tuple: (训练集DataFrame, 测试集DataFrame, 训练集所有原始数据（含恶意）DataFrame)
    """
    dataset_path = csv_file_name
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test CSV not found: {dataset_path}")

    # 读取 CSV 文件
    dataset_df = pd.read_csv(dataset_path)

    return dataset_df


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


def get_training_behavior_sequence_samples(flow_df, time_slot_interval, behavior_time_window_len, stride):
    ip_pair_info = build_ip_pair_behavior_sequences_fast(flow_df, time_slot_interval)
    all_samples = []
    samples_number = 0

    for ip_pair, info in ip_pair_info.items():
        behavior_seq = info["behavior_sequence"]
        
        # 构造映射用于后面按照time_slot_index索引
        time_slot_to_behavior = {b["time_slot_index"]: b for b in behavior_seq}
        time_slots = sorted(time_slot_to_behavior.keys())

        for start in range(0, len(time_slots), stride):
            begin_time_slot = time_slots[start]
            sample_behaviors = []
            sample_flow_indices = []

            for offset in range(behavior_time_window_len):
                current_slot = begin_time_slot + offset
                if current_slot in time_slot_to_behavior:
                    # b = time_slot_to_behavior[current_slot]
                    
                    # 修改 time_slot_index 为相对值
                    b = time_slot_to_behavior[current_slot].copy()
                    b["time_slot_index"] -= begin_time_slot
                    
                    sample_behaviors.append(b)
                    sample_flow_indices.extend(b["flow_indices"])
            
            clean_sequence = [{k: b[k] for k in feature_names} for b in sample_behaviors]
            
            samples_number += 1
            
            all_samples.append({
                "sample_idx": samples_number,
                "ip_pair": ip_pair,
                "start_time_slot_index": begin_time_slot,
                "behaviour_sequence": clean_sequence,
                "flow_indices": sample_flow_indices,
                "label": info["ip_pair_label"]
            })

    return all_samples


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


def generate_samples(behavior_sequence_samples, behavior_time_window_len):
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
        concept_label = torch.tensor(self.concept_labels[idx], dtype=torch.float32)
        return input_sample, training_label, concept_label
    
class TestBehaviorSequenceDataset(Dataset):
    def __init__(self, input_samples, training_labels, flow_sample_index_list):
        self.input_samples = input_samples
        self.training_labels = training_labels
        self.flow_sample_index_list = flow_sample_index_list

    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        input_sample = torch.tensor(self.input_samples[idx], dtype=torch.float32)
        training_label = torch.tensor(self.training_labels[idx], dtype=torch.long)
        flow_index_id = self.flow_sample_index_list[idx]
        return input_sample, training_label, flow_index_id

def evaluate_model(model, dataloader, test_flow_indices_list, test_df, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_concepts = []

    all_flow_pred_labels = []
    all_flow_true_labels = []

    # 新增：为 test_df 中每个 flow 初始化预测值容器（index -> label）
    flow_pred_map = [-1] * len(test_df)

    with torch.no_grad():
        for batch in dataloader:
            x, y_true, flow_sample_index = batch
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred, concepts = model(x)
            y_prob = y_pred.squeeze().cpu().numpy()
            y_pred_label = (y_pred.squeeze() > 0.5).long().cpu().numpy()

            all_probs.extend(y_prob)
            all_preds.extend(y_pred_label)
            all_labels.extend(y_true.cpu().numpy())
            all_concepts.append(concepts.cpu())

            for pred_label, flow_sample_idx in zip(y_pred_label, flow_sample_index):
                flow_indices = test_flow_indices_list[flow_sample_idx]  # list of int

                # Flow-level 结果扩展
                all_flow_pred_labels.extend([pred_label] * len(flow_indices))

                # 对应 true label
                flow_true_labels = test_df.loc[flow_indices, "label"].apply(lambda x: 0 if x == "BENIGN" else 1).tolist()
                all_flow_true_labels.extend(flow_true_labels)

                # 填入 flow_pred_map 中（按原始 index 对应）
                for idx in flow_indices:
                    flow_pred_map[idx] = pred_label

    # 构造完整的 flow-level prediction DataFrame
    flow_result_df = test_df.copy()
    flow_result_df["predicted_label"] = flow_pred_map
    flow_result_df["true_label"] = flow_result_df["label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    
    # 提取预测错误的行
    flow_misclassified_df = flow_result_df[flow_result_df["predicted_label"] != flow_result_df["true_label"]]

    # 样本级别指标
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
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

    all_concepts = torch.cat(all_concepts, dim=0)

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

    return metrics, all_preds, all_probs, all_labels, all_concepts, all_flow_pred_labels, all_flow_true_labels, flow_result_df, flow_misclassified_df

def save_debug_logs(
    all_preds, all_labels, all_concepts, flow_sample_index_list,
    test_ip_pair_list, test_start_time_slot_index_list, test_samples,
    concept_manager, output_dir="debug_logs_txt"
):
    import os

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



