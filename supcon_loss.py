import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, concept_labels, class_labels):
        """
        Args:
            features: [B, D], L2-normalized inside
            concept_labels: [B]
            class_labels: [B]
        Returns:
            Scalar contrastive loss
        """
        device = features.device
        batch_size = features.size(0)

        # L2 normalize
        features = F.normalize(features, p=2, dim=1)  # [B, D]

        # Similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Positive pair mask: same concept & same class (excluding self)
        concept_mask = concept_labels.unsqueeze(0) == concept_labels.unsqueeze(1)  # [B, B]
        class_mask = class_labels.unsqueeze(0) == class_labels.unsqueeze(1)        # [B, B]
        positive_mask = (concept_mask & class_mask).fill_diagonal_(False)          # remove self

        # For numerical stability: set diagonal (self-sim) to large negative
        logits_mask = torch.ones_like(sim_matrix, device=device).fill_diagonal_(0)
        sim_matrix = sim_matrix.masked_fill(logits_mask == 0, -1e9)

        # Compute denominator: logsumexp over all valid entries
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)  # [B, B]

        # Compute loss for only anchors that have at least one positive
        positive_mask = positive_mask.float()
        positives_per_row = positive_mask.sum(dim=1)  # [B]

        valid_rows = positives_per_row > 0
        if valid_rows.sum() == 0:
            # No valid positive pairs in this batch → return 0 (or raise warning)
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / (positives_per_row + 1e-9)

        # Only average over anchors with at least one positive
        loss = -mean_log_prob_pos[valid_rows].mean()

        return loss
    
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_class=1.0, lambda_concept=1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_class = lambda_class
        self.lambda_concept = lambda_concept

    def forward(self, features, class_labels, concept_labels):
        """
        features: [B, D]
        class_labels: [B]
        concept_labels: [B]
        """
        # loss_class = self._prototype_contrastive_loss(features, class_labels)
        loss_concept = self._prototype_contrastive_loss(features, concept_labels)
        return loss_concept
    

    def _prototype_contrastive_loss(self, features, labels):
        """
        features: [B, D]
        labels: [B]
        """
        device = features.device
        features = F.normalize(features, dim=1)

        # 1. 获取 unique label 和其 prototype 索引
        unique_labels, inverse_indices = torch.unique(labels, sorted=True, return_inverse=True)
        num_prototypes = unique_labels.size(0)  # P

        # 2. Prototype matrix: [P, D]
        prototypes = torch.zeros(num_prototypes, features.size(1), device=device)
        prototypes.index_add_(0, inverse_indices, features)
        counts = torch.bincount(inverse_indices, minlength=num_prototypes).unsqueeze(1)
        prototypes = prototypes / (counts + 1e-9)

        # 3. 相似度矩阵 [B, P]
        logits = torch.matmul(features, prototypes.T) / self.temperature

        # 4. 正确原型的索引位置
        positive_indices = inverse_indices.unsqueeze(1)  # [B, 1]

        # 5. CrossEntropy Loss（等价于 InfoNCE）
        loss = F.cross_entropy(logits, inverse_indices)
        return loss


def compute_prototypes(model, dataloader, device, num_classes, num_concepts):
    model.eval()
    prototype_dict = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            x, class_labels, concept_ids = batch[:3]
            x = x.to(device)
            class_labels = class_labels.to(device)
            concept_ids = concept_ids.to(device)

            features,_ = model(x)  # [B, hidden_dim]
            for i in range(features.size(0)):
                key = (int(class_labels[i]), int(concept_ids[i]))
                prototype_dict[key].append(features[i].detach().cpu())

    # Compute mean prototype for each (class, concept)
    feature_dim = next(iter(prototype_dict.values()))[0].size(0)
    num_prototypes = num_classes * num_concepts
    prototypes = torch.zeros(num_prototypes, feature_dim)

    for cls in range(num_classes):
        for cid in range(num_concepts):
            key = (cls, cid)
            vecs = prototype_dict[key]
            if len(vecs) > 0:
                prototypes[cls * num_concepts + cid] = torch.stack(vecs).mean(dim=0)
            else:
                print(f"[WARN] No samples for prototype ({cls}, {cid})")

    return F.normalize(prototypes, dim=1)  # 归一化

def prototype_contrastive_loss(features, class_labels, concept_ids, prototypes, temperature=0.1):
    """
    features: [B, D]
    class_labels: [B]
    concept_ids: [B]
    prototypes: [2, num_concepts, D]
    """
    device = features.device
    B, D = features.size()
    _, num_concepts, _ = prototypes.shape
    
    # features = F.normalize(features, dim=1)  # [B, D]
    # prototypes = F.normalize(prototypes, dim=2)  # [2, num_concepts, D]

    # 构造 batch 中用到的 (class_label, concept_id) 对
    keys = [(int(c.item()), int(g.item())) for c, g in zip(class_labels, concept_ids)]

    # 找出唯一的 (class_label, concept_id)
    unique_keys = sorted(set(keys))

    # 建立 (class_label, concept_id) → 索引 映射
    key_to_local_idx = {key: i for i, key in enumerate(unique_keys)}

    # 构造当前 batch 实际使用到的 prototype 子集
    local_prototypes = torch.stack([
        prototypes[class_label, concept_id] for (class_label, concept_id) in unique_keys
    ], dim=0).to(device)  # [P_used, D]

    # 为每个样本找到其对应的 prototype 索引
    local_proto_indices = torch.tensor(
        [key_to_local_idx[key] for key in keys], dtype=torch.long, device=device
    )  # [B]

    # 相似度计算（点积 + 温度缩放）
    sim_matrix = torch.mm(features, local_prototypes.T) / temperature  # [B, P_used]

    # 对比损失（交叉熵）
    loss = F.cross_entropy(sim_matrix, local_proto_indices)
    return loss

def prototype_contrastive_loss_v2(features, class_labels, concept_ids, prototypes, temperature=0.1):
    """
    features: [B, D]
    class_labels: [B]
    concept_ids: [B]
    prototypes: [2, num_concepts, D] — all available prototypes
    """
    device = features.device
    B, D = features.size()
    num_classes, num_concepts, _ = prototypes.size()  # typically [2, num_concepts]

    # Flatten all prototypes: [2 * num_concepts, D]
    all_prototypes = prototypes.view(-1, D).to(device)  # [P_total, D]

    # Compute similarity: dot product + temperature scaling
    logits = torch.mm(features, all_prototypes.T) / temperature  # [B, P_total]

    # Compute target indices: where the positive prototype is located in all_prototypes
    # index = class_label * num_concepts + concept_id
    positive_indices = (class_labels * num_concepts + concept_ids).to(device)  # [B]

    # Cross-entropy loss: log-softmax over all prototypes, target is positive_indices
    loss = F.cross_entropy(logits, positive_indices)

    return loss



