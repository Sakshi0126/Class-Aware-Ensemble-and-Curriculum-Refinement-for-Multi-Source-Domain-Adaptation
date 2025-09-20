# =============================================================
# STEP-BY-STEP COMMENTARY + IMPLEMENTATION OF OT-BASED Lpsd + ALIGNMENT
# =============================================================

import torch
import torch.nn.functional as F

# -------------------------
# STEP 1: OT-based Lpsd Loss (Replacing conf_reg_cls)
# -------------------------

def ot_reg_cls(pred1, pred2, avg_preds, class_heads, w2v_embeddings, threshold=0.9, alpha=0.5):
    """
    OT-based semantic pseudo-labeling loss (OTMatch-style replacement for L_psd).
    
    Args:
        pred1: [B, K] - teacher softmax prediction (e.g., esm_tar_preds)
        pred2: [B, K] - student softmax prediction (e.g., esm_aug_tar_preds)
        avg_preds: [B, K] - ensemble logits for pseudo-labels + confidence filtering
        class_heads: [K, D1] - classifier head weights (learned)
        w2v_embeddings: [K, D2] - Word2Vec embeddings (fixed)
        threshold: float - confidence threshold
        alpha: float - blend weight between classifier heads and W2V

    Returns:
        Scalar OT-based consistency loss
    """
    B, K = pred1.size()
    device = pred1.device

    # Confidence filtering
    conf, pseudo_labels = avg_preds.max(dim=1)
    mask = conf >= threshold
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    selected = mask.nonzero(as_tuple=True)[0]

    # Compute blended semantic cost matrix
    fc_heads = F.normalize(class_heads, dim=1)
    w2v_heads = F.normalize(w2v_embeddings, dim=1)
    sim_fc = torch.matmul(fc_heads, fc_heads.T)
    sim_w2v = torch.matmul(w2v_heads, w2v_heads.T)
    cost_matrix = 1.0 - (alpha * sim_fc + (1 - alpha) * sim_w2v)

    # Compute semantic penalty for selected samples
    loss = 0.0
    for i in selected:
        y_hat = pseudo_labels[i].item()
        cost_vec = cost_matrix[y_hat].to(device)
        loss += torch.dot(pred2[i], cost_vec)

    return loss / len(selected)

# -------------------------
# STEP 2: OT-based Alignment Regularizer (added to L_align)
# -------------------------

def align_cost_regularizer(pred_tar, pseudo_labels, cost_matrix):
    """
    Semantic regularizer for alignment loss (penalizes target predictions far from pseudo-labels).

    Args:
        pred_tar: [B, K] - softmax output of domain classifier (GRL) on target
        pseudo_labels: [B] - hard pseudo-labels from teacher
        cost_matrix: [K, K] - precomputed class-aware cost matrix

    Returns:
        Scalar regularization loss
    """
    B, K = pred_tar.size()
    device = pred_tar.device

    loss = 0.0
    for i in range(B):
        y_hat = pseudo_labels[i].item()
        loss += torch.dot(pred_tar[i], cost_matrix[y_hat].to(device))

    return loss / B

# -------------------------
# STEP 3: CSR-style Alignment Loss with OT Regularizer
# -------------------------

tar_alignment_loss = torch.zeros(1).cuda()
for ii in range(0, len(args.src)):
    # Original CSR source-side alignment
    t_align_loss1 = 3 * F.cross_entropy(
        specific_src_advs[ii],
        main_src_preds[ii][ii].data.max(1)[1],
        reduction='none').mean()

    # Original CSR target-side alignment (with GRL)
    t_align_loss2 = 0.0
    if args.consis_tar:
        t_align_loss2 = F.nll_loss(
            shift_log(1. - F.softmax(specific_tar_advs[ii], dim=1)),
            esm_tar_preds.data.max(1)[1], reduction='none').mean()

    tar_alignment_loss += t_align_loss1 + t_align_loss2

    # OT-based semantic regularization on target alignment
    if args.use_align_cost_reg:
        align_probs = F.softmax(specific_tar_advs[ii], dim=1)
        pseudo_labels = esm_tar_preds.data.max(1)[1]
        align_cost = align_cost_regularizer(
            align_probs, pseudo_labels, cost_matrix
        )
        tar_alignment_loss += args.align_cost_weight * align_cost
