#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

import config_asse_trace_v3 as C
from stage_a_asse_trace import (
    MechanismDiscovery,
    split_path,
    load_json,
)


# =========================
# Utils
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cfg(name, default=None):
    return getattr(C, name, default)


def ensure_config_defaults():
    defaults = {
        "TRACE_USE_RAW_PROJ": True,
        "TRACE_RAW_DIM": 256,
        "TRACE_USE_MECH_Z": True,
        "TRACE_USE_MECH_GATES": True,
        "TRACE_USE_MECH_SCORES": True,
        "TRACE_USE_DELTA_FEATURES": True,

        "TRACE_FREEZE_STAGE_A": True,

        "TRACE_GRU_HIDDEN": 256,
        "TRACE_GRU_LAYERS": 1,
        "TRACE_BIDIRECTIONAL": False,

        "TRACE_MIL_TOPK": 3,

        "LAMBDA_STATE_FINAL": 1.0,
        "LAMBDA_EVIDENCE_MIL": 0.3,

        "USE_PREFIX_LOSS": True,
        "LAMBDA_PREFIX": 0.2,
        "PREFIX_LOSS_RHO": 0.2,
        "PREFIX_LOSS_GAMMA": 2.0,

        "USE_PREFIX_RANK_LOSS": True,
        "LAMBDA_PREFIX_RANK": 0.05,
        "PREFIX_RANK_EARLY_FRAC": 0.4,
        "PREFIX_RANK_LATE_FRAC": 0.8,
        "PREFIX_RANK_MARGIN": 0.2,

        "USE_PREFIX_STABLE_LOSS": False,
        "LAMBDA_PREFIX_STABLE": 0.005,
        "PREFIX_STABLE_MARGIN": 0.1,

        "LAMBDA_EVIDENCE_SPARSE": 0.0,
        "LAMBDA_EVIDENCE_SMOOTH": 0.0,

        "EPOCHS_STAGE_B": 30,
        "LR_STAGE_B": 5e-4,
        "WEIGHT_DECAY": 1e-4,
        "DROPOUT": 0.1,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "EPS": 1e-8,

        "DROP_NON_CONSTANT_LABEL_TRAJS": True,

        "EARLY_AWARE_SELECTION": True,
        "SELECT_WEIGHT_FULL": 0.25,
        "SELECT_WEIGHT_PREFIX40": 0.25,
        "SELECT_WEIGHT_PREFIX60": 0.25,
        "SELECT_WEIGHT_PREFIX80": 0.25,
    }

    for k, v in defaults.items():
        if not hasattr(C, k):
            setattr(C, k, v)


def last_valid_logits(seq_logits, mask):
    """
    seq_logits: [B, T]
    mask: [B, T]
    return: [B]
    """
    lengths = mask.sum(dim=1).long().clamp_min(1)
    idx = lengths - 1
    return seq_logits[
        torch.arange(seq_logits.size(0), device=seq_logits.device),
        idx,
    ]


def topk_mean_mil_logits(step_logits, mask, k):
    """
    step_logits: [B, T]
    mask: [B, T]
    return: [B]
    """
    step_logits = step_logits.masked_fill(mask == 0, -1e9)

    T = step_logits.size(1)
    k = min(k, T)

    topk_values, _ = torch.topk(step_logits, k=k, dim=1)

    valid_topk = topk_values > -1e8
    topk_values = torch.where(
        valid_topk,
        topk_values,
        torch.zeros_like(topk_values),
    )

    denom = valid_topk.float().sum(dim=1).clamp_min(1.0)
    return topk_values.sum(dim=1) / denom


def asymmetric_prefix_loss_on_state(
    state_logits,
    mask,
    traj_y,
    rho=0.2,
    gamma=2.0,
):
    """
    Prefix-aware loss on cumulative trajectory state q_t.

    Safe trajectories:
      all prefixes are supervised as safe.

    Unsafe trajectories:
      early prefixes before rho are not strongly forced to unsafe;
      after rho, positive supervision weight gradually increases.
    """
    B, T = state_logits.shape
    device = state_logits.device

    valid_len = mask.sum(dim=1).clamp_min(1).float()

    pos = torch.arange(1, T + 1, device=device).float().unsqueeze(0)
    rel = pos / valid_len.unsqueeze(1)

    valid = mask.bool()
    y_expand = traj_y.unsqueeze(1).expand_as(state_logits)

    bce = F.binary_cross_entropy_with_logits(
        state_logits,
        y_expand,
        reduction="none",
    )

    safe_mask = (traj_y.unsqueeze(1) < 0.5) & valid
    unsafe_mask = (traj_y.unsqueeze(1) > 0.5) & valid

    if safe_mask.any():
        safe_loss = bce[safe_mask].mean()
    else:
        safe_loss = torch.tensor(0.0, device=device)

    rel_after = ((rel - rho) / max(1e-6, 1.0 - rho)).clamp(0.0, 1.0)
    weights = rel_after ** gamma

    unsafe_weights = weights * unsafe_mask.float()

    if unsafe_weights.sum() > 0:
        unsafe_loss = (bce * unsafe_weights).sum() / unsafe_weights.sum().clamp_min(1.0)
    else:
        unsafe_loss = torch.tensor(0.0, device=device)

    return safe_loss + unsafe_loss


def unsafe_prefix_ranking_loss_on_state(
    state_logits,
    mask,
    traj_y,
    early_frac=0.4,
    late_frac=0.8,
    margin=0.2,
):
    """
    For unsafe trajectories, encourage cumulative risk state at later prefix
    to be higher than at earlier prefix.
    """
    B, T = state_logits.shape
    device = state_logits.device

    lengths = mask.sum(dim=1).long()
    losses = []

    for i in range(B):
        if traj_y[i].item() < 0.5:
            continue

        L = int(lengths[i].item())
        if L <= 1:
            continue

        early_t = max(1, int(np.ceil(L * early_frac))) - 1
        late_t = max(1, int(np.ceil(L * late_frac))) - 1

        early_t = min(early_t, L - 1)
        late_t = min(late_t, L - 1)

        s_early = state_logits[i, early_t]
        s_late = state_logits[i, late_t]

        losses.append(F.relu(margin - s_late + s_early))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def prefix_stability_loss_on_state(
    state_logits,
    mask,
    margin=0.1,
):
    """
    Penalize large downward jumps in cumulative trajectory risk state.
    """
    if state_logits.size(1) <= 1:
        return torch.tensor(0.0, device=state_logits.device)

    drop = state_logits[:, :-1] - state_logits[:, 1:]
    valid = (mask[:, :-1] == 1) & (mask[:, 1:] == 1)

    loss = F.relu(drop - margin)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=state_logits.device)

    return loss[valid].mean()


def evidence_sparsity_loss(evidence_scores, mask, topk=None):
    """
    Optional: penalize non-top-k local evidence scores.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=evidence_scores.device)

    if topk is None:
        return (evidence_scores * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    scores_masked = evidence_scores.masked_fill(mask == 0, -1e9)
    k = min(topk, scores_masked.size(1))

    _, idx = torch.topk(scores_masked, k=k, dim=1)

    topk_mask = torch.zeros_like(evidence_scores, dtype=torch.bool)
    topk_mask.scatter_(1, idx, True)

    non_topk_mask = (mask == 1) & (~topk_mask)

    if non_topk_mask.sum() == 0:
        return torch.tensor(0.0, device=evidence_scores.device)

    return evidence_scores[non_topk_mask].mean()


def evidence_smooth_loss(evidence_scores, mask):
    """
    Optional: smooth local evidence scores.
    """
    if evidence_scores.size(1) <= 1:
        return torch.tensor(0.0, device=evidence_scores.device)

    diff = torch.abs(evidence_scores[:, 1:] - evidence_scores[:, :-1])
    valid = (mask[:, 1:] == 1) & (mask[:, :-1] == 1)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=evidence_scores.device)

    return diff[valid].mean()


def slice_prefix_batch(x, mask, frac):
    """
    Actually truncate input to first frac of valid steps.
    """
    B, T, D = x.shape
    lengths = mask.sum(dim=1).long()

    prefix_lengths = []

    for i in range(B):
        L = int(lengths[i].item())
        pL = max(1, int(np.ceil(L * frac)))
        prefix_lengths.append(pL)

    max_prefix_T = max(prefix_lengths)

    x_prefix = x[:, :max_prefix_T, :]
    mask_prefix = torch.zeros(
        B,
        max_prefix_T,
        dtype=mask.dtype,
        device=mask.device,
    )

    for i, pL in enumerate(prefix_lengths):
        mask_prefix[i, :pL] = 1

    return x_prefix, mask_prefix


def add_delta_features(feat, mask):
    """
    feat: [B, T, D]
    mask: [B, T]

    returns:
      concat([feat_t, delta_feat_t])
    where delta_feat_0 = 0 and delta_feat_t = feat_t - feat_{t-1}.
    """
    if feat.size(1) <= 1:
        delta = torch.zeros_like(feat)
        return torch.cat([feat, delta], dim=-1)

    delta = feat[:, 1:, :] - feat[:, :-1, :]
    zero = torch.zeros_like(delta[:, :1, :])
    delta = torch.cat([zero, delta], dim=1)

    valid_pair = (mask == 1).float().unsqueeze(-1)
    delta = delta * valid_pair

    return torch.cat([feat, delta], dim=-1)


# =========================
# Dataset
# =========================

class TrajExample:
    def __init__(self, traj_id, x_seq, traj_y, event_indices):
        self.traj_id = int(traj_id)
        self.x_seq = x_seq.astype(np.float32)
        self.traj_y = int(traj_y)
        self.event_indices = event_indices.astype(np.int64)


def build_examples(step_df):
    examples = []

    for tid, g in step_df.groupby("traj_id", sort=False):
        g = g.sort_values("event_order")

        labels = g["label"].astype(int).values
        if len(set(labels.tolist())) != 1:
            raise RuntimeError(
                f"traj_id={tid} has non-constant labels: {set(labels.tolist())}. "
                f"Make sure label is trajectory-level and broadcast to all steps."
            )

        traj_y = int(labels[0])
        x_seq = np.stack(g["x"].values).astype(np.float32)
        event_indices = g["event_idx"].astype(int).values

        examples.append(
            TrajExample(
                traj_id=int(tid),
                x_seq=x_seq,
                traj_y=traj_y,
                event_indices=event_indices,
            )
        )

    return examples


class TrajDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    B = len(batch)
    max_T = max(ex.x_seq.shape[0] for ex in batch)
    D = batch[0].x_seq.shape[1]

    x = torch.zeros(B, max_T, D, dtype=torch.float32)
    mask = torch.zeros(B, max_T, dtype=torch.long)
    traj_y = torch.zeros(B, dtype=torch.float32)

    traj_ids = []
    event_indices = []

    for i, ex in enumerate(batch):
        T = ex.x_seq.shape[0]

        x[i, :T] = torch.tensor(ex.x_seq, dtype=torch.float32)
        mask[i, :T] = 1
        traj_y[i] = float(ex.traj_y)

        traj_ids.append(ex.traj_id)

        padded_events = np.full(max_T, -1, dtype=np.int64)
        padded_events[:T] = ex.event_indices
        event_indices.append(padded_events)

    event_indices = torch.tensor(np.stack(event_indices), dtype=torch.long)

    return {
        "x": x,
        "mask": mask,
        "traj_y": traj_y,
        "traj_ids": traj_ids,
        "event_indices": event_indices,
    }


# =========================
# Model
# =========================

class FrozenMechanismProvider(nn.Module):
    def __init__(self, ckpt, freeze=True):
        super().__init__()

        input_dim = int(ckpt["input_dim"])

        self.model = MechanismDiscovery(input_dim)
        self.model.load_state_dict(ckpt["model_state"])

        self.freeze = freeze

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x_flat):
        if self.freeze:
            with torch.no_grad():
                out = self.model(x_flat)
        else:
            out = self.model(x_flat)

        return out["z"], out["gates"], out["scores"]


class TrajectoryStateAuditor(nn.Module):
    """
    Stage B V3.

    Main idea:
      - local evidence head e_t: detects local risk evidence
      - cumulative state head q_t: estimates P(trajectory unsafe | prefix 1..t)
      - delta features explicitly represent trajectory transitions
    """

    def __init__(self, input_dim, stage_a_ckpt):
        super().__init__()

        self.input_dim = input_dim

        self.use_raw = cfg("TRACE_USE_RAW_PROJ", True)
        self.use_z = cfg("TRACE_USE_MECH_Z", True)
        self.use_gates = cfg("TRACE_USE_MECH_GATES", True)
        self.use_scores = cfg("TRACE_USE_MECH_SCORES", True)
        self.use_delta = cfg("TRACE_USE_DELTA_FEATURES", True)

        self.mech_provider = FrozenMechanismProvider(
            stage_a_ckpt,
            freeze=cfg("TRACE_FREEZE_STAGE_A", True),
        )

        base_dim = 0

        if self.use_raw:
            self.raw_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, C.TRACE_RAW_DIM),
                nn.GELU(),
                nn.Dropout(C.DROPOUT),
                nn.Linear(C.TRACE_RAW_DIM, C.TRACE_RAW_DIM),
            )
            base_dim += C.TRACE_RAW_DIM

        if self.use_z:
            base_dim += C.Z_DIM

        if self.use_gates:
            base_dim += C.K_MECH

        if self.use_scores:
            base_dim += C.K_MECH

        if base_dim <= 0:
            raise ValueError("No input features enabled.")

        feat_dim = base_dim * (2 if self.use_delta else 1)

        self.input_ln = nn.LayerNorm(feat_dim)

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=C.TRACE_GRU_HIDDEN,
            num_layers=C.TRACE_GRU_LAYERS,
            dropout=C.DROPOUT if C.TRACE_GRU_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=C.TRACE_BIDIRECTIONAL,
        )

        gru_out_dim = C.TRACE_GRU_HIDDEN * (
            2 if C.TRACE_BIDIRECTIONAL else 1
        )

        self.local_evidence_head = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Linear(gru_out_dim, C.TRACE_GRU_HIDDEN),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.TRACE_GRU_HIDDEN, 1),
        )

        self.state_risk_head = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Linear(gru_out_dim, C.TRACE_GRU_HIDDEN),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.TRACE_GRU_HIDDEN, 1),
        )

    def forward(self, x, mask):
        """
        x: [B, T, input_dim]
        mask: [B, T]
        """
        B, T, D = x.shape

        x_flat = x.reshape(B * T, D)

        z_flat, gates_flat, scores_flat = self.mech_provider(x_flat)

        z = z_flat.reshape(B, T, C.Z_DIM)
        gates = gates_flat.reshape(B, T, C.K_MECH)
        scores = scores_flat.reshape(B, T, C.K_MECH)

        base_features = []

        if self.use_raw:
            raw_feat = self.raw_proj(x)
            base_features.append(raw_feat)

        if self.use_z:
            base_features.append(z)

        if self.use_gates:
            base_features.append(gates)

        if self.use_scores:
            base_features.append(scores)

        feat = torch.cat(base_features, dim=-1)

        if self.use_delta:
            feat = add_delta_features(feat, mask)

        feat = self.input_ln(feat)

        g, _ = self.gru(feat)

        evidence_logits = self.local_evidence_head(g).squeeze(-1)
        state_logits = self.state_risk_head(g).squeeze(-1)

        evidence_mil_logit = topk_mean_mil_logits(
            step_logits=evidence_logits,
            mask=mask,
            k=C.TRACE_MIL_TOPK,
        )

        final_state_logit = last_valid_logits(
            state_logits,
            mask,
        )

        traj_logit = final_state_logit
        traj_score = torch.sigmoid(traj_logit)

        return {
            "traj_logit": traj_logit,
            "traj_score": traj_score,

            "state_logits": state_logits,
            "state_scores": torch.sigmoid(state_logits),

            "evidence_logits": evidence_logits,
            "evidence_scores": torch.sigmoid(evidence_logits),
            "evidence_mil_logit": evidence_mil_logit,
            "evidence_mil_score": torch.sigmoid(evidence_mil_logit),

            # compatibility names
            "step_logits": state_logits,
            "step_scores": torch.sigmoid(state_logits),

            "local_evidence_step_logits": evidence_logits,
            "local_evidence_step_scores": torch.sigmoid(evidence_logits),

            "g": g,
            "z": z,
            "gates": gates,
            "scores": scores,
            "features": feat,
        }


# =========================
# Evaluation
# =========================

def find_best_threshold(y_true, y_prob, metric="acc"):
    thresholds = np.linspace(0.05, 0.95, 181)

    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        pred = (y_prob >= t).astype(int)

        if metric == "acc":
            score = accuracy_score(y_true, pred)
        elif metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(metric)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()

    all_y = []
    all_p = []

    rows = []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["traj_y"].to(device)

        out = model(x, mask)

        p = out["traj_score"].detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        all_y.extend(y_np.tolist())
        all_p.extend(p.tolist())

        state_scores = out["state_scores"].detach().cpu().numpy()
        evidence_scores = out["evidence_scores"].detach().cpu().numpy()

        mask_np = batch["mask"].numpy()
        event_indices = batch["event_indices"].numpy()

        for i, tid in enumerate(batch["traj_ids"]):
            T = int(mask_np[i].sum())

            for t in range(T):
                rows.append({
                    "traj_id": int(tid),
                    "event_idx": int(event_indices[i, t]),
                    "t": int(t),
                    "T": int(T),
                    "relative_t": float((t + 1) / T),
                    "traj_y": int(y_np[i]),
                    "traj_score": float(p[i]),
                    "state_risk_score": float(state_scores[i, t]),
                    "local_evidence_score": float(evidence_scores[i, t]),

                    # compatibility name for old evaluator
                    "step_risk_score": float(state_scores[i, t]),
                })

    return np.array(all_y), np.array(all_p), pd.DataFrame(rows)


@torch.no_grad()
def evaluate(model, loader, device):
    y, p, step_df = collect_outputs(model, loader, device)

    pred = (p >= 0.5).astype(int)

    out = {
        "acc_at_0.5": float(accuracy_score(y, pred)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "pos_rate_true": float(y.mean()),
        "pos_rate_pred_0.5": float(pred.mean()),
        "mean_score": float(p.mean()),
    }

    best_t_acc, best_acc = find_best_threshold(y, p, metric="acc")
    best_t_f1, best_f1 = find_best_threshold(y, p, metric="f1")

    out["best_acc_threshold"] = float(best_t_acc)
    out["best_acc"] = float(best_acc)
    out["best_f1_threshold"] = float(best_t_f1)
    out["best_f1"] = float(best_f1)

    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
        out["main_score"] = out["auroc"]
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
        out["main_score"] = out["best_acc"]

    return out, step_df


@torch.no_grad()
def evaluate_prefix(model, loader, device, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    model.eval()

    rows = []

    for frac in fractions:
        all_y = []
        all_p = []

        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["traj_y"].to(device)

            x_prefix, mask_prefix = slice_prefix_batch(x, mask, frac)

            out = model(x_prefix, mask_prefix)

            all_y.extend(y.detach().cpu().numpy().tolist())
            all_p.extend(out["traj_score"].detach().cpu().numpy().tolist())

        y_np = np.array(all_y)
        p_np = np.array(all_p)
        pred = (p_np >= 0.5).astype(int)

        row = {
            "prefix_frac": frac,
            "acc_at_0.5": float(accuracy_score(y_np, pred)),
            "f1_at_0.5": float(f1_score(y_np, pred, zero_division=0)),
            "pos_rate_true": float(y_np.mean()),
            "pos_rate_pred_0.5": float(pred.mean()),
            "mean_score": float(p_np.mean()),
        }

        best_t_acc, best_acc = find_best_threshold(y_np, p_np, metric="acc")
        best_t_f1, best_f1 = find_best_threshold(y_np, p_np, metric="f1")

        row["best_acc_threshold"] = float(best_t_acc)
        row["best_acc"] = float(best_acc)
        row["best_f1_threshold"] = float(best_t_f1)
        row["best_f1"] = float(best_f1)

        if len(np.unique(y_np)) > 1:
            row["auroc"] = float(roc_auc_score(y_np, p_np))
            row["ap"] = float(average_precision_score(y_np, p_np))
        else:
            row["auroc"] = float("nan")
            row["ap"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def compute_pos_weight(examples, device):
    ys = np.array([ex.traj_y for ex in examples])
    num_pos = float((ys == 1).sum())
    num_neg = float((ys == 0).sum())

    if num_pos <= 0:
        return None

    return torch.tensor(
        [num_neg / max(num_pos, 1.0)],
        dtype=torch.float32,
        device=device,
    )


def early_aware_selection_score(val_metrics, prefix_df):
    """
    Select checkpoint based on both full and prefix performance.
    """
    full = float(val_metrics.get("auroc", np.nan))

    def get_prefix_auc(frac):
        rows = prefix_df[prefix_df["prefix_frac"] == frac]
        if len(rows) == 0:
            return np.nan
        return float(rows["auroc"].iloc[0])

    p40 = get_prefix_auc(0.4)
    p60 = get_prefix_auc(0.6)
    p80 = get_prefix_auc(0.8)

    vals = []
    weights = []

    if not np.isnan(full):
        vals.append(full)
        weights.append(C.SELECT_WEIGHT_FULL)

    if not np.isnan(p40):
        vals.append(p40)
        weights.append(C.SELECT_WEIGHT_PREFIX40)

    if not np.isnan(p60):
        vals.append(p60)
        weights.append(C.SELECT_WEIGHT_PREFIX60)

    if not np.isnan(p80):
        vals.append(p80)
        weights.append(C.SELECT_WEIGHT_PREFIX80)

    if len(vals) == 0:
        return val_metrics["main_score"]

    vals = np.array(vals)
    weights = np.array(weights)
    weights = weights / weights.sum()

    return float((vals * weights).sum())


def handle_non_constant_labels(step_df):
    """
    For quick ASSE runs:
      drop trajectories whose step labels are not constant.

    For formal experiments, better fix Stage A label broadcast
    so every step uses the trajectory-level label.
    """
    label_nunique = step_df.groupby("traj_id")["label"].nunique()
    bad = label_nunique[label_nunique != 1]

    if len(bad) == 0:
        return step_df

    bad_ids = set(map(int, bad.index.tolist()))

    if not C.DROP_NON_CONSTANT_LABEL_TRAJS:
        raise RuntimeError(
            f"Found {len(bad_ids)} trajectories with non-constant labels. "
            f"Set DROP_NON_CONSTANT_LABEL_TRAJS=True to drop them, or fix Stage A label broadcast."
        )

    print(
        f"[warning] Found {len(bad_ids)} trajectories with non-constant labels. "
        f"Dropping them from Stage B."
    )

    n_steps_before = len(step_df)
    n_trajs_before = step_df["traj_id"].nunique()

    step_df = step_df[~step_df["traj_id"].astype(int).isin(bad_ids)].copy()

    n_steps_after = len(step_df)
    n_trajs_after = step_df["traj_id"].nunique()

    print(
        f"[drop] trajs: {n_trajs_before} -> {n_trajs_after}, "
        f"steps: {n_steps_before} -> {n_steps_after}"
    )

    label_nunique_after = step_df.groupby("traj_id")["label"].nunique()
    assert (label_nunique_after == 1).all()

    return step_df


# =========================
# Training
# =========================

def train(args):
    ensure_config_defaults()
    set_seed(C.SEED)
    C.ensure_dir(C.OUT_DIR)

    print("Using device:", C.DEVICE)
    print("[Stage B V3] ASSE strict trajectory-state auditor with delta features")
    print("[Stage B V3] Stage A unchanged / frozen by default")
    print("[Stage B V3] No dense step-level safety supervision")

    step_df_path = os.path.join(C.OUT_DIR, "step_df.pkl")

    if not os.path.exists(step_df_path):
        raise FileNotFoundError(
            f"{step_df_path} not found. Run Stage A first."
        )

    step_df = pd.read_pickle(step_df_path)
    step_df = handle_non_constant_labels(step_df)

    split_file = split_path()
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"{split_file} not found. Run Stage A split creation first."
        )

    split = load_json(split_file)

    train_ids = set(map(int, split["train_ids"]))

    if "val_ids" in split:
        val_ids = set(map(int, split["val_ids"]))
    elif "dev_ids" in split:
        print("[warning] split has no val_ids; falling back to dev_ids.")
        val_ids = set(map(int, split["dev_ids"]))
    else:
        raise ValueError("Split must contain val_ids or dev_ids.")

    test_ids = set(map(int, split.get("test_ids", [])))

    # Drop bad trajectory ids from split automatically.
    valid_ids = set(map(int, step_df["traj_id"].unique()))

    train_ids = train_ids & valid_ids
    val_ids = val_ids & valid_ids
    test_ids = test_ids & valid_ids

    train_df = step_df[step_df["traj_id"].isin(train_ids)].copy()
    val_df = step_df[step_df["traj_id"].isin(val_ids)].copy()
    test_df = step_df[step_df["traj_id"].isin(test_ids)].copy()

    print(f"[data] train steps={len(train_df)}, val steps={len(val_df)}, test steps={len(test_df)}")
    print(
        f"[data] train trajs={train_df['traj_id'].nunique()}, "
        f"val trajs={val_df['traj_id'].nunique()}, "
        f"test trajs={test_df['traj_id'].nunique()}"
    )

    train_examples = build_examples(train_df)
    val_examples = build_examples(val_df)

    if len(train_examples) == 0:
        raise RuntimeError("No train examples after filtering.")
    if len(val_examples) == 0:
        raise RuntimeError("No val examples after filtering.")

    train_loader = DataLoader(
        TrajDataset(train_examples),
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        TrajDataset(val_examples),
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    input_dim = train_examples[0].x_seq.shape[1]

    stage_a_path = args.stage_a_ckpt
    if stage_a_path is None:
        stage_a_path = os.path.join(C.OUT_DIR, "stage_a_latent_mechanism.pt")

    if not os.path.exists(stage_a_path):
        raise FileNotFoundError(
            f"{stage_a_path} not found. Pass --stage_a_ckpt explicitly if needed."
        )

    stage_a_ckpt = torch.load(stage_a_path, map_location="cpu")

    model = TrajectoryStateAuditor(
        input_dim=input_dim,
        stage_a_ckpt=stage_a_ckpt,
    ).to(C.DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=C.LR_STAGE_B,
        weight_decay=C.WEIGHT_DECAY,
    )

    pos_weight = compute_pos_weight(train_examples, C.DEVICE)

    if pos_weight is not None:
        print(f"[loss] pos_weight={float(pos_weight.item()):.4f}")
    else:
        print("[loss] pos_weight=None")

    print("[V3 config]")
    print({
        "TRACE_USE_DELTA_FEATURES": C.TRACE_USE_DELTA_FEATURES,
        "LAMBDA_STATE_FINAL": C.LAMBDA_STATE_FINAL,
        "LAMBDA_EVIDENCE_MIL": C.LAMBDA_EVIDENCE_MIL,
        "USE_PREFIX_LOSS": C.USE_PREFIX_LOSS,
        "LAMBDA_PREFIX": C.LAMBDA_PREFIX,
        "PREFIX_LOSS_RHO": C.PREFIX_LOSS_RHO,
        "PREFIX_LOSS_GAMMA": C.PREFIX_LOSS_GAMMA,
        "USE_PREFIX_RANK_LOSS": C.USE_PREFIX_RANK_LOSS,
        "LAMBDA_PREFIX_RANK": C.LAMBDA_PREFIX_RANK,
        "USE_PREFIX_STABLE_LOSS": C.USE_PREFIX_STABLE_LOSS,
        "EARLY_AWARE_SELECTION": C.EARLY_AWARE_SELECTION,
        "DROP_NON_CONSTANT_LABEL_TRAJS": C.DROP_NON_CONSTANT_LABEL_TRAJS,
    })

    best_score = -1.0
    best_state = None
    history = []

    for epoch in range(1, C.EPOCHS_STAGE_B + 1):
        model.train()

        total_loss = 0.0

        comp = {
            "state_final": 0.0,
            "evidence_mil": 0.0,
            "prefix": 0.0,
            "prefix_rank": 0.0,
            "prefix_stable": 0.0,
            "evidence_sparse": 0.0,
            "evidence_smooth": 0.0,
        }

        for batch in train_loader:
            x = batch["x"].to(C.DEVICE)
            mask = batch["mask"].to(C.DEVICE)
            traj_y = batch["traj_y"].to(C.DEVICE)

            out = model(x, mask)

            loss_state_final = F.binary_cross_entropy_with_logits(
                out["traj_logit"],
                traj_y,
                pos_weight=pos_weight,
            )

            loss_evidence_mil = F.binary_cross_entropy_with_logits(
                out["evidence_mil_logit"],
                traj_y,
                pos_weight=pos_weight,
            )

            if C.USE_PREFIX_LOSS:
                loss_prefix = asymmetric_prefix_loss_on_state(
                    state_logits=out["state_logits"],
                    mask=mask,
                    traj_y=traj_y,
                    rho=C.PREFIX_LOSS_RHO,
                    gamma=C.PREFIX_LOSS_GAMMA,
                )
            else:
                loss_prefix = torch.tensor(0.0, device=C.DEVICE)

            if C.USE_PREFIX_RANK_LOSS:
                loss_prefix_rank = unsafe_prefix_ranking_loss_on_state(
                    state_logits=out["state_logits"],
                    mask=mask,
                    traj_y=traj_y,
                    early_frac=C.PREFIX_RANK_EARLY_FRAC,
                    late_frac=C.PREFIX_RANK_LATE_FRAC,
                    margin=C.PREFIX_RANK_MARGIN,
                )
            else:
                loss_prefix_rank = torch.tensor(0.0, device=C.DEVICE)

            if C.USE_PREFIX_STABLE_LOSS:
                loss_prefix_stable = prefix_stability_loss_on_state(
                    state_logits=out["state_logits"],
                    mask=mask,
                    margin=C.PREFIX_STABLE_MARGIN,
                )
            else:
                loss_prefix_stable = torch.tensor(0.0, device=C.DEVICE)

            loss_evidence_sparse = evidence_sparsity_loss(
                out["evidence_scores"],
                mask,
                topk=C.TRACE_MIL_TOPK,
            )

            loss_evidence_smooth = evidence_smooth_loss(
                out["evidence_scores"],
                mask,
            )

            loss = (
                C.LAMBDA_STATE_FINAL * loss_state_final
                + C.LAMBDA_EVIDENCE_MIL * loss_evidence_mil
                + C.LAMBDA_PREFIX * loss_prefix
                + C.LAMBDA_PREFIX_RANK * loss_prefix_rank
                + C.LAMBDA_PREFIX_STABLE * loss_prefix_stable
                + C.LAMBDA_EVIDENCE_SPARSE * loss_evidence_sparse
                + C.LAMBDA_EVIDENCE_SMOOTH * loss_evidence_smooth
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())

            comp["state_final"] += float(loss_state_final.item())
            comp["evidence_mil"] += float(loss_evidence_mil.item())
            comp["prefix"] += float(loss_prefix.item())
            comp["prefix_rank"] += float(loss_prefix_rank.item())
            comp["prefix_stable"] += float(loss_prefix_stable.item())
            comp["evidence_sparse"] += float(loss_evidence_sparse.item())
            comp["evidence_smooth"] += float(loss_evidence_smooth.item())

        train_metrics, _ = evaluate(model, train_loader, C.DEVICE)
        val_metrics, val_step_scores = evaluate(model, val_loader, C.DEVICE)
        prefix_df = evaluate_prefix(model, val_loader, C.DEVICE)

        n = max(len(train_loader), 1)

        if C.EARLY_AWARE_SELECTION:
            score = early_aware_selection_score(val_metrics, prefix_df)
        else:
            score = val_metrics["main_score"]

        row = {
            "epoch": epoch,
            "loss": total_loss / n,
            "selection_score": score,
            **{f"loss_{k}": v / n for k, v in comp.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
            rows = prefix_df[prefix_df["prefix_frac"] == frac]
            if len(rows) > 0:
                row[f"val_prefix{int(frac * 100)}_auroc"] = float(rows["auroc"].iloc[0])
                row[f"val_prefix{int(frac * 100)}_ap"] = float(rows["ap"].iloc[0])

        history.append(row)

        print("=" * 80)
        print(row)
        print("[val prefix metrics]")
        print(prefix_df)

        if not np.isnan(score) and score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            print(f"[best] epoch={epoch}, selection_score={best_score:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_metrics, _ = evaluate(model, train_loader, C.DEVICE)
    final_val_metrics, val_step_scores = evaluate(model, val_loader, C.DEVICE)
    prefix_df = evaluate_prefix(model, val_loader, C.DEVICE)

    prefix = args.out_prefix

    ckpt_out = os.path.join(C.OUT_DIR, f"{prefix}.pt")
    hist_out = os.path.join(C.OUT_DIR, f"{prefix}_history.csv")
    metrics_out = os.path.join(C.OUT_DIR, f"{prefix}_val_metrics.json")
    pred_out = os.path.join(C.OUT_DIR, f"{prefix}_val_step_scores.csv")
    prefix_out = os.path.join(C.OUT_DIR, f"{prefix}_val_prefix_metrics.csv")

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "stage_a_path": stage_a_path,
            "split_path": split_file,
            "config": {
                "model": "asse_strict_trajectory_state_auditor_v3",
                "stage_a": "center_subspace_latent_mechanism",
                "label_source": C.TRAJ_JSONL_PATH,
                "no_step_level_safety_supervision": True,
                "cumulative_state_head": True,
                "local_evidence_mil_auxiliary": True,
                "delta_features": C.TRACE_USE_DELTA_FEATURES,
                "asymmetric_prefix_loss": C.USE_PREFIX_LOSS,
                "unsafe_prefix_ranking_loss": C.USE_PREFIX_RANK_LOSS,
                "prefix_stability_loss": C.USE_PREFIX_STABLE_LOSS,
                "early_aware_selection": C.EARLY_AWARE_SELECTION,

                "raw_residual_path": C.TRACE_USE_RAW_PROJ,
                "raw_dim": C.TRACE_RAW_DIM,
                "mil_topk": C.TRACE_MIL_TOPK,
                "gru_hidden": C.TRACE_GRU_HIDDEN,
                "gru_layers": C.TRACE_GRU_LAYERS,
                "bidirectional": C.TRACE_BIDIRECTIONAL,
                "use_z": C.TRACE_USE_MECH_Z,
                "use_gates": C.TRACE_USE_MECH_GATES,
                "use_scores": C.TRACE_USE_MECH_SCORES,

                "lambda_state_final": C.LAMBDA_STATE_FINAL,
                "lambda_evidence_mil": C.LAMBDA_EVIDENCE_MIL,
                "lambda_prefix": C.LAMBDA_PREFIX,
                "prefix_loss_rho": C.PREFIX_LOSS_RHO,
                "prefix_loss_gamma": C.PREFIX_LOSS_GAMMA,
                "lambda_prefix_rank": C.LAMBDA_PREFIX_RANK,
                "lambda_prefix_stable": C.LAMBDA_PREFIX_STABLE,
                "selected_on": "early_aware_val_score" if C.EARLY_AWARE_SELECTION else "val_full_auroc",
                "drop_non_constant_label_trajs": C.DROP_NON_CONSTANT_LABEL_TRAJS,
            },
            "final_train_metrics": final_train_metrics,
            "final_val_metrics": final_val_metrics,
        },
        ckpt_out,
    )

    pd.DataFrame(history).to_csv(hist_out, index=False)
    val_step_scores.to_csv(pred_out, index=False)
    prefix_df.to_csv(prefix_out, index=False)

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": final_train_metrics,
                "val": final_val_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("Final train metrics:")
    print(final_train_metrics)
    print("Final val metrics:")
    print(final_val_metrics)
    print("Final val prefix:")
    print(prefix_df)

    print(f"Saved model to {ckpt_out}")
    print(f"Saved history to {hist_out}")
    print(f"Saved val metrics to {metrics_out}")
    print(f"Saved val step scores to {pred_out}")
    print(f"Saved val prefix metrics to {prefix_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_prefix",
        type=str,
        default="asse_strict_trace_temporal_mil_v3",
    )

    parser.add_argument(
        "--stage_a_ckpt",
        type=str,
        default=None,
        help="Optional explicit Stage-A checkpoint path.",
    )

    args = parser.parse_args()

    train(args)