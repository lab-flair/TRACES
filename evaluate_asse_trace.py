#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import config_asse_trace_v3 as C
from stage_a_latent_mechanism import (
    MechanismDiscovery,
    split_path,
    load_json,
)


# =========================
# Basic utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cfg(name: str, default=None):
    return getattr(C, name, default)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Optional[str]) -> List[dict]:
    rows = []
    if path is None or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


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

        "BATCH_SIZE": 32,
        "DROPOUT": 0.1,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "EPS": 1e-8,
    }

    for k, v in defaults.items():
        if not hasattr(C, k):
            setattr(C, k, v)


# =========================
# Model utilities
# =========================

def last_valid_logits(seq_logits: torch.Tensor, mask: torch.Tensor):
    """
    seq_logits: [B, T]
    mask: [B, T]

    returns:
      last_logits: [B]
    """
    lengths = mask.sum(dim=1).long().clamp_min(1)
    idx = lengths - 1
    return seq_logits[
        torch.arange(seq_logits.size(0), device=seq_logits.device),
        idx,
    ]


def topk_mean_mil_logits(step_logits: torch.Tensor, mask: torch.Tensor, k: int):
    """
    step_logits: [B, T]
    mask: [B, T]

    returns:
      traj_logits: [B]
      topk_idx: [B, k]
    """
    step_logits = step_logits.masked_fill(mask == 0, -1e9)

    T = step_logits.size(1)
    k = min(int(k), T)

    topk_values, topk_idx = torch.topk(step_logits, k=k, dim=1)

    valid_topk = topk_values > -1e8
    topk_values_safe = torch.where(
        valid_topk,
        topk_values,
        torch.zeros_like(topk_values),
    )

    denom = valid_topk.float().sum(dim=1).clamp_min(1.0)
    traj_logits = topk_values_safe.sum(dim=1) / denom

    return traj_logits, topk_idx


def add_delta_features(feat: torch.Tensor, mask: torch.Tensor):
    """
    feat: [B, T, D]
    mask: [B, T]

    returns:
      concat([feat_t, delta_feat_t])

    delta_feat_1 = 0
    delta_feat_t = feat_t - feat_{t-1}
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


def slice_prefix_batch(x: torch.Tensor, mask: torch.Tensor, frac: float):
    """
    Truncate each trajectory to first frac of valid steps.
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

    return x_prefix, mask_prefix, prefix_lengths


def slice_prefix_batch_by_lengths(
    x: torch.Tensor,
    mask: torch.Tensor,
    prefix_lengths: List[int],
):
    """
    Truncate each trajectory to explicit prefix length.
    """
    B, T, D = x.shape
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


# =========================
# Dataset construction
# =========================

def _prefer_traj_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    For ASSE/R-Judge style tables.

    Some OOD tables may contain:
      - label: event-level / step-level label
      - traj_label: trajectory-level safe/unsafe label

    This evaluator needs label = trajectory-level label.
    """
    df = df.copy()

    if "traj_label" in df.columns:
        if "label" in df.columns and "event_label" not in df.columns:
            df["event_label"] = df["label"]
        df["label"] = df["traj_label"]
        print("[label fix] Using traj_label as trajectory-level label; original label saved as event_label.")

    return df


def normalize_existing_step_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      1. already-built step_df with column x
      2. raw event_repr_table with rep/layer columns

    Returns:
      traj_id, event_idx, event_order, label, event_label(optional), x
    """
    df = df.copy()
    df = _prefer_traj_label(df)

    # Case 1: already-built step_df with x vectors.
    if "x" in df.columns:
        if "event_order" not in df.columns:
            if "event_order_among_decisions" in df.columns:
                df["event_order"] = maybe_numeric(df["event_order_among_decisions"])
            else:
                df["event_order"] = df.groupby("traj_id").cumcount()

        if "label" not in df.columns:
            df["label"] = np.nan

        df["label"] = maybe_numeric(df["label"])

        if "event_label" in df.columns:
            df["event_label"] = maybe_numeric(df["event_label"])

        df["traj_id"] = maybe_numeric(df["traj_id"]).astype(int)
        df["event_idx"] = maybe_numeric(df["event_idx"]).astype(int)
        df["event_order"] = maybe_numeric(df["event_order"]).astype(int)

        return df.sort_values(["traj_id", "event_order"]).reset_index(drop=True)

    # Case 2: raw event_repr_table with one row per layer.
    required = {"traj_id", "event_idx", "layer", "rep"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input table has neither x column nor required raw columns. Missing={missing}"
        )

    if "event_type" in df.columns and cfg("EVENT_TYPE", None) is not None:
        before = len(df)
        df = df[df["event_type"] == C.EVENT_TYPE].copy()
        print(f"[filter] event_type={C.EVENT_TYPE}: {before} -> {len(df)} rows")

    if "layer" in df.columns and cfg("LAYERS", None) is not None:
        before = len(df)
        df = df[df["layer"].isin(C.LAYERS)].copy()
        print(f"[filter] layers={C.LAYERS}: {before} -> {len(df)} rows")

    if "event_order_among_decisions" in df.columns:
        df["event_order_among_decisions"] = maybe_numeric(
            df["event_order_among_decisions"]
        )
    else:
        df["event_order_among_decisions"] = df.groupby(
            ["traj_id", "event_idx"]
        ).ngroup()

    if "label" in df.columns:
        df["label"] = maybe_numeric(df["label"])
    else:
        df["label"] = np.nan

    if "event_label" in df.columns:
        df["event_label"] = maybe_numeric(df["event_label"])

    rows = []

    for (tid, eidx), g in df.groupby(["traj_id", "event_idx"], sort=False):
        row0 = g.iloc[0]

        layer_to_rep = {}
        for _, r in g.iterrows():
            rep = r["rep"]

            # Some CSVs may store arrays as strings. Prefer pickle, but this fallback helps.
            if isinstance(rep, str):
                rep = np.array(json.loads(rep), dtype=np.float32)
            else:
                rep = np.asarray(rep, dtype=np.float32)

            layer_to_rep[int(r["layer"])] = rep

        if not all(layer in layer_to_rep for layer in C.LAYERS):
            continue

        x = np.concatenate(
            [layer_to_rep[layer] for layer in C.LAYERS],
            axis=0,
        ).astype(np.float32)

        label = row0["label"]
        label_out = np.nan if pd.isna(label) else int(label)

        event_label_out = np.nan
        if "event_label" in row0.index and not pd.isna(row0["event_label"]):
            event_label_out = int(row0["event_label"])

        rows.append({
            "traj_id": int(tid),
            "event_idx": int(eidx),
            "event_order": int(row0["event_order_among_decisions"]),
            "label": label_out,
            "event_label": event_label_out,
            "x": x,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["traj_id", "event_order"]).reset_index(drop=True)

    print("[normalize] output rows:", len(out))
    print("[normalize] output trajs:", out["traj_id"].nunique())
    if "label" in out.columns:
        print("[normalize] trajectory-level label counts:")
        print(out.groupby("traj_id")["label"].first().value_counts(dropna=False))
        print("[normalize] label nunique per traj:")
        print(out.groupby("traj_id")["label"].nunique(dropna=False).value_counts())
    if "event_label" in out.columns:
        print("[normalize] event_label counts:")
        print(out["event_label"].value_counts(dropna=False))

    return out


def load_step_table(table_path: str) -> pd.DataFrame:
    if table_path.endswith(".pkl") or table_path.endswith(".pickle"):
        df = pd.read_pickle(table_path)
    elif table_path.endswith(".csv"):
        df = pd.read_csv(table_path)
    else:
        raise ValueError(
            f"Unsupported table format: {table_path}. Use .pkl/.pickle or .csv."
        )

    return normalize_existing_step_df(df)


class TrajExample:
    def __init__(
        self,
        traj_id: int,
        x_seq: np.ndarray,
        traj_y: Optional[int],
        event_indices: np.ndarray,
    ):
        self.traj_id = int(traj_id)
        self.x_seq = x_seq.astype(np.float32)
        self.traj_y = None if traj_y is None else int(traj_y)
        self.event_indices = event_indices.astype(np.int64)


def build_examples(step_df: pd.DataFrame, require_labels: bool = False):
    examples = []

    for tid, g in step_df.groupby("traj_id", sort=False):
        g = g.sort_values("event_order")

        x_seq = np.stack(g["x"].values).astype(np.float32)

        labels = g["label"].values if "label" in g.columns else np.array([np.nan])

        if np.all(pd.isna(labels)):
            traj_y = None
        else:
            labels_clean = pd.Series(labels).dropna().astype(int).values
            if len(labels_clean) == 0:
                traj_y = None
            else:
                traj_y = int(labels_clean[0])
                if not np.all(labels_clean == traj_y):
                    print(
                        f"[warning] traj_id={tid} has non-constant trajectory labels. "
                        f"Using first non-null label={traj_y}. "
                        f"Check whether label should be traj_label."
                    )

        if require_labels and traj_y is None:
            continue

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
    def __init__(self, examples: List[TrajExample]):
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

    # y = -1 means unknown label.
    traj_y = torch.full((B,), -1.0, dtype=torch.float32)

    traj_ids = []
    event_indices = []

    for i, ex in enumerate(batch):
        T = ex.x_seq.shape[0]

        x[i, :T] = torch.tensor(ex.x_seq, dtype=torch.float32)
        mask[i, :T] = 1

        if ex.traj_y is not None:
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
# Model definition
# Must match Stage B V3
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

        evidence_mil_logit, evidence_topk_idx = topk_mean_mil_logits(
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
            "evidence_topk_idx": evidence_topk_idx,

            # Compatibility aliases
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
# Loading model
# =========================

def resolve_stage_a_path(model_ckpt: dict, args) -> str:
    if args.stage_a_ckpt is not None:
        return args.stage_a_ckpt

    if "stage_a_path" in model_ckpt and model_ckpt["stage_a_path"] is not None:
        return model_ckpt["stage_a_path"]

    return os.path.join(C.OUT_DIR, "stage_a_latent_mechanism.pt")


def load_model(args, input_dim: int):
    model_ckpt = torch.load(args.ckpt, map_location="cpu")
    stage_a_path = resolve_stage_a_path(model_ckpt, args)

    if not os.path.exists(stage_a_path):
        raise FileNotFoundError(
            f"Stage-A checkpoint not found: {stage_a_path}. "
            f"Pass --stage_a_ckpt explicitly."
        )

    stage_a_ckpt = torch.load(stage_a_path, map_location="cpu")

    model_name = model_ckpt.get("config", {}).get("model", None)
    if model_name is not None and model_name != "trajectory_state_auditor_v3":
        print(
            f"[warning] checkpoint config model={model_name}; "
            f"this evaluator expects trajectory_state_auditor_v3."
        )

    model = TrajectoryStateAuditor(
        input_dim=input_dim,
        stage_a_ckpt=stage_a_ckpt,
    )

    model.load_state_dict(model_ckpt["model_state"], strict=True)
    model.to(C.DEVICE)
    model.eval()

    return model, model_ckpt, stage_a_path


# =========================
# Metrics
# =========================

def has_labels(y: np.ndarray) -> bool:
    return y is not None and len(y) > 0 and np.all(y >= 0)


def binary_metrics(y, p, threshold: float) -> Dict[str, float]:
    if not has_labels(y):
        return {}

    pred = (p >= threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "pos_rate_true": float(np.mean(y)),
        "pos_rate_pred": float(np.mean(pred)),
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }

    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")

    return out


def find_best_threshold(y, p, metric="f1"):
    if not has_labels(y):
        return 0.5, float("nan")

    thresholds = np.linspace(0.01, 0.99, 197)

    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        pred = (p >= t).astype(int)

        if metric == "f1":
            score = f1_score(y, pred, zero_division=0)
        elif metric == "acc":
            score = accuracy_score(y, pred)
        elif metric == "recall":
            score = recall_score(y, pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)


def safe_get(d, key, default=np.nan):
    if d is None:
        return default
    return d.get(key, default)


# =========================
# Prediction collection
# =========================

@torch.no_grad()
def collect_full_predictions(model, loader, device):
    model.eval()

    traj_rows = []
    step_rows = []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["traj_y"].detach().cpu().numpy()

        out = model(x, mask)

        traj_score = out["traj_score"].detach().cpu().numpy()
        traj_logit = out["traj_logit"].detach().cpu().numpy()

        state_scores = out["state_scores"].detach().cpu().numpy()
        state_logits = out["state_logits"].detach().cpu().numpy()

        evidence_scores = out["evidence_scores"].detach().cpu().numpy()
        evidence_logits = out["evidence_logits"].detach().cpu().numpy()
        evidence_mil_score = out["evidence_mil_score"].detach().cpu().numpy()
        evidence_mil_logit = out["evidence_mil_logit"].detach().cpu().numpy()
        evidence_topk_idx = out["evidence_topk_idx"].detach().cpu().numpy()

        mask_np = batch["mask"].numpy()
        event_indices = batch["event_indices"].numpy()

        for i, tid in enumerate(batch["traj_ids"]):
            T = int(mask_np[i].sum())

            evidence_topk_valid = []
            for idx in evidence_topk_idx[i].tolist():
                if 0 <= idx < T:
                    evidence_topk_valid.append(int(idx))

            # State top-k is not used for final trajectory prediction, but useful for inspection.
            state_topk_valid = []
            if T > 0:
                k = min(int(C.TRACE_MIL_TOPK), T)
                state_topk_valid = np.argsort(-state_scores[i, :T])[:k].astype(int).tolist()

            traj_rows.append({
                "traj_id": int(tid),
                "T": int(T),
                "traj_y": int(y[i]) if y[i] >= 0 else None,

                # Main score: final cumulative state q_T.
                "traj_score": float(traj_score[i]),
                "traj_logit": float(traj_logit[i]),

                # Auxiliary local evidence MIL score.
                "evidence_mil_score": float(evidence_mil_score[i]),
                "evidence_mil_logit": float(evidence_mil_logit[i]),

                "evidence_topk_step_indices_0based": json.dumps(evidence_topk_valid),
                "evidence_topk_event_indices": json.dumps(
                    [int(event_indices[i, j]) for j in evidence_topk_valid]
                ),
                "state_topk_step_indices_0based": json.dumps(state_topk_valid),
                "state_topk_event_indices": json.dumps(
                    [int(event_indices[i, j]) for j in state_topk_valid]
                ),
            })

            for t in range(T):
                step_rows.append({
                    "traj_id": int(tid),
                    "event_idx": int(event_indices[i, t]),
                    "t_0based": int(t),
                    "t_1based": int(t + 1),
                    "T": int(T),
                    "relative_t": float((t + 1) / T),
                    "traj_y": int(y[i]) if y[i] >= 0 else None,

                    "full_traj_score": float(traj_score[i]),

                    # Cumulative state head q_t.
                    "state_risk_score": float(state_scores[i, t]),
                    "state_risk_logit": float(state_logits[i, t]),

                    # Local evidence head e_t.
                    "local_evidence_score": float(evidence_scores[i, t]),
                    "local_evidence_logit": float(evidence_logits[i, t]),

                    # Compatibility names for old analysis scripts.
                    "step_risk_score": float(state_scores[i, t]),
                    "step_risk_logit": float(state_logits[i, t]),

                    "is_evidence_topk_full": int(t in evidence_topk_valid),
                    "is_state_topk_full": int(t in state_topk_valid),
                })

    return pd.DataFrame(traj_rows), pd.DataFrame(step_rows)


@torch.no_grad()
def collect_prefix_predictions_all_steps(model, loader, device):
    """
    For every trajectory and every prefix length t=1..T, predict final
    trajectory safety from the truncated prefix.

    For V3, prefix prediction is q_t, i.e. the final cumulative state of
    the truncated prefix.
    """
    model.eval()

    rows = []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)

        y = batch["traj_y"].detach().cpu().numpy()
        lengths = mask.sum(dim=1).long().detach().cpu().numpy()

        B, T_max, D = x.shape

        for prefix_t in range(1, T_max + 1):
            active = lengths >= prefix_t
            if not np.any(active):
                continue

            active_indices = np.where(active)[0].tolist()

            x_sub = x[active_indices]
            mask_sub = mask[active_indices]

            prefix_lengths = [prefix_t for _ in active_indices]
            x_prefix, mask_prefix = slice_prefix_batch_by_lengths(
                x_sub,
                mask_sub,
                prefix_lengths,
            )

            out = model(x_prefix, mask_prefix)

            p = out["traj_score"].detach().cpu().numpy()
            logit = out["traj_logit"].detach().cpu().numpy()
            evidence_mil_p = out["evidence_mil_score"].detach().cpu().numpy()

            # Last state/evidence score of the truncated prefix.
            state_scores = out["state_scores"].detach().cpu().numpy()
            evidence_scores = out["evidence_scores"].detach().cpu().numpy()

            for j, bi in enumerate(active_indices):
                T = int(lengths[bi])
                last_idx = prefix_t - 1

                rows.append({
                    "traj_id": int(batch["traj_ids"][bi]),
                    "prefix_t": int(prefix_t),
                    "T": int(T),
                    "relative_prefix": float(prefix_t / T),
                    "traj_y": int(y[bi]) if y[bi] >= 0 else None,

                    "prefix_traj_score": float(p[j]),
                    "prefix_traj_logit": float(logit[j]),

                    "prefix_state_score": float(state_scores[j, last_idx]),
                    "prefix_local_evidence_score": float(evidence_scores[j, last_idx]),
                    "prefix_evidence_mil_score": float(evidence_mil_p[j]),
                })

    return pd.DataFrame(rows)


@torch.no_grad()
def collect_prefix_fraction_predictions(
    model,
    loader,
    device,
    fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
):
    model.eval()

    rows = []

    for frac in fractions:
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)

            y = batch["traj_y"].detach().cpu().numpy()
            lengths = mask.sum(dim=1).long().detach().cpu().numpy()

            x_prefix, mask_prefix, prefix_lengths = slice_prefix_batch(x, mask, frac)

            out = model(x_prefix, mask_prefix)

            p = out["traj_score"].detach().cpu().numpy()
            logit = out["traj_logit"].detach().cpu().numpy()
            evidence_mil_p = out["evidence_mil_score"].detach().cpu().numpy()

            state_scores = out["state_scores"].detach().cpu().numpy()
            evidence_scores = out["evidence_scores"].detach().cpu().numpy()

            for i, tid in enumerate(batch["traj_ids"]):
                T = int(lengths[i])
                prefix_t = int(prefix_lengths[i])
                last_idx = prefix_t - 1

                rows.append({
                    "traj_id": int(tid),
                    "prefix_frac": float(frac),
                    "prefix_t": int(prefix_t),
                    "T": int(T),
                    "traj_y": int(y[i]) if y[i] >= 0 else None,

                    "prefix_traj_score": float(p[i]),
                    "prefix_traj_logit": float(logit[i]),

                    "prefix_state_score": float(state_scores[i, last_idx]),
                    "prefix_local_evidence_score": float(evidence_scores[i, last_idx]),
                    "prefix_evidence_mil_score": float(evidence_mil_p[i]),
                })

    return pd.DataFrame(rows)


# =========================
# Prefix metrics
# =========================

def metrics_by_prefix_fraction(prefix_frac_df: pd.DataFrame, threshold: float):
    rows = []

    if "traj_y" not in prefix_frac_df.columns:
        return pd.DataFrame(rows)

    for frac, g in prefix_frac_df.groupby("prefix_frac"):
        g = g.dropna(subset=["traj_y"])
        if len(g) == 0:
            continue

        y = g["traj_y"].astype(int).values
        p = g["prefix_traj_score"].values

        m = binary_metrics(y, p, threshold)

        best_t_f1, best_f1 = find_best_threshold(y, p, metric="f1")
        best_t_acc, best_acc = find_best_threshold(y, p, metric="acc")
        best_t_recall, best_recall = find_best_threshold(y, p, metric="recall")

        m.update({
            "prefix_frac": float(frac),
            "best_f1_threshold_oracle": float(best_t_f1),
            "best_f1_oracle": float(best_f1),
            "best_acc_threshold_oracle": float(best_t_acc),
            "best_acc_oracle": float(best_acc),
            "best_recall_threshold_oracle": float(best_t_recall),
            "best_recall_oracle": float(best_recall),
        })

        rows.append(m)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("prefix_frac")


def metrics_by_absolute_prefix_step(prefix_step_df: pd.DataFrame, threshold: float):
    rows = []

    if "traj_y" not in prefix_step_df.columns:
        return pd.DataFrame(rows)

    for prefix_t, g in prefix_step_df.groupby("prefix_t"):
        g = g.dropna(subset=["traj_y"])
        if len(g) == 0:
            continue

        y = g["traj_y"].astype(int).values
        p = g["prefix_traj_score"].values

        m = binary_metrics(y, p, threshold)
        m["prefix_t"] = int(prefix_t)

        rows.append(m)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("prefix_t")


def metrics_by_relative_bins(
    prefix_step_df: pd.DataFrame,
    threshold: float,
    bin_size: float = 0.1,
):
    df = prefix_step_df.copy()
    df = df.dropna(subset=["traj_y"])

    if len(df) == 0:
        return pd.DataFrame()

    bins = np.arange(0, 1.0 + bin_size, bin_size)
    bins[-1] = 1.000001

    df["relative_bin"] = pd.cut(
        df["relative_prefix"],
        bins=bins,
        include_lowest=True,
        right=True,
    ).astype(str)

    rows = []

    for b, g in df.groupby("relative_bin", sort=False):
        y = g["traj_y"].astype(int).values
        p = g["prefix_traj_score"].values

        m = binary_metrics(y, p, threshold)
        m["relative_bin"] = b
        m["mean_relative_prefix"] = float(g["relative_prefix"].mean())

        rows.append(m)

    return pd.DataFrame(rows)


# =========================
# Early-warning metrics
# =========================

def early_warning_metrics(prefix_step_df: pd.DataFrame, threshold: float):
    """
    Early warning uses cumulative prefix state predictions:
      prefix_traj_score(prefix 1..t) >= threshold

    Existing metrics:
      first_detection_rate:
        among all unsafe trajectories, fraction detected at any prefix including final step.

      stable_detection_rate:
        among all unsafe trajectories, fraction whose score stays above threshold from
        some prefix onward.

    New stricter metrics:
      first_early_detection_rate:
        among all unsafe trajectories, fraction detected at least 1 step before the end.

      first_early_detection_among_full_tp_rate:
        among unsafe trajectories that are successfully predicted as unsafe at the full
        trajectory level, fraction that were detected at least 1 step before the end.

      stable_early_detection_among_full_tp_rate:
        among unsafe trajectories that are successfully predicted as unsafe at the full
        trajectory level, fraction that were stably detected at least 1 step before the end.

      early_alert_precision:
        among trajectories that trigger an alert before the final step, fraction that are
        truly unsafe. This helps control the false-alarm side of proactive detection.

    advance steps:
      T - t_detect
    """
    df = prefix_step_df.copy()
    df = df.dropna(subset=["traj_y"])

    if len(df) == 0:
        return {}, pd.DataFrame(), pd.DataFrame()

    # Use the last prefix as the full-trajectory decision under the same threshold.
    last_rows = (
        df.sort_values(["traj_id", "prefix_t"])
          .groupby("traj_id", as_index=False)
          .tail(1)
          .copy()
    )
    last_rows["full_pred_unsafe"] = (
        last_rows["prefix_traj_score"].values >= threshold
    ).astype(int)
    last_rows["traj_y_int"] = last_rows["traj_y"].astype(int)

    full_tp_ids = set(
        last_rows[
            (last_rows["traj_y_int"] == 1)
            & (last_rows["full_pred_unsafe"] == 1)
        ]["traj_id"].astype(int).tolist()
    )

    unsafe = df[df["traj_y"].astype(int) == 1].copy()

    if len(unsafe) == 0:
        # Still compute early alert precision over safe-only data if needed.
        early_alert_rows = []
        for tid, g in df.groupby("traj_id"):
            g = g.sort_values("prefix_t")
            T = int(g["T"].iloc[0])
            if T <= 1:
                continue
            early_g = g[g["prefix_t"].astype(int) < T]
            if len(early_g) == 0:
                continue
            scores = early_g["prefix_traj_score"].values
            if np.any(scores >= threshold):
                early_alert_rows.append({
                    "traj_id": int(tid),
                    "T": T,
                    "traj_y": int(g["traj_y"].iloc[0]),
                    "first_early_alert_t": int(early_g.iloc[int(np.argmax(scores >= threshold))]["prefix_t"]),
                    "score_at_first_early_alert": float(scores[int(np.argmax(scores >= threshold))]),
                })

        early_alert_df = pd.DataFrame(early_alert_rows)
        out = {
            "threshold": float(threshold),
            "n_unsafe": 0,
            "n_full_tp_unsafe": 0,
            "first_detection_rate": float("nan"),
            "stable_detection_rate": float("nan"),
            "first_early_detection_rate": float("nan"),
            "stable_early_detection_rate": float("nan"),
            "first_early_detection_among_full_tp_rate": float("nan"),
            "stable_early_detection_among_full_tp_rate": float("nan"),
            "early_alert_precision": float("nan") if len(early_alert_df) == 0 else 0.0,
            "early_alert_rate_all_traj": float(len(early_alert_df) / max(df["traj_id"].nunique(), 1)),
        }
        return out, pd.DataFrame(), pd.DataFrame()

    first_detect_rows = []
    stable_detect_rows = []
    early_alert_rows = []

    for tid, g in df.groupby("traj_id"):
        g = g.sort_values("prefix_t")
        T = int(g["T"].iloc[0])
        y_int = int(g["traj_y"].iloc[0])

        scores = g["prefix_traj_score"].values
        prefix_ts = g["prefix_t"].astype(int).values
        above = scores >= threshold

        # Any early alert, including false positives, for early-alert precision.
        if T > 1:
            early_mask = prefix_ts < T
            if np.any(above & early_mask):
                first_early_idx = int(np.where(above & early_mask)[0][0])
                early_alert_rows.append({
                    "traj_id": int(tid),
                    "T": T,
                    "traj_y": int(y_int),
                    "first_early_alert_t": int(prefix_ts[first_early_idx]),
                    "advance_steps": int(T - prefix_ts[first_early_idx]),
                    "relative_alert": float(prefix_ts[first_early_idx] / T),
                    "score_at_first_early_alert": float(scores[first_early_idx]),
                })

        # Detection metrics below are defined for unsafe trajectories only.
        if y_int != 1:
            continue

        # First detection, including final step.
        if np.any(above):
            idx = int(np.argmax(above))
            t_det = int(prefix_ts[idx])
            advance_steps = int(T - t_det)

            first_detect_rows.append({
                "traj_id": int(tid),
                "T": T,
                "t_detect": t_det,
                "advance_steps": advance_steps,
                "is_early_at_least_1_step": int(advance_steps >= 1),
                "relative_detect": float(t_det / T),
                "score_at_detect": float(scores[idx]),
                "is_full_tp_unsafe": int(int(tid) in full_tp_ids),
            })

        # Stable detection, including final step.
        stable_idx = None
        for i in range(len(above)):
            if np.all(above[i:]):
                stable_idx = i
                break

        if stable_idx is not None:
            t_stable = int(prefix_ts[stable_idx])
            stable_advance_steps = int(T - t_stable)

            stable_detect_rows.append({
                "traj_id": int(tid),
                "T": T,
                "t_stable_detect": t_stable,
                "stable_advance_steps": stable_advance_steps,
                "is_stable_early_at_least_1_step": int(stable_advance_steps >= 1),
                "relative_stable_detect": float(t_stable / T),
                "score_at_stable_detect": float(scores[stable_idx]),
                "is_full_tp_unsafe": int(int(tid) in full_tp_ids),
            })

    n_unsafe = unsafe["traj_id"].nunique()
    n_total_traj = df["traj_id"].nunique()
    n_full_tp_unsafe = len(full_tp_ids)

    first_df = pd.DataFrame(first_detect_rows)
    stable_df = pd.DataFrame(stable_detect_rows)
    early_alert_df = pd.DataFrame(early_alert_rows)

    first_early_df = (
        first_df[first_df["is_early_at_least_1_step"] == 1].copy()
        if len(first_df) > 0 else pd.DataFrame()
    )
    stable_early_df = (
        stable_df[stable_df["is_stable_early_at_least_1_step"] == 1].copy()
        if len(stable_df) > 0 else pd.DataFrame()
    )

    if len(first_df) > 0:
        first_full_tp_df = first_df[first_df["is_full_tp_unsafe"] == 1].copy()
        first_full_tp_early_df = first_full_tp_df[
            first_full_tp_df["is_early_at_least_1_step"] == 1
        ].copy()
    else:
        first_full_tp_df = pd.DataFrame()
        first_full_tp_early_df = pd.DataFrame()

    if len(stable_df) > 0:
        stable_full_tp_df = stable_df[stable_df["is_full_tp_unsafe"] == 1].copy()
        stable_full_tp_early_df = stable_full_tp_df[
            stable_full_tp_df["is_stable_early_at_least_1_step"] == 1
        ].copy()
    else:
        stable_full_tp_df = pd.DataFrame()
        stable_full_tp_early_df = pd.DataFrame()

    out = {
        "threshold": float(threshold),
        "n_unsafe": int(n_unsafe),
        "n_full_tp_unsafe": int(n_full_tp_unsafe),

        # Original style: detected at any prefix, including final.
        "first_detection_rate": float(len(first_df) / max(n_unsafe, 1)),
        "stable_detection_rate": float(len(stable_df) / max(n_unsafe, 1)),

        # Stricter: detected before final step.
        "first_early_detection_rate": float(len(first_early_df) / max(n_unsafe, 1)),
        "stable_early_detection_rate": float(len(stable_early_df) / max(n_unsafe, 1)),

        # Your requested metric:
        # among successful full unsafe predictions, how many were warned at least 1 step early.
        "first_early_detection_among_full_tp_rate": float(
            len(first_full_tp_early_df) / max(n_full_tp_unsafe, 1)
        ) if n_full_tp_unsafe > 0 else float("nan"),
        "stable_early_detection_among_full_tp_rate": float(
            len(stable_full_tp_early_df) / max(n_full_tp_unsafe, 1)
        ) if n_full_tp_unsafe > 0 else float("nan"),

        "n_first_detected_unsafe": int(len(first_df)),
        "n_first_early_detected_unsafe": int(len(first_early_df)),
        "n_stable_detected_unsafe": int(len(stable_df)),
        "n_stable_early_detected_unsafe": int(len(stable_early_df)),
        "n_first_full_tp_early_detected_unsafe": int(len(first_full_tp_early_df)),
        "n_stable_full_tp_early_detected_unsafe": int(len(stable_full_tp_early_df)),

        # False-alarm-aware proactive metric.
        "n_early_alert_all_traj": int(len(early_alert_df)),
        "early_alert_rate_all_traj": float(len(early_alert_df) / max(n_total_traj, 1)),
        "early_alert_precision": (
            float((early_alert_df["traj_y"].astype(int) == 1).mean())
            if len(early_alert_df) > 0
            else float("nan")
        ),
    }

    if len(first_df) > 0:
        out.update({
            "first_detect_mean_advance_steps_detected_only": float(first_df["advance_steps"].mean()),
            "first_detect_median_advance_steps_detected_only": float(first_df["advance_steps"].median()),
            "first_detect_mean_relative_detect_detected_only": float(first_df["relative_detect"].mean()),
            "first_detect_median_relative_detect_detected_only": float(first_df["relative_detect"].median()),
        })
    else:
        out.update({
            "first_detect_mean_advance_steps_detected_only": float("nan"),
            "first_detect_median_advance_steps_detected_only": float("nan"),
            "first_detect_mean_relative_detect_detected_only": float("nan"),
            "first_detect_median_relative_detect_detected_only": float("nan"),
        })

    if len(first_early_df) > 0:
        out.update({
            "first_early_detect_mean_advance_steps_detected_only": float(first_early_df["advance_steps"].mean()),
            "first_early_detect_median_advance_steps_detected_only": float(first_early_df["advance_steps"].median()),
            "first_early_detect_mean_relative_detect_detected_only": float(first_early_df["relative_detect"].mean()),
        })
    else:
        out.update({
            "first_early_detect_mean_advance_steps_detected_only": float("nan"),
            "first_early_detect_median_advance_steps_detected_only": float("nan"),
            "first_early_detect_mean_relative_detect_detected_only": float("nan"),
        })

    if len(stable_df) > 0:
        out.update({
            "stable_detect_mean_advance_steps_detected_only": float(stable_df["stable_advance_steps"].mean()),
            "stable_detect_median_advance_steps_detected_only": float(stable_df["stable_advance_steps"].median()),
            "stable_detect_mean_relative_detect_detected_only": float(stable_df["relative_stable_detect"].mean()),
            "stable_detect_median_relative_detect_detected_only": float(stable_df["relative_stable_detect"].median()),
        })
    else:
        out.update({
            "stable_detect_mean_advance_steps_detected_only": float("nan"),
            "stable_detect_median_advance_steps_detected_only": float("nan"),
            "stable_detect_mean_relative_detect_detected_only": float("nan"),
            "stable_detect_median_relative_detect_detected_only": float("nan"),
        })

    if len(stable_early_df) > 0:
        out.update({
            "stable_early_detect_mean_advance_steps_detected_only": float(stable_early_df["stable_advance_steps"].mean()),
            "stable_early_detect_median_advance_steps_detected_only": float(stable_early_df["stable_advance_steps"].median()),
            "stable_early_detect_mean_relative_detect_detected_only": float(stable_early_df["relative_stable_detect"].mean()),
        })
    else:
        out.update({
            "stable_early_detect_mean_advance_steps_detected_only": float("nan"),
            "stable_early_detect_median_advance_steps_detected_only": float("nan"),
            "stable_early_detect_mean_relative_detect_detected_only": float("nan"),
        })

    return out, first_df, stable_df


# =========================
# Aggregate summary
# =========================

def build_aggregate_summary(
    args,
    metrics: dict,
    prefix_frac_metrics: pd.DataFrame,
    prefix_abs_step_metrics: pd.DataFrame,
    prefix_relative_bin_metrics: pd.DataFrame,
    threshold: float,
):
    full = metrics.get("full_traj", {})
    ew = metrics.get("early_warning", {})
    threshold_info = metrics.get("threshold_info", {})

    row = {
        "mode": args.mode,
        "ood_name": args.ood_name,
        "n_traj": metrics.get("n_traj", np.nan),
        "n_steps": metrics.get("n_steps", np.nan),

        "threshold": float(threshold),
        "threshold_source": threshold_info.get("source", None),
        "threshold_metric": threshold_info.get("threshold_metric", None),

        "full_acc": safe_get(full, "acc"),
        "full_precision": safe_get(full, "precision"),
        "full_recall": safe_get(full, "recall"),
        "full_f1": safe_get(full, "f1"),
        "full_auroc": safe_get(full, "auroc"),
        "full_ap": safe_get(full, "ap"),
        "full_pos_rate_true": safe_get(full, "pos_rate_true"),
        "full_pos_rate_pred": safe_get(full, "pos_rate_pred"),

        "first_detection_rate": safe_get(ew, "first_detection_rate"),
        "first_detect_mean_advance_steps": safe_get(ew, "first_detect_mean_advance_steps_detected_only"),
        "first_detect_median_advance_steps": safe_get(ew, "first_detect_median_advance_steps_detected_only"),
        "first_detect_mean_relative_detect": safe_get(ew, "first_detect_mean_relative_detect_detected_only"),

        "stable_detection_rate": safe_get(ew, "stable_detection_rate"),
        "stable_detect_mean_advance_steps": safe_get(ew, "stable_detect_mean_advance_steps_detected_only"),
        "stable_detect_median_advance_steps": safe_get(ew, "stable_detect_median_advance_steps_detected_only"),
        "stable_detect_mean_relative_detect": safe_get(ew, "stable_detect_mean_relative_detect_detected_only"),

        # Stricter proactive metrics: warnings must occur at least 1 step before the end.
        "first_early_detection_rate": safe_get(ew, "first_early_detection_rate"),
        "first_early_detect_mean_advance_steps": safe_get(ew, "first_early_detect_mean_advance_steps_detected_only"),
        "first_early_detect_median_advance_steps": safe_get(ew, "first_early_detect_median_advance_steps_detected_only"),
        "first_early_detect_mean_relative_detect": safe_get(ew, "first_early_detect_mean_relative_detect_detected_only"),

        "stable_early_detection_rate": safe_get(ew, "stable_early_detection_rate"),
        "stable_early_detect_mean_advance_steps": safe_get(ew, "stable_early_detect_mean_advance_steps_detected_only"),
        "stable_early_detect_median_advance_steps": safe_get(ew, "stable_early_detect_median_advance_steps_detected_only"),
        "stable_early_detect_mean_relative_detect": safe_get(ew, "stable_early_detect_mean_relative_detect_detected_only"),

        # Requested metric:
        # among unsafe trajectories successfully predicted unsafe at full trajectory,
        # how many were warned at least one step before the end.
        "n_full_tp_unsafe": safe_get(ew, "n_full_tp_unsafe"),
        "first_early_detection_among_full_tp_rate": safe_get(ew, "first_early_detection_among_full_tp_rate"),
        "stable_early_detection_among_full_tp_rate": safe_get(ew, "stable_early_detection_among_full_tp_rate"),
        "n_first_full_tp_early_detected_unsafe": safe_get(ew, "n_first_full_tp_early_detected_unsafe"),
        "n_stable_full_tp_early_detected_unsafe": safe_get(ew, "n_stable_full_tp_early_detected_unsafe"),

        # False-alarm-aware proactive metric.
        "early_alert_rate_all_traj": safe_get(ew, "early_alert_rate_all_traj"),
        "early_alert_precision": safe_get(ew, "early_alert_precision"),
        "n_early_alert_all_traj": safe_get(ew, "n_early_alert_all_traj"),
    }

    if prefix_frac_metrics is not None and len(prefix_frac_metrics) > 0:
        pf = prefix_frac_metrics.copy()

        for _, r in pf.iterrows():
            frac = float(r["prefix_frac"])
            frac_key = int(round(frac * 100))

            row[f"prefix{frac_key}_acc"] = r.get("acc", np.nan)
            row[f"prefix{frac_key}_precision"] = r.get("precision", np.nan)
            row[f"prefix{frac_key}_recall"] = r.get("recall", np.nan)
            row[f"prefix{frac_key}_f1"] = r.get("f1", np.nan)
            row[f"prefix{frac_key}_auroc"] = r.get("auroc", np.nan)
            row[f"prefix{frac_key}_ap"] = r.get("ap", np.nan)
            row[f"prefix{frac_key}_pos_rate_pred"] = r.get("pos_rate_pred", np.nan)

        if "auroc" in pf.columns and pf["auroc"].notna().any():
            best_auroc_row = pf.loc[pf["auroc"].idxmax()]
            row["best_prefix_by_auroc_frac"] = best_auroc_row["prefix_frac"]
            row["best_prefix_by_auroc"] = best_auroc_row["auroc"]

        if "ap" in pf.columns and pf["ap"].notna().any():
            best_ap_row = pf.loc[pf["ap"].idxmax()]
            row["best_prefix_by_ap_frac"] = best_ap_row["prefix_frac"]
            row["best_prefix_by_ap"] = best_ap_row["ap"]

        if "f1" in pf.columns and pf["f1"].notna().any():
            best_f1_row = pf.loc[pf["f1"].idxmax()]
            row["best_prefix_by_f1_frac"] = best_f1_row["prefix_frac"]
            row["best_prefix_by_f1"] = best_f1_row["f1"]

    if prefix_abs_step_metrics is not None and len(prefix_abs_step_metrics) > 0:
        ps = prefix_abs_step_metrics.copy()

        if "auroc" in ps.columns and ps["auroc"].notna().any():
            best_step_auroc = ps.loc[ps["auroc"].idxmax()]
            row["best_abs_prefix_step_by_auroc"] = best_step_auroc["prefix_t"]
            row["best_abs_prefix_step_auroc"] = best_step_auroc["auroc"]

        if "f1" in ps.columns and ps["f1"].notna().any():
            best_step_f1 = ps.loc[ps["f1"].idxmax()]
            row["best_abs_prefix_step_by_f1"] = best_step_f1["prefix_t"]
            row["best_abs_prefix_step_f1"] = best_step_f1["f1"]

    if prefix_relative_bin_metrics is not None and len(prefix_relative_bin_metrics) > 0:
        rb = prefix_relative_bin_metrics.copy()

        if "auroc" in rb.columns and rb["auroc"].notna().any():
            best_bin_auroc = rb.loc[rb["auroc"].idxmax()]
            row["best_relative_bin_by_auroc"] = best_bin_auroc["relative_bin"]
            row["best_relative_bin_auroc"] = best_bin_auroc["auroc"]
            row["best_relative_bin_mean_prefix"] = best_bin_auroc.get(
                "mean_relative_prefix",
                np.nan,
            )

    return pd.DataFrame([row])


# =========================
# Split handling
# =========================

def load_eval_step_df(args):
    """
    mode:
      train / dev / val / test / all / ood
    """
    if args.mode == "ood":
        if args.ood_table_path is None:
            raise ValueError(
                "OOD evaluation requires --ood_table_path containing representations."
            )
        return load_step_table(args.ood_table_path)

    table_path = args.table_path
    if table_path is None:
        table_path = os.path.join(C.OUT_DIR, "step_df.pkl")

    step_df = load_step_table(table_path)

    split_file = args.split_path
    if split_file is None:
        split_file = split_path()

    if args.mode == "all":
        return step_df

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    split = load_json(split_file)

    if args.mode == "train":
        ids = set(map(int, split["train_ids"]))
    elif args.mode == "val":
        if "val_ids" in split:
            ids = set(map(int, split["val_ids"]))
        elif "dev_ids" in split:
            print("[warning] split has no val_ids; falling back to dev_ids.")
            ids = set(map(int, split["dev_ids"]))
        else:
            raise ValueError("Requested --mode val but split has no val_ids/dev_ids.")
    elif args.mode == "dev":
        if "dev_ids" in split:
            ids = set(map(int, split["dev_ids"]))
        elif "val_ids" in split:
            print("[warning] split has no dev_ids; falling back to val_ids.")
            ids = set(map(int, split["val_ids"]))
        else:
            raise ValueError("Requested --mode dev but split has no dev_ids/val_ids.")
    elif args.mode == "test":
        if "test_ids" in split:
            ids = set(map(int, split["test_ids"]))
        elif args.allow_dev_as_test:
            print(
                "[warning] split has no test_ids. Using dev_ids/val_ids as test because "
                "--allow_dev_as_test was set. Do not report this as untouched test."
            )
            fallback_key = "dev_ids" if "dev_ids" in split else "val_ids"
            ids = set(map(int, split[fallback_key]))
        else:
            raise ValueError(
                "Requested --mode test, but split file has no test_ids. "
                "Create a train/val/test split or use --allow_dev_as_test for debugging."
            )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return step_df[step_df["traj_id"].isin(ids)].copy()


# =========================
# Threshold calibration
# =========================

def make_loader_from_step_df(step_df: pd.DataFrame, batch_size: int, require_labels: bool):
    examples = build_examples(step_df, require_labels=require_labels)
    if len(examples) == 0:
        raise ValueError("No examples found for loader.")
    loader = DataLoader(
        TrajDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return examples, loader


def collect_traj_scores_only(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    ys = []
    ps = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            out = model(x, mask)

            y = batch["traj_y"].detach().cpu().numpy()
            p = out["traj_score"].detach().cpu().numpy()

            valid = y >= 0
            ys.extend(y[valid].astype(int).tolist())
            ps.extend(p[valid].tolist())

    return np.array(ys), np.array(ps)


def calibrate_threshold_if_needed(args, model):
    """
    If --threshold is provided, use it.
    Else if --calibrate / --calibrate_table_path is provided, tune threshold.
    Else use 0.5.
    """
    if args.threshold is not None:
        return float(args.threshold), {
            "source": "manual",
            "threshold": float(args.threshold),
        }

    if not args.calibrate and args.calibrate_table_path is None:
        return 0.5, {
            "source": "default",
            "threshold": 0.5,
        }

    if args.calibrate_table_path is not None:
        cal_step_df = load_step_table(args.calibrate_table_path)
        source = args.calibrate_table_path
    else:
        old_mode = args.mode
        args.mode = args.calibrate_split
        cal_step_df = load_eval_step_df(args)
        args.mode = old_mode
        source = args.calibrate_split

    _, cal_loader = make_loader_from_step_df(
        cal_step_df,
        batch_size=C.BATCH_SIZE,
        require_labels=True,
    )

    y, p = collect_traj_scores_only(model, cal_loader, C.DEVICE)

    if len(y) == 0:
        return 0.5, {
            "source": "calibration_failed_no_labels",
            "threshold": 0.5,
        }

    t, score = find_best_threshold(y, p, metric=args.threshold_metric)

    return float(t), {
        "source": source,
        "threshold_metric": args.threshold_metric,
        "threshold": float(t),
        "calibration_score": float(score),
        "n_calibration": int(len(y)),
    }


# =========================
# Optional OOD metadata merge
# =========================

def attach_ood_metadata(pred_traj_df, raw_jsonl_path: Optional[str]):
    if raw_jsonl_path is None or not os.path.exists(raw_jsonl_path):
        return pred_traj_df

    rows = load_jsonl(raw_jsonl_path)
    if len(rows) == 0:
        return pred_traj_df

    meta = {}

    for i, obj in enumerate(rows):
        possible_ids = [
            obj.get("traj_id"),
            obj.get("trajectory_id"),
            obj.get("id"),
            obj.get("idx"),
            obj.get("index"),
        ]

        tid = None
        for v in possible_ids:
            if v is not None:
                try:
                    tid = int(v)
                    break
                except Exception:
                    pass

        if tid is None:
            tid = i

        meta[int(tid)] = obj

    pred = pred_traj_df.copy()
    pred["raw_json"] = pred["traj_id"].map(
        lambda tid: json.dumps(meta.get(int(tid), {}), ensure_ascii=False)
    )

    return pred


# =========================
# Main evaluation
# =========================

def run_evaluation(args):
    ensure_config_defaults()
    set_seed(C.SEED)
    ensure_dir(args.out_dir)

    step_df = load_eval_step_df(args)

    require_labels = args.require_labels
    if args.mode in {"train", "dev", "val", "test", "all"}:
        require_labels = True

    examples = build_examples(step_df, require_labels=require_labels)

    if len(examples) == 0:
        raise ValueError("No examples found after filtering.")

    input_dim = examples[0].x_seq.shape[1]

    model, model_ckpt, stage_a_path = load_model(args, input_dim=input_dim)

    loader = DataLoader(
        TrajDataset(examples),
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    threshold, threshold_info = calibrate_threshold_if_needed(args, model)

    print(f"[eval] mode={args.mode}")
    print(f"[eval] n_traj={len(examples)}")
    print(f"[eval] threshold={threshold:.6f}")
    print(f"[eval] threshold_info={threshold_info}")
    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] stage_a={stage_a_path}")
    print(f"[eval] config model={model_ckpt.get('config', {}).get('model', 'unknown')}")
    print(f"[eval] main score=head=final cumulative state q_T")
    print(f"[eval] local evidence head is auxiliary and saved separately")

    y_list = [ex.traj_y for ex in examples if ex.traj_y is not None]
    if len(y_list) > 0:
        print("[eval] example-level traj_y counts:")
        print(pd.Series(y_list).value_counts(dropna=False))

    traj_pred_df, step_score_df = collect_full_predictions(model, loader, C.DEVICE)

    if args.mode == "ood":
        traj_pred_df = attach_ood_metadata(traj_pred_df, args.ood_jsonl)

    prefix_frac_df = collect_prefix_fraction_predictions(
        model,
        loader,
        C.DEVICE,
        fractions=args.prefix_fracs,
    )

    prefix_step_df = collect_prefix_predictions_all_steps(
        model,
        loader,
        C.DEVICE,
    )

    metrics = {
        "mode": args.mode,
        "ood_name": args.ood_name,
        "n_traj": int(len(traj_pred_df)),
        "n_steps": int(len(step_score_df)),
        "threshold_info": threshold_info,
        "ckpt": args.ckpt,
        "stage_a_path": stage_a_path,
        "main_score_definition": "final cumulative state head q_T",
        "prefix_score_definition": "final cumulative state head q_t after truncating prefix",
        "local_evidence_definition": "auxiliary local evidence head e_t and evidence MIL score",
        "model_config": model_ckpt.get("config", {}),
    }

    if traj_pred_df["traj_y"].notna().any():
        valid = traj_pred_df["traj_y"].notna()
        y = traj_pred_df.loc[valid, "traj_y"].astype(int).values
        p = traj_pred_df.loc[valid, "traj_score"].values

        metrics["full_traj"] = binary_metrics(y, p, threshold)

        best_f1_t, best_f1 = find_best_threshold(y, p, metric="f1")
        best_acc_t, best_acc = find_best_threshold(y, p, metric="acc")
        best_recall_t, best_recall = find_best_threshold(y, p, metric="recall")

        metrics["full_traj_oracle_thresholds"] = {
            "best_f1_threshold_oracle": float(best_f1_t),
            "best_f1_oracle": float(best_f1),
            "best_acc_threshold_oracle": float(best_acc_t),
            "best_acc_oracle": float(best_acc),
            "best_recall_threshold_oracle": float(best_recall_t),
            "best_recall_oracle": float(best_recall),
        }

        if "evidence_mil_score" in traj_pred_df.columns:
            e = traj_pred_df.loc[valid, "evidence_mil_score"].values
            metrics["full_traj_evidence_mil_aux"] = binary_metrics(y, e, threshold)
    else:
        metrics["full_traj"] = {}

    prefix_frac_metrics = metrics_by_prefix_fraction(prefix_frac_df, threshold)
    prefix_abs_step_metrics = metrics_by_absolute_prefix_step(prefix_step_df, threshold)
    prefix_relative_bin_metrics = metrics_by_relative_bins(
        prefix_step_df,
        threshold,
        bin_size=args.relative_bin_size,
    )

    ew_metrics, first_detect_df, stable_detect_df = early_warning_metrics(
        prefix_step_df,
        threshold,
    )

    metrics["early_warning"] = ew_metrics

    aggregate_summary_df = build_aggregate_summary(
        args=args,
        metrics=metrics,
        prefix_frac_metrics=prefix_frac_metrics,
        prefix_abs_step_metrics=prefix_abs_step_metrics,
        prefix_relative_bin_metrics=prefix_relative_bin_metrics,
        threshold=threshold,
    )

    # Save outputs
    prefix = args.out_prefix
    base = os.path.join(args.out_dir, prefix)

    traj_pred_path = f"{base}_traj_predictions.csv"
    step_score_path = f"{base}_step_scores.csv"
    prefix_frac_path = f"{base}_prefix_fraction_predictions.csv"
    prefix_step_path = f"{base}_prefix_step_predictions.csv"

    prefix_frac_metrics_path = f"{base}_prefix_fraction_metrics.csv"
    prefix_abs_step_metrics_path = f"{base}_prefix_abs_step_metrics.csv"
    prefix_relative_bin_metrics_path = f"{base}_prefix_relative_bin_metrics.csv"

    first_detect_path = f"{base}_first_detection.csv"
    stable_detect_path = f"{base}_stable_detection.csv"
    metrics_path = f"{base}_metrics.json"
    aggregate_summary_path = f"{base}_aggregate_summary.csv"

    traj_pred_df.to_csv(traj_pred_path, index=False)
    step_score_df.to_csv(step_score_path, index=False)
    prefix_frac_df.to_csv(prefix_frac_path, index=False)
    prefix_step_df.to_csv(prefix_step_path, index=False)

    prefix_frac_metrics.to_csv(prefix_frac_metrics_path, index=False)
    prefix_abs_step_metrics.to_csv(prefix_abs_step_metrics_path, index=False)
    prefix_relative_bin_metrics.to_csv(prefix_relative_bin_metrics_path, index=False)

    first_detect_df.to_csv(first_detect_path, index=False)
    stable_detect_df.to_csv(stable_detect_path, index=False)

    save_json(metrics, metrics_path)
    aggregate_summary_df.to_csv(aggregate_summary_path, index=False)

    print("=" * 80)
    print("[Full trajectory metrics: final cumulative state q_T]")
    print(json.dumps(metrics["full_traj"], indent=2, ensure_ascii=False))
    print("=" * 80)
    print("[Auxiliary evidence MIL metrics, if labels exist]")
    print(json.dumps(metrics.get("full_traj_evidence_mil_aux", {}), indent=2, ensure_ascii=False))
    print("=" * 80)
    print("[Oracle full-trajectory thresholds]")
    print(json.dumps(metrics.get("full_traj_oracle_thresholds", {}), indent=2, ensure_ascii=False))
    print("=" * 80)
    print("[Early warning metrics]")
    print(json.dumps(metrics["early_warning"], indent=2, ensure_ascii=False))
    print("=" * 80)
    print("[Prefix fraction metrics]")
    print(prefix_frac_metrics)
    print("=" * 80)
    print("[Aggregate summary]")
    print(aggregate_summary_df)
    print("=" * 80)
    print("[saved]")
    for p in [
        traj_pred_path,
        step_score_path,
        prefix_frac_path,
        prefix_step_path,
        prefix_frac_metrics_path,
        prefix_abs_step_metrics_path,
        prefix_relative_bin_metrics_path,
        first_detect_path,
        stable_detect_path,
        metrics_path,
        aggregate_summary_path,
    ]:
        print(p)


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "dev", "val", "test", "all", "ood"],
    )

    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        required=True,
        help="Stage-B V3 checkpoint, e.g. trace_trajectory_state_v3.pt",
    )

    ap.add_argument(
        "--stage_a_ckpt",
        type=str,
        default=None,
        help="Optional Stage-A checkpoint. If omitted, use stage_a_path in Stage-B ckpt or C.OUT_DIR/stage_a_latent_mechanism.pt.",
    )

    ap.add_argument(
        "--table_path",
        type=str,
        default=None,
        help="In-domain step_df path. Default: C.OUT_DIR/step_df.pkl",
    )

    ap.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Split json path. Default: split_path() from stage_a_latent_mechanism.py",
    )

    ap.add_argument(
        "--allow_dev_as_test",
        action="store_true",
        help="For debugging only: if split has no test_ids, use dev_ids/val_ids as test.",
    )

    ap.add_argument(
        "--ood_name",
        type=str,
        default=None,
        help="Name of OOD benchmark, e.g. asse, rjudge, agentdojo.",
    )

    ap.add_argument(
        "--ood_table_path",
        type=str,
        default=None,
        help="OOD event_repr_table / step table path.",
    )

    ap.add_argument(
        "--ood_jsonl",
        type=str,
        default=None,
        help="Optional raw OOD jsonl for attaching metadata to trajectory prediction CSV.",
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: C.OUT_DIR/eval_state_v3",
    )

    ap.add_argument(
        "--out_prefix",
        type=str,
        default="eval",
        help="Prefix for output CSV/JSON files.",
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual decision threshold. If omitted, use 0.5 unless --calibrate is set.",
    )

    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Tune threshold on --calibrate_split using in-domain table.",
    )

    ap.add_argument(
        "--calibrate_split",
        type=str,
        default="val",
        choices=["train", "dev", "val", "test"],
        help="Split used for threshold calibration when --calibrate is set.",
    )

    ap.add_argument(
        "--calibrate_table_path",
        type=str,
        default=None,
        help="External table for threshold calibration. Overrides --calibrate_split.",
    )

    ap.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "acc", "recall"],
        help="Metric used to choose calibrated threshold.",
    )

    ap.add_argument(
        "--prefix_fracs",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Prefix fractions for compact prefix evaluation.",
    )

    ap.add_argument(
        "--relative_bin_size",
        type=float,
        default=0.1,
        help="Bin size for relative-prefix metrics.",
    )

    ap.add_argument(
        "--require_labels",
        action="store_true",
        help="Require trajectory labels. Automatically true for in-domain split modes.",
    )

    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(C.OUT_DIR, "eval_state_v3")

    return args


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
