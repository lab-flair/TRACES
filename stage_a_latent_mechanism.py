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

from sklearn.model_selection import train_test_split

import config_trace_prefix as C


# =========================
# Utils
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_or_none(x, mp):
    if pd.isna(x):
        return None
    return mp.get(x, x)


def split_path():
    return os.path.join(
        C.OUT_DIR,
        getattr(C, "SPLIT_FILENAME", "trace_train_val_test_split_ids.json"),
    )


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Data loading
# =========================

def load_step_df():
    df = pd.read_pickle(C.TABLE_PATH).copy()

    df = df[df["event_type"] == C.EVENT_TYPE].copy()
    df = df[df["layer"].isin(C.LAYERS)].copy()

    df["risk_source_coarse"] = df["risk_source"].map(
        lambda x: map_or_none(x, C.SOURCE_MAP)
    )
    df["failure_mode_coarse"] = df["failure_mode"].map(
        lambda x: map_or_none(x, C.FAILURE_MAP)
    )
    df["harm_coarse"] = df["real_world_harm"].map(
        lambda x: map_or_none(x, C.HARM_MAP)
    )

    for col in ["event_idx", "event_order_among_decisions", "label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []

    for (tid, eidx), g in df.groupby(["traj_id", "event_idx"], sort=False):
        row0 = g.iloc[0]

        layer_to_rep = {
            int(r["layer"]): r["rep"]
            for _, r in g.iterrows()
        }

        if not all(layer in layer_to_rep for layer in C.LAYERS):
            continue

        x = np.concatenate(
            [layer_to_rep[layer] for layer in C.LAYERS],
            axis=0,
        ).astype(np.float32)

        traj_label = int(row0["label"])

        rows.append({
            "traj_id": int(tid),
            "event_idx": int(eidx),
            "event_order": int(row0["event_order_among_decisions"]),
            "label": traj_label,
            "risk_source_coarse": (
                row0["risk_source_coarse"]
                if traj_label == 1
                else C.SAFE_TOKEN
            ),
            "failure_mode_coarse": (
                row0["failure_mode_coarse"]
                if traj_label == 1
                else C.SAFE_TOKEN
            ),
            "harm_coarse": (
                row0["harm_coarse"]
                if traj_label == 1
                else C.SAFE_TOKEN
            ),
            "x": x,
        })

    step_df = pd.DataFrame(rows)
    step_df = step_df.sort_values(
        ["traj_id", "event_order"]
    ).reset_index(drop=True)

    return step_df


def create_or_load_split(
    step_df,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
    overwrite=False,
):
    """
    Create a stratified train/val/test split at trajectory level.

    The split is balanced by trajectory-level safe/unsafe label.
    Stage A and Stage B should train only on train_ids.
    Stage B should select best checkpoint on val_ids.
    Test_ids are reserved for final evaluation only.
    """
    path = split_path()

    if os.path.exists(path) and not overwrite:
        split = load_json(path)

        train_ids = set(map(int, split["train_ids"]))
        val_ids = set(map(int, split["val_ids"]))
        test_ids = set(map(int, split["test_ids"]))

        print(f"[split] Loaded existing train/val/test split from {path}")
        print(
            f"[split] train trajs={len(train_ids)}, "
            f"val trajs={len(val_ids)}, "
            f"test trajs={len(test_ids)}"
        )

        return train_ids, val_ids, test_ids

    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total
    test_ratio = test_ratio / total

    traj_df = step_df.groupby("traj_id").agg(
        label=("label", "first"),
    ).reset_index()

    traj_ids = traj_df["traj_id"].values
    labels = traj_df["label"].values

    if getattr(C, "SPLIT_STRATIFY_BY_LABEL", True):
        stratify_main = labels
    else:
        stratify_main = None

    train_val_ids, test_ids, train_val_y, test_y = train_test_split(
        traj_ids,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_main,
    )

    val_ratio_within_train_val = val_ratio / (train_ratio + val_ratio)

    if getattr(C, "SPLIT_STRATIFY_BY_LABEL", True):
        stratify_second = train_val_y
    else:
        stratify_second = None

    train_ids, val_ids, train_y, val_y = train_test_split(
        train_val_ids,
        train_val_y,
        test_size=val_ratio_within_train_val,
        random_state=seed,
        stratify=stratify_second,
    )

    split = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "stratify_by_label": bool(getattr(C, "SPLIT_STRATIFY_BY_LABEL", True)),
        "train_ids": [int(x) for x in train_ids],
        "val_ids": [int(x) for x in val_ids],
        "test_ids": [int(x) for x in test_ids],
    }

    save_json(split, path)

    train_ids = set(map(int, train_ids))
    val_ids = set(map(int, val_ids))
    test_ids = set(map(int, test_ids))

    def _print_balance(name, ids):
        sub = traj_df[traj_df["traj_id"].isin(ids)]
        n = len(sub)
        pos = int((sub["label"] == 1).sum())
        neg = int((sub["label"] == 0).sum())
        pos_rate = pos / max(n, 1)

        print(
            f"[split:{name}] n={n}, safe={neg}, unsafe={pos}, "
            f"unsafe_rate={pos_rate:.3f}"
        )

    print(f"[split] Created train/val/test split at {path}")
    _print_balance("train", train_ids)
    _print_balance("val", val_ids)
    _print_balance("test", test_ids)

    return train_ids, val_ids, test_ids


class StepDataset(Dataset):
    """
    Stage A does not use step labels.
    It only learns latent center+subspace mechanisms from train steps.
    """

    def __init__(self, step_df):
        self.x = np.stack(step_df["x"].values).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx])


# =========================
# Stage-A model: center + subspace mechanism bank
# =========================

class MechanismDiscovery(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, C.Z_DIM),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.Z_DIM, C.Z_DIM),
        )

        self.centers = nn.Parameter(
            torch.randn(C.K_MECH, C.Z_DIM) * 0.02
        )

        self.subspaces = nn.Parameter(
            torch.randn(C.K_MECH, C.Z_DIM, C.SUBSPACE_RANK) * 0.02
        )

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def normalized_subspaces(self):
        q_list = []

        for k in range(C.K_MECH):
            q, _ = torch.linalg.qr(self.subspaces[k], mode="reduced")
            q_list.append(q)

        return torch.stack(q_list, dim=0)

    def match(self, z):
        centers = self.centers
        U = self.normalized_subspaces()

        z_n = F.normalize(z, dim=-1)
        c_n = F.normalize(centers, dim=-1)

        center_score = z_n @ c_n.t()

        residual = z.unsqueeze(1) - centers.unsqueeze(0)
        proj = torch.einsum("bkz,kzr->bkr", residual, U)
        subspace_score = torch.norm(proj, dim=-1)

        alpha = F.softplus(self.log_alpha)
        beta = F.softplus(self.log_beta)

        scores = alpha * center_score + beta * subspace_score
        gates = F.softmax(scores, dim=-1)

        return gates, scores, U

    def reconstruct(self, z, gates, U):
        residual = z.unsqueeze(1) - self.centers.unsqueeze(0)

        coeff = torch.einsum("bkz,kzr->bkr", residual, U)

        recon_each = self.centers.unsqueeze(0) + torch.einsum(
            "bkr,kzr->bkz",
            coeff,
            U,
        )

        recon = (gates.unsqueeze(-1) * recon_each).sum(dim=1)

        return recon

    def forward(self, x):
        z = self.encoder(x)
        gates, scores, U = self.match(z)
        recon = self.reconstruct(z, gates, U)

        return {
            "z": z,
            "gates": gates,
            "scores": scores,
            "recon": recon,
            "U": U,
        }

    def diversity_loss(self):
        c = F.normalize(self.centers, dim=-1)
        gram = c @ c.t()
        eye = torch.eye(C.K_MECH, device=gram.device)

        center_loss = ((gram - eye) ** 2).mean()

        U = self.normalized_subspaces()

        sub_loss = 0.0
        count = 0

        for i in range(C.K_MECH):
            for j in range(i + 1, C.K_MECH):
                cross = U[i].t() @ U[j]
                sub_loss = sub_loss + (cross ** 2).mean()
                count += 1

        return center_loss + sub_loss / max(count, 1)


def gate_entropy_loss(gates):
    entropy = -(gates * torch.log(gates + C.EPS)).sum(dim=-1)
    return entropy.mean()


def coverage_loss(gates):
    usage = gates.mean(dim=0)
    return -torch.log(usage + C.EPS).mean()


# =========================
# Train Stage A
# =========================

def train_stage_a(args):
    set_seed(C.SEED)
    C.ensure_dir(C.OUT_DIR)

    step_df = load_step_df()

    step_df.to_pickle(os.path.join(C.OUT_DIR, "step_df.pkl"))
    step_df.to_pickle(os.path.join(C.OUT_DIR, "step_df_full.pkl"))

    train_ids, val_ids, test_ids = create_or_load_split(
        step_df,
        train_ratio=getattr(C, "TRAIN_RATIO", 0.6),
        val_ratio=getattr(C, "VAL_RATIO", 0.2),
        test_ratio=getattr(C, "TEST_RATIO", 0.2),
        seed=C.SEED,
        overwrite=args.overwrite_split,
    )

    train_df = step_df[step_df["traj_id"].isin(train_ids)].copy()
    val_df = step_df[step_df["traj_id"].isin(val_ids)].copy()
    test_df = step_df[step_df["traj_id"].isin(test_ids)].copy()

    train_df.to_pickle(os.path.join(C.OUT_DIR, "stage_a_train_step_df.pkl"))
    val_df.to_pickle(os.path.join(C.OUT_DIR, "stage_a_val_step_df.pkl"))
    test_df.to_pickle(os.path.join(C.OUT_DIR, "stage_a_test_step_df.pkl"))

    print(f"[data] full steps={len(step_df)}")
    print(f"[data] train steps={len(train_df)}")
    print(f"[data] val steps={len(val_df)}")
    print(f"[data] test steps={len(test_df)}")
    print(f"[data] train trajs={train_df['traj_id'].nunique()}")
    print(f"[data] val trajs={val_df['traj_id'].nunique()}")
    print(f"[data] test trajs={test_df['traj_id'].nunique()}")

    dataset = StepDataset(train_df)

    loader = DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    input_dim = dataset[0].shape[0]

    model = MechanismDiscovery(input_dim).to(C.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=C.LR_STAGE_A,
        weight_decay=C.WEIGHT_DECAY,
    )

    history = []

    for epoch in range(1, C.EPOCHS_STAGE_A + 1):
        model.train()

        total = 0.0
        comp = {
            "recon": 0.0,
            "sparse": 0.0,
            "diverse": 0.0,
            "coverage": 0.0,
        }

        for x in loader:
            x = x.to(C.DEVICE)

            out = model(x)

            loss_recon = F.mse_loss(out["recon"], out["z"])
            loss_sparse = gate_entropy_loss(out["gates"])
            loss_diverse = model.diversity_loss()
            loss_coverage = coverage_loss(out["gates"])

            loss = (
                C.LAMBDA_A_RECON * loss_recon
                + C.LAMBDA_A_SPARSE * loss_sparse
                + C.LAMBDA_A_DIVERSE * loss_diverse
                + C.LAMBDA_A_COVERAGE * loss_coverage
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item())
            comp["recon"] += float(loss_recon.item())
            comp["sparse"] += float(loss_sparse.item())
            comp["diverse"] += float(loss_diverse.item())
            comp["coverage"] += float(loss_coverage.item())

        n = max(len(loader), 1)

        row = {
            "epoch": epoch,
            "loss": total / n,
            **{k: v / n for k, v in comp.items()},
        }

        history.append(row)
        print(row)

    ckpt_path = os.path.join(C.OUT_DIR, "stage_a_latent_mechanism.pt")

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "split_path": split_path(),
            "trained_on": "train_ids_only",
            "stage_a_objective": "latent_mechanism_discovery_no_step_bce",
            "config": {
                "z_dim": C.Z_DIM,
                "k_mech": C.K_MECH,
                "subspace_rank": C.SUBSPACE_RANK,
                "split_type": "train_val_test",
            },
        },
        ckpt_path,
    )

    pd.DataFrame(history).to_csv(
        os.path.join(C.OUT_DIR, "stage_a_latent_history.csv"),
        index=False,
    )

    print(f"Saved Stage A latent mechanism bank to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_split", action="store_true")
    args = parser.parse_args()

    train_stage_a(args)