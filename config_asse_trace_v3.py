import os
import torch


# =========================
# Paths
# =========================

'''
TABLE_PATH = "llama3_analysis_asse_events/asse_event_repr_table.pkl"
TRAJ_JSONL_PATH = "analysis_asse_events/asse_trajectories.jsonl"
OUT_DIR = "llama3_analysis_asse_events/traces_asse_outputs"
'''


TABLE_PATH = "qwen3-4b_analysis_asse_events/asse_event_repr_table.pkl"
TRAJ_JSONL_PATH = "analysis_asse_events/asse_trajectories.jsonl"
OUT_DIR = "qwen3-4b_analysis_asse_events/traces_asse_outputs"

# =========================
# Basic setup
# =========================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Data filtering
# =========================

EVENT_TYPE = "agent_action"

# Qwen3-4b
LAYERS = [32]

# Llama3-8b
# LAYERS = [30] 

DROP_NON_CONSTANT_LABEL_TRAJS = True


# =========================
# Train / Val / Test split
# =========================

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

SPLIT_FILENAME = "asse_train_val_test_split_ids.json"


# =========================
# Training
# =========================

BATCH_SIZE = 32

EPOCHS_STAGE_A = 30
EPOCHS_STAGE_B = 30

LR_STAGE_A = 5e-4
LR_STAGE_B = 5e-4

# fallback
LR = 5e-4

WEIGHT_DECAY = 1e-4
DROPOUT = 0.1


# =========================
# Model dimensions
# =========================

Z_DIM = 256
HIDDEN_DIM = 256

K_MECH = 32
SUBSPACE_RANK = 8

EPS = 1e-8


# =========================
# Stage A: latent center-subspace mechanism bank
# =========================

LAMBDA_A_RECON = 1.0
LAMBDA_A_SPARSE = 0.01
LAMBDA_A_DIVERSE = 0.01
LAMBDA_A_COVERAGE = 0.01

# fallback names
LAMBDA_RECON = LAMBDA_A_RECON
LAMBDA_SPARSE = LAMBDA_A_SPARSE
LAMBDA_DIVERSE = LAMBDA_A_DIVERSE
LAMBDA_COVERAGE = LAMBDA_A_COVERAGE


# =========================
# Stage B V3: trajectory-state auditor
# =========================

TRACE_USE_RAW_PROJ = True
TRACE_RAW_DIM = 256

TRACE_USE_STAGE_A = True
TRACE_USE_MECH_Z = True
TRACE_USE_MECH_GATES = True
TRACE_USE_MECH_SCORES = True
TRACE_USE_DELTA_FEATURES = True

TRACE_FREEZE_STAGE_A = True

TRACE_GRU_HIDDEN = 256
TRACE_GRU_LAYERS = 1
TRACE_BIDIRECTIONAL = False

TRACE_MIL_TOPK = 3


# =========================
# Stage B V3 losses
# =========================

# Final cumulative trajectory-state objective
LAMBDA_STATE_FINAL = 1.0

# Local evidence MIL auxiliary objective
LAMBDA_EVIDENCE_MIL = 0

# Prefix-aware cumulative state loss
USE_PREFIX_LOSS = True
LAMBDA_PREFIX = 0.2
PREFIX_LOSS_RHO = 0.2
PREFIX_LOSS_GAMMA = 2.0

# Unsafe prefix ranking:
# encourage q_late > q_early
USE_PREFIX_RANK_LOSS = True
LAMBDA_PREFIX_RANK = 0.05
PREFIX_RANK_EARLY_FRAC = 0.4
PREFIX_RANK_LATE_FRAC = 0.8
PREFIX_RANK_MARGIN = 0.2

# Optional prefix stability loss
USE_PREFIX_STABLE_LOSS = False
LAMBDA_PREFIX_STABLE = 0.005
PREFIX_STABLE_MARGIN = 0.1

# Optional local evidence regularization
LAMBDA_EVIDENCE_SPARSE = 0.0
LAMBDA_EVIDENCE_SMOOTH = 0.0


# =========================
# Checkpoint selection
# =========================
EARLY_AWARE_SELECTION = True

SELECT_WEIGHT_FULL = 0.25
SELECT_WEIGHT_PREFIX40 = 0.25
SELECT_WEIGHT_PREFIX60 = 0.25
SELECT_WEIGHT_PREFIX80 = 0.25


# =========================
# Backward-compatible aliases
# =========================
LAMBDA_B_TRAJ = LAMBDA_STATE_FINAL
LAMBDA_B_RISK_SPARSE = LAMBDA_EVIDENCE_SPARSE
LAMBDA_B_RISK_SMOOTH = LAMBDA_EVIDENCE_SMOOTH

LAMBDA_B_GATE_BALANCE = 0.0
LAMBDA_B_GATE_SMOOTH = 0.0


# =========================
# Misc
# =========================

SAFE_TOKEN = "__SAFE__"


# =========================
# Helpers
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)