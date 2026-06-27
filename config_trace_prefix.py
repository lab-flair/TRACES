import os
import torch


# =========================
# Paths
# =========================

#TABLE_PATH = "atbench_qwen3_4b/event_repr_table_action_end.pkl"
#OUT_DIR = "atbench_qwen3_4b/traces_prefix_outputs"

TABLE_PATH = "qwen3-4b_analysis_asse_events/asse_event_repr_table.pkl"
OUT_DIR = "qwen3-4b_analysis_asse_events/traces_asse_outputs"

# For ATBench, you may leave it unused.
TRAJ_JSONL_PATH = "./analysis_asse_events/asse_trajectories.jsonl"


# =========================
# Basic setup
# =========================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Split
# =========================

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

SPLIT_FILENAME = "trace_train_val_test_split_ids.json"


# =========================
# Data filtering
# =========================

EVENT_TYPE = "agent_action"

# Important: must match your Stage-A training setup.
# If Stage A was trained with [32], Stage B/eval must also use [32].
LAYERS = [32]


# =========================
# Training
# =========================

BATCH_SIZE = 32

EPOCHS_STAGE_A = 30
EPOCHS_STAGE_B = 30

LR_STAGE_A = 5e-4
LR_STAGE_B = 5e-4

WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

EPS = 1e-8


# =========================
# Stage A: center-subspace mechanism bank
# =========================

Z_DIM = 256
K_MECH = 32
SUBSPACE_RANK = 8

# Stage A unsupervised / weak mechanism discovery losses
LAMBDA_A_RECON = 1.0
LAMBDA_A_SPARSE = 0.01
LAMBDA_A_DIVERSE = 0.01
LAMBDA_A_COVERAGE = 0.01


# =========================
# Stage B: temporal MIL auditor
# =========================

TRACE_USE_RAW_PROJ = True
TRACE_RAW_DIM = 256

TRACE_USE_MECH_Z = True
TRACE_USE_MECH_GATES = True
TRACE_USE_MECH_SCORES = True

TRACE_FREEZE_STAGE_A = True

TRACE_GRU_HIDDEN = 256
TRACE_GRU_LAYERS = 1
TRACE_BIDIRECTIONAL = False

TRACE_MIL_TOPK = 3


# =========================
# Stage B main losses
# =========================

# Full trajectory-level MIL loss
LAMBDA_B_TRAJ = 1.0

# Usually keep these 0 during debugging.
# You can enable later if step scores become too dense/noisy.
LAMBDA_B_RISK_SPARSE = 0.0
LAMBDA_B_RISK_SMOOTH = 0.0


# =========================
# Prefix-aware MIL loss
# =========================

USE_PREFIX_LOSS = True

# Main prefix loss weight.
# Start conservative.
LAMBDA_PREFIX = 0.2

# Unsafe prefix warm-up.
# For unsafe trajectories, prefixes before rho are not forced to be unsafe.
PREFIX_LOSS_RHO = 0.2

# Increasing weight after rho.
# gamma=2 means later prefixes are much more strongly supervised.
PREFIX_LOSS_GAMMA = 2.0


# =========================
# Unsafe prefix ranking loss
# =========================

USE_PREFIX_RANK_LOSS = True

# Encourage late unsafe prefixes to score higher than early unsafe prefixes.
LAMBDA_PREFIX_RANK = 0.05

PREFIX_RANK_EARLY_FRAC = 0.4
PREFIX_RANK_LATE_FRAC = 0.8
PREFIX_RANK_MARGIN = 0.2


# =========================
# Optional prefix stability loss
# =========================

# Start with False. Enable only if prefix predictions flicker.
USE_PREFIX_STABLE_LOSS = False

LAMBDA_PREFIX_STABLE = 0.005
PREFIX_STABLE_MARGIN = 0.1

TRACE_USE_DELTA_FEATURES = True

LAMBDA_STATE_FINAL = 1.0
LAMBDA_EVIDENCE_MIL = 0.3

USE_PREFIX_LOSS = True
LAMBDA_PREFIX = 0.2
PREFIX_LOSS_RHO = 0.2
PREFIX_LOSS_GAMMA = 2.0

USE_PREFIX_RANK_LOSS = True
LAMBDA_PREFIX_RANK = 0.05
PREFIX_RANK_EARLY_FRAC = 0.4
PREFIX_RANK_LATE_FRAC = 0.8
PREFIX_RANK_MARGIN = 0.2

USE_PREFIX_STABLE_LOSS = False
LAMBDA_PREFIX_STABLE = 0.005
PREFIX_STABLE_MARGIN = 0.1

LAMBDA_EVIDENCE_SPARSE = 0.0
LAMBDA_EVIDENCE_SMOOTH = 0.0

EARLY_AWARE_SELECTION = True
SELECT_WEIGHT_FULL = 0.25
SELECT_WEIGHT_PREFIX40 = 0.25
SELECT_WEIGHT_PREFIX60 = 0.25
SELECT_WEIGHT_PREFIX80 = 0.25
# =========================
# Misc
# =========================

SAFE_TOKEN = "__SAFE__"



# =========================
# Taxonomy mappings
# Not required by this Stage B, but kept for compatibility with Stage A data processing.
# =========================

SOURCE_MAP = {
    "malicious_user_instruction_or_jailbreak": "user",
    "direct_prompt_injection": "env_or_injection",
    "indirect_prompt_injection": "env_or_injection",
    "unreliable_or_misinformation": "env_or_injection",

    "tool_description_injection": "tool",
    "malicious_tool_execution": "tool",
    "corrupted_tool_feedback": "tool",

    "inherent_agent_or_llm_failures": "inherent_agent_failures",
}


FAILURE_MAP = {
    "unconfirmed_or_over_privileged_action": "action_failure",

    "flawed_planning_or_reasoning": "reasoning_failure",
    "procedural_deviation_or_inaction": "reasoning_failure",
    "inefficient_or_wasteful_execution": "reasoning_failure",

    "improper_tool_use": "tool_failure",
    "insecure_interaction_or_execution": "tool_failure",

    "generation_of_harmful_or_offensive_content": "content_failure",
    "generation_of_malicious_executables": "content_failure",
    "unauthorized_information_disclosure": "content_failure",
    "provide_inaccurate_misleading_or_unverified_information": "content_failure",

    "instruction_for_harmful_or_illegal_activity": "instruction_for_harmful_illegal_activity",

    "choosing_malicious_tool": "choosing_malicious_tool",
    "incorrect_tool_parameters": "incorrect_tool_parameters",
    "tool_misuse_in_specific_context": "tool_misuse_in_specific_context",
    "failure_to_validate_tool_outputs": "failure_to_validate_tool_outputs",

    "action_failure": "action_failure",
    "reasoning_failure": "reasoning_failure",
    "tool_failure": "tool_failure",
    "content_failure": "content_failure",
}


HARM_MAP = {
    "privacy_and_confidentiality_harm": "privacy_security",
    "security_and_system_integrity_harm": "privacy_security",

    "financial_and_economic_harm": "economic",

    "physical_and_health_harm": "physical",

    "public_service_and_resource_harm": "societal_functional",
    "info_ecosystem_and_societal_harm": "societal_functional",
    "functional_and_opportunity_harm": "societal_functional",

    "reputational_and_interpersonal_harm": "social_psych",
    "psychological_and_emotional_harm": "social_psych",
    "fairness_equity_and_allocative_harm": "social_psych",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)