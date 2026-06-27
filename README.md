# Trajectory Representation for Agent-Centric Early Safety

This repository contains the pipeline for extracting action-end representations, building event-level representation tables, training trace-level auditors, and evaluating the final models on ASSE and ATBench.

The pipeline is organized into five stages:

1. Extract multi-layer action-end representations.
2. Build event-level representation tables.
3. Run Stage-A trace preprocessing.
4. Train Stage-B temporal trace auditor.
5. Evaluate the trained checkpoint and read aggregate metrics.

---

## 1. ASSE Pipeline

### Step 1: Extract action-end representations

```bash
python extract_asse_repr_multilayer_action_end.py \
  --input_jsonl analysis_asse_events/asse_trajectories.jsonl \
  --output_npz qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end.npz \
  --output_meta qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end_meta.json \
  --layers 20 24 28 32 36 \
  --target_event_types agent_action \
  --no_tool_pool \
  --resume
```

This extracts hidden states at the end of each `agent_action` event:

```text
representation = hidden_state(history before action_t + action_t)
```

The extracted representation file will be saved to:

```text
qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end.npz
```

The metadata file will be saved to:

```text
qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end_meta.json
```

---

### Step 2: Build ASSE event representation table

```bash
python scripts/build_asse_event_repr_table.py \
  --jsonl analysis_asse_events/asse_trajectories.jsonl \
  --npz qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end.npz \
  --out qwen3-4b_analysis_asse_events/asse_event_repr_table.pkl \
  --layers 20 24 28 32 36
```

This builds the event-level representation table used by the trace auditor.

Output:

```text
qwen3-4b_analysis_asse_events/asse_event_repr_table.pkl
```

---

### Step 3: Run ASSE Stage-A preprocessing

```bash
python stage_a_asse_trace.py --overwrite_split
```

---

### Step 4: Train ASSE Stage-B temporal MIL auditor

```bash
python stage_b_asse_trace_v3.py \
  --out_prefix asse_strict_trace_temporal_mil
```

Expected checkpoint:

```text
qwen3-4b_analysis_asse_events/traces_asse_outputs/asse_trace_temporal_mil.pt
```

---

### Step 5: Evaluate ASSE model

```bash
python evaluate_asse_trace.py \
  --mode test \
  --ckpt qwen3-4b_analysis_asse_events/traces_asse_outputs/asse_trace_temporal_mil.pt \
  --out_dir eval_outputs/asse_test \
  --out_prefix asse_test \
  --threshold 0.3
```

The aggregate evaluation summary can be found at:

```text
eval_outputs/asse_test/asse_test_aggregate_summary.csv
```

---

## 2. ATBench Pipeline

### Step 1: Extract action-end representations

```bash
python extract_atbench_repr_multilayer_action_end.py \
  --input_jsonl analysis_atbench_events/atbench_trajectories.jsonl \
  --output_npz qwen3-4b_analysis_atbench_events/atbench_qwen3_repr_multilayer_action_end.npz \
  --output_meta qwen3-4b_analysis_atbench_events/atbench_qwen3_repr_multilayer_action_end_meta.json \
  --layers 20 24 28 32 36 \
  --target_event_types agent_action \
  --no_tool_pool \
  --resume
```

This extracts hidden states at the end of each `agent_action` event:

```text
representation = hidden_state(history before action_t + action_t)
```

The extracted representation file will be saved to:

```text
qwen3-4b_analysis_atbench_events/atbench_qwen3_repr_multilayer_action_end.npz
```

The metadata file will be saved to:

```text
qwen3-4b_analysis_atbench_events/atbench_qwen3_repr_multilayer_action_end_meta.json
```

---

### Step 2: Build ATBench event representation table

```bash
python scripts/build_atbench_event_repr_table.py \
  --jsonl analysis_atbench_events/atbench_trajectories.jsonl \
  --npz qwen3-4b_analysis_atbench_events/atbench_qwen3_repr_multilayer_action_end.npz \
  --out qwen3-4b_analysis_atbench_events/atbench_event_repr_table.pkl \
  --layers 20 24 28 32 36
```

This builds the event-level representation table used by the trace auditor.

Output:

```text
qwen3-4b_analysis_atbench_events/atbench_event_repr_table.pkl
```

---

### Step 3: Run ATBench Stage-A preprocessing

```bash
python stage_a_atbench_trace.py --overwrite_split
```

---

### Step 4: Train ATBench Stage-B temporal MIL auditor

```bash
python stage_b_atbench_trace_v3.py \
  --out_prefix atbench_trace_temporal_mil_prefix
```

Expected checkpoint:

```text
atbench_qwen3_4b/traces_prefix_outputs_v3/atbench_trace_temporal_mil_prefix.pt
```

---

### Step 5: Evaluate ATBench model

```bash
python evaluate_atbench_trace.py \
  --mode test \
  --ckpt atbench_qwen3_4b/traces_prefix_outputs_v3/atbench_trace_temporal_mil_prefix.pt \
  --out_dir eval_outputs/atbench_test \
  --out_prefix atbench_test \
  --threshold 0.3
```

The aggregate evaluation summary can be found at:

```text
eval_outputs/atbench_test/atbench_test_aggregate_summary.csv
```

---

