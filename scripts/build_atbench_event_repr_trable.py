# build_diagnostic_table.py

import json
import numpy as np
import pandas as pd
from pathlib import Path

NPZ_PATH = "qwen3-4b_analysis_atbench_events/atbench_qwen_repr_multilayer_action_end.npz"
JSONL_PATH = "./analysis_atbench_events/atbench_trajectories.jsonl"
OUT_PATH = "qwen3-4b_analysis_atbench_events/event_repr_table_action_end.pkl"   

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_event_metadata(trajs):
    event_meta = {}
    for traj in trajs:
        tid = traj["id"]
        label = traj["label"]
        risk_source = traj.get("risk_source")
        failure_mode = traj.get("failure_mode")
        real_world_harm = traj.get("real_world_harm")
        risk_step_event_idx = traj.get("risk_step_event_idx", -1)

        action_event_idxs = [e["event_idx"] for e in traj["events"] if e["event_type"] == "agent_action"]
        complete_event_idxs = [e["event_idx"] for e in traj["events"] if e["event_type"] == "agent_complete"]
        last_action_idx = max(action_event_idxs) if action_event_idxs else None
        last_complete_idx = max(complete_event_idxs) if complete_event_idxs else None

        # 给 decision events 编序号
        decision_types = {"agent_action", "agent_complete"}
        decision_events = [e for e in traj["events"] if e["event_type"] in decision_types]
        decision_events = sorted(decision_events, key=lambda x: x["event_idx"])
        decision_order = {e["event_idx"]: i for i, e in enumerate(decision_events)}

        for e in traj["events"]:
            eidx = e["event_idx"]
            event_meta[(tid, eidx)] = {
                "traj_id": tid,
                "label": label,
                "risk_source": risk_source,
                "failure_mode": failure_mode,
                "real_world_harm": real_world_harm,
                "risk_step_event_idx": risk_step_event_idx,
                "event_idx": eidx,
                "event_type": e.get("event_type"),
                "turn_idx": e.get("turn_idx"),
                "tool_name": e.get("tool_name"),
                "tool_has_side_effect": e.get("tool_has_side_effect"),
                "arg_from_prev_env": e.get("arg_from_prev_env"),
                "tool_desc_injection_marker": e.get("tool_desc_injection_marker"),
                "is_likely_risk_step": e.get("is_likely_risk_step", 0),
                "dist_to_risk_step": (eidx - risk_step_event_idx) if risk_step_event_idx is not None else None,
                "is_before_risk_step": int(risk_step_event_idx is not None and eidx < risk_step_event_idx),
                "is_at_risk_step": int(risk_step_event_idx is not None and eidx == risk_step_event_idx),
                "is_after_risk_step": int(risk_step_event_idx is not None and eidx > risk_step_event_idx),
                "event_order_among_decisions": decision_order.get(eidx, None),
                "is_last_agent_action": int(last_action_idx is not None and eidx == last_action_idx),
                "is_last_agent_complete": int(last_complete_idx is not None and eidx == last_complete_idx),
            }
    return event_meta

def parse_npz_key(key):
    # 形如: {id}_event_{event_idx}_{layer}
    parts = key.split("_event_")
    tid = int(parts[0])
    rest = parts[1]
    eidx_str, layer_str = rest.rsplit("_", 1)
    return tid, int(eidx_str), int(layer_str)

def main():
    trajs = load_jsonl(JSONL_PATH)
    event_meta = build_event_metadata(trajs)

    arrs = np.load(NPZ_PATH)
    rows = []

    for key in arrs.files:
        tid, eidx, layer = parse_npz_key(key)
        meta = event_meta.get((tid, eidx))
        if meta is None:
            continue

        row = dict(meta)
        row["layer"] = layer
        row["rep"] = arrs[key].astype(np.float32)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_pickle(OUT_PATH)

    print(f"saved: {OUT_PATH}")
    print(f"shape: {df.shape}")
    print(df.head())
    print(df[['event_type', 'layer']].value_counts().head(20))

if __name__ == "__main__":
    main()