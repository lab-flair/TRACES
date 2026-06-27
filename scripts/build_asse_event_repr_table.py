#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="analysis_asse_events/asse_trajectories.jsonl")
    ap.add_argument("--npz", default="qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end.npz")
    ap.add_argument("--out", default="qwen3-4b_analysis_asse_events/asse_event_repr_table.pkl")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 24, 28, 32, 36])
    ap.add_argument("--target_event_types", nargs="+", default=["agent_action", "agent_complete"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    trajs = load_jsonl(args.jsonl)
    arr = np.load(args.npz)

    target_set = set(args.target_event_types)
    rows = []
    missing = 0

    for traj in trajs:
        tid = int(traj["id"])
        traj_label = int(traj["label"])

        risk_source = traj.get("risk_source", "unknown" if traj_label == 1 else "safe")
        failure_mode = traj.get("failure_mode", "unknown" if traj_label == 1 else "safe")
        real_world_harm = traj.get("real_world_harm", "unknown" if traj_label == 1 else "safe")

        order = 0

        for e in traj.get("events", []):
            if e["event_type"] not in target_set:
                continue

            eidx = int(e["event_idx"])
            step_label = int(e.get("is_likely_risk_step", 0))

            for layer in args.layers:
                key = f"{tid}_event_{eidx}_{layer}"

                if key not in arr:
                    missing += 1
                    continue

                rows.append({
                    "traj_id": tid,
                    "event_idx": eidx,
                    "event_order_among_decisions": order,
                    "event_type": e["event_type"],
                    "layer": layer,
                    "rep": arr[key].astype(np.float32),

                    # step-level weak label
                    "label": step_label,

                    # trajectory label
                    "traj_label": traj_label,

                    # ATBench-style fields
                    "risk_source": risk_source if traj_label == 1 else "safe",
                    "failure_mode": failure_mode if traj_label == 1 else "safe",
                    "real_world_harm": real_world_harm if traj_label == 1 else "safe",

                    # ASSE metadata
                    "benchmark": "asse",
                    "application_scenario": traj.get("application_scenario", ""),
                    "risk_type": traj.get("risk_type", ""),
                    "ambiguous": traj.get("ambiguous", None),
                    "reason": traj.get("reason", ""),
                })

            order += 1

    df = pd.DataFrame(rows)
    df.to_pickle(args.out)

    print("=" * 80)
    print("[done] ASSE event representation table built")
    print(f"output: {args.out}")
    print(f"rows: {len(df)}")
    print(f"missing repr points: {missing}")
    if len(df):
        print(df[["traj_id", "event_idx", "event_type", "layer", "label", "traj_label"]].head())
        print("event types:", df["event_type"].value_counts().to_dict())
        print("labels:", df["label"].value_counts().to_dict())
        print("traj labels:", df.groupby("traj_id")["traj_label"].first().value_counts().to_dict())
    print("=" * 80)


if __name__ == "__main__":
    main()