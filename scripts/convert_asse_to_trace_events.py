#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_asse_to_trace_events.py

Convert asse-safety.json into ATBench-style trajectory jsonl for TRACE.

Input:
  datasets/asse-safety.json

Output:
  analysis_asse_events/asse_trajectories.jsonl

The output format follows your ATBench-style trajectory:
  {
    id,
    label,
    risk_source,
    failure_mode,
    real_world_harm,
    reason,
    risk_step_event_idx,
    events: [
      {
        event_type: user / agent_action / environment / agent_complete,
        role,
        text,
        event_idx,
        is_likely_risk_step
      }
    ]
  }
"""

import os
import json
import argparse
from collections import Counter


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def flatten_contents(contents):
    """
    ASSE format:
      contents = [
        [
          {"role": "user", ...},
          {"role": "agent", "thought": ..., "action": ...},
          {"role": "environment", "content": ...}
        ],
        [
          ...
        ]
      ]

    We flatten all turns into one event sequence.
    """
    if contents is None:
        return []

    if isinstance(contents, dict):
        return [contents]

    if not isinstance(contents, list):
        return [{"role": "user", "content": str(contents)}]

    flat = []
    for item in contents:
        if isinstance(item, list):
            flat.extend(flatten_contents(item))
        else:
            flat.append(item)
    return flat


def normalize_role(role):
    role = str(role or "user").strip().lower()

    if role in {"agent", "assistant", "ai", "bot", "model"}:
        return "assistant"

    if role in {"environment", "env", "observation", "tool"}:
        return "environment"

    if role in {"human", "user"}:
        return "user"

    return role


def normalize_message(msg):
    """
    Normalize ASSE message to:
      {"role": "user/assistant/environment", "content": "..."}
    """
    if not isinstance(msg, dict):
        return {
            "role": "user",
            "content": "" if msg is None else str(msg),
            "raw": msg,
        }

    raw_role = msg.get("role", "user")
    role = normalize_role(raw_role)

    if str(raw_role).lower() == "agent":
        thought = msg.get("thought", None)
        action = msg.get("action", None)
        content = msg.get("content", None)

        parts = []
        if thought not in [None, ""]:
            parts.append(f"Thought: {thought}")
        if action not in [None, ""]:
            parts.append(str(action))
        if not parts and content not in [None, ""]:
            parts.append(str(content))

        return {
            "role": "assistant",
            "content": "\n".join(parts).strip(),
            "raw": msg,
        }

    content = msg.get("content", msg.get("text", msg.get("message", "")))
    if content is None:
        content = ""

    return {
        "role": role,
        "content": str(content),
        "raw": msg,
    }


def make_event(event_type, role, text, event_idx):
    return {
        "event_type": event_type,
        "role": role,
        "turn_idx": 1,
        "text": "" if text is None else str(text),

        # ATBench-compatible fields
        "tool_name": None,
        "tool_has_side_effect": False,
        "tool_desc_injection_marker": False,
        "tool_desc_injection_hits": [],
        "arg_from_prev_env": False,
        "arg_overlap_ratio": 0.0,
        "arg_matched_keys": [],

        "event_idx": event_idx,
        "is_likely_risk_step": 0,
    }


def guess_event_type(role, is_last=False):
    role = normalize_role(role)

    if role == "user":
        return "user"

    if role == "environment":
        return "environment"

    if role == "assistant":
        return "agent_complete" if is_last else "agent_action"

    return "user"


def convert_one(record, mark_last_k=1):
    rid = int(record["id"])
    label = int(record.get("label", 0))

    raw_contents = record.get("contents", [])
    messages = [normalize_message(x) for x in flatten_contents(raw_contents)]

    events = []
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        is_last = i == len(messages) - 1
        event_type = guess_event_type(role, is_last=is_last)

        events.append(
            make_event(
                event_type=event_type,
                role=role,
                text=content,
                event_idx=len(events),
            )
        )

    # fallback
    if not events:
        events.append(
            make_event(
                event_type="user",
                role="user",
                text="",
                event_idx=0,
            )
        )

    # ASSE does not provide exact risk_step_event_idx.
    # Weak label: for unsafe trajectory, mark the last K agent decisions as risk steps.
    risk_step_event_idx = None
    if label == 1:
        agent_indices = [
            e["event_idx"]
            for e in events
            if e["event_type"] in {"agent_action", "agent_complete"}
        ]

        if agent_indices:
            selected = agent_indices[-mark_last_k:]
            for e in events:
                if e["event_idx"] in selected:
                    e["is_likely_risk_step"] = 1
            risk_step_event_idx = selected[0]

    risk_type = record.get("risk_type", "unknown")
    failure_mode = record.get("failure_mode", "unknown")
    application_scenario = record.get("application_scenario", "unknown")
    reason = record.get("risk_description", "")

    return {
        "id": rid,
        "orig_id": rid,
        "benchmark": "asse",
        "profile": record.get("profile", ""),

        "label": label,

        # Map ASSE metadata into ATBench-style fields.
        # These are meaningful ASSE labels, but not exactly ATBench taxonomy.
        "risk_source": risk_type if label == 1 else "safe",
        "failure_mode": failure_mode if label == 1 else "safe",
        "real_world_harm": application_scenario if label == 1 else "safe",
        "reason": reason,

        "application_scenario": application_scenario,
        "risk_type": risk_type,
        "ambiguous": record.get("ambiguous", None),

        "num_events": len(events),
        "num_turns": len(raw_contents) if isinstance(raw_contents, list) else 1,
        "num_side_effect_calls": 0,
        "num_injection_tools": 0,
        "risk_step_event_idx": risk_step_event_idx,
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", default="benchmarks/asse-security.json")
    ap.add_argument("--output_jsonl", default="analysis_asse_events/asse_security_trajectories.jsonl")
    ap.add_argument("--mark_last_k", type=int, default=1)
    ap.add_argument("--max_records", type=int, default=-1)
    args = ap.parse_args()

    data = load_json(args.input_json)

    if args.max_records > 0:
        data = data[: args.max_records]

    rows = []
    label_counter = Counter()
    event_counter = Counter()
    risk_counter = Counter()
    scenario_counter = Counter()

    for r in data:
        out = convert_one(r, mark_last_k=args.mark_last_k)
        rows.append(out)

        label_counter[out["label"]] += 1
        risk_counter[out["risk_type"]] += 1
        scenario_counter[out["application_scenario"]] += 1

        for e in out["events"]:
            event_counter[e["event_type"]] += 1

    save_jsonl(rows, args.output_jsonl)

    print("=" * 80)
    print("[done] ASSE conversion complete")
    print(f"input:            {args.input_json}")
    print(f"output:           {args.output_jsonl}")
    print(f"num trajectories: {len(rows)}")
    print(f"labels:           {dict(label_counter)}")
    print(f"event types:      {dict(event_counter)}")
    print(f"risk types:       {dict(risk_counter)}")
    print(f"scenarios:        {dict(scenario_counter)}")
    print("=" * 80)

    if rows:
        ex = rows[0]
        print("\n[example]")
        print(
            json.dumps(
                {
                    "id": ex["id"],
                    "label": ex["label"],
                    "risk_source": ex["risk_source"],
                    "failure_mode": ex["failure_mode"],
                    "real_world_harm": ex["real_world_harm"],
                    "risk_step_event_idx": ex["risk_step_event_idx"],
                    "events_preview": ex["events"][:5],
                },
                ensure_ascii=False,
                indent=2,
            )[:3000]
        )


if __name__ == "__main__":
    main()