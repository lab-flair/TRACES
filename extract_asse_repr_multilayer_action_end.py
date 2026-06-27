import os
import json
import argparse
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "../models/qwen3-4b/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DEFAULT_INPUT_JSONL = "./analysis_asse_events/asse_trajectories.jsonl"

DEFAULT_OUTPUT_NPZ = "./qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end.npz"
DEFAULT_OUTPUT_META = "./qwen3-4b_analysis_asse_events/asse_qwen3_repr_multilayer_action_end_meta.json"

'''
DEFAULT_MODEL_ID = "../models/llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
DEFAULT_INPUT_JSONL = "./analysis_asse_events/asse_trajectories.jsonl"

DEFAULT_OUTPUT_NPZ = "./llama3_analysis_asse_events/asse_llama3_repr_multilayer_action_end.npz"
DEFAULT_OUTPUT_META = "./llama3_analysis_asse_events/asse_llama3_repr_multilayer_action_end_meta.json"
'''

DEFAULT_LAYERS = [20, 24, 28, 32, 36]
DEFAULT_TARGET_EVENT_TYPES = ["agent_action", "agent_complete"]


def load_trajectories(jsonl_path):
    trajs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajs.append(json.loads(line))
    return trajs


def load_existing_npz(npz_path):
    repr_dict = {}
    if os.path.exists(npz_path):
        existing = np.load(npz_path)
        for k in existing.keys():
            repr_dict[k] = existing[k]
    return repr_dict


def save_npz_safe(npz_path, repr_dict):
    tmp_path = npz_path + ".tmp.npz"
    np.savez(tmp_path, **repr_dict)
    if os.path.exists(npz_path):
        os.remove(npz_path)
    os.rename(tmp_path, npz_path)


def event_to_message(event):
    etype = event.get("event_type")
    text = event.get("text", "") or ""

    if etype == "user":
        return {"role": "user", "content": text}

    if etype == "environment":
        return {"role": "user", "content": f"[TOOL RESULT]\n{text}"}

    if etype == "agent_thought":
        return {"role": "assistant", "content": f"Thought: {text}"}

    if etype == "agent_action":
        return {"role": "assistant", "content": f"Action: {text}"}

    if etype == "agent_complete":
        return {"role": "assistant", "content": text if text else "[DONE]"}

    return {"role": "user", "content": text}


def merge_consecutive_same_role(messages):
    if not messages:
        return messages

    merged = [dict(messages[0])]

    for m in messages[1:]:
        if m["role"] == merged[-1]["role"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + m["content"]
        else:
            merged.append(dict(m))

    return merged


def build_messages_through_event(events, target_idx, tool_pool=None):
    msgs = []

    if tool_pool:
        tool_lines = []

        for t in tool_pool:
            name = t.get("name", "")
            desc = (t.get("description", "") or "").strip().split("\n")[0][:200]
            if name:
                tool_lines.append(f"- {name}: {desc}")

        if tool_lines:
            sys_text = (
                "You are an AI agent with access to the following tools:\n"
                + "\n".join(tool_lines)
            )
            msgs.append({"role": "system", "content": sys_text})

    for i in range(target_idx + 1):
        msgs.append(event_to_message(events[i]))

    msgs = merge_consecutive_same_role(msgs)
    return msgs


def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-layer Llama hidden states at the END of ATBench agent events"
    )

    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)

    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=DEFAULT_INPUT_JSONL,
        help="atbench_trajectories.jsonl produced by build_atbench_event_dataset.py",
    )

    parser.add_argument(
        "--atbench_raw_json",
        type=str,
        default="./ATBench/ATBench/test.json",
        help="Raw ATBench json, used only to recover per-trajectory tool_used",
    )

    parser.add_argument("--output_npz", type=str, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--output_meta", type=str, default=DEFAULT_OUTPUT_META)

    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)

    parser.add_argument(
        "--target_event_types",
        type=str,
        nargs="+",
        default=DEFAULT_TARGET_EVENT_TYPES,
        choices=[
            "agent_action",
            "agent_complete",
            "agent_thought",
            "user",
            "environment",
        ],
        help="Which event types to extract representations for",
    )

    parser.add_argument(
        "--include_tool_pool",
        action="store_true",
        default=True,
        help="Prepend tool_used as a system message",
    )
    parser.add_argument("--no_tool_pool", dest="include_tool_pool", action="store_false")

    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=7500,
        help="Truncate prompt from the left if longer than this, keeping recent history",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=-1)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_meta) or ".", exist_ok=True)

    print(f"[*] Loading model: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading trajectories: {args.input_jsonl}")

    trajs = load_trajectories(args.input_jsonl)

    print(f"    {len(trajs)} trajectories")

    tool_pool_by_id = {}

    if args.include_tool_pool and os.path.exists(args.atbench_raw_json):
        raw_size = os.path.getsize(args.atbench_raw_json)
        print(f"    raw ATBench json: {args.atbench_raw_json} ({raw_size} bytes)")

        try:
            with open(args.atbench_raw_json, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for s in raw:
                tool_pool_by_id[s["id"]] = s.get("tool_used", [])

            print(f"    loaded tool pools for {len(tool_pool_by_id)} trajectories")

        except json.JSONDecodeError as ex:
            print(
                f"[!] raw ATBench json is corrupted "
                f"(JSONDecodeError at char {ex.pos}); continuing WITHOUT tool_pool."
            )

    elif args.include_tool_pool:
        print(
            f"[!] raw ATBench json not found at {args.atbench_raw_json}; "
            "falling back to empty tool pool"
        )

    if args.max_samples != -1:
        trajs = trajs[: args.max_samples]
        print(f"    truncated to {len(trajs)}")

    repr_dict = {}

    if args.resume and os.path.exists(args.output_npz):
        print(f"[*] Resume from {args.output_npz}")
        repr_dict = load_existing_npz(args.output_npz)
        print(f"    existing repr points: {len(repr_dict)}")

    target_set = set(args.target_event_types)

    total_targets = 0
    event_type_counter = Counter()

    for traj in trajs:
        for e in traj.get("events", []):
            event_type_counter[e["event_type"]] += 1
            if e["event_type"] in target_set:
                total_targets += 1

    print(f"[*] target event types: {sorted(target_set)}")
    print(f"[*] total target events: {total_targets}")
    print(f"[*] layers: {args.layers}")
    print("[*] extraction position: action_end / post_event")
    print("    representation = hidden state at the end of history + target event")

    processed_events = 0
    newly_added_points = 0
    truncated_prompts = 0
    skipped_empty_target_text = 0

    pbar = tqdm(trajs, desc="ATBench action-end repr", total=len(trajs))

    for traj in pbar:
        tid = traj["id"]
        events = traj.get("events", [])

        if not events:
            continue

        tool_pool = tool_pool_by_id.get(tid, []) if args.include_tool_pool else None

        for e in events:
            if e["event_type"] not in target_set:
                continue

            eidx = e["event_idx"]

            all_exist = all(
                f"{tid}_event_{eidx}_{layer}" in repr_dict for layer in args.layers
            )

            if all_exist:
                continue

            if not (e.get("text", "") or "").strip():
                skipped_empty_target_text += 1

            msgs = build_messages_through_event(
                events,
                eidx,
                tool_pool=tool_pool,
            )

            if not msgs:
                msgs = [{"role": "user", "content": ""}]

            try:
                input_ids = tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )

            except Exception as ex:
                tqdm.write(
                    f"[skip] chat_template failed on traj {tid} event {eidx}: {ex}"
                )
                continue

            if input_ids.shape[1] > args.max_prompt_tokens:
                input_ids = input_ids[:, -args.max_prompt_tokens:]
                truncated_prompts += 1

            input_ids = input_ids.to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            for layer in args.layers:
                key = f"{tid}_event_{eidx}_{layer}"

                if key in repr_dict:
                    continue

                h = (
                    outputs.hidden_states[layer][0, -1, :]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )

                repr_dict[key] = h
                newly_added_points += 1

            del outputs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            processed_events += 1

            if processed_events % (args.save_every * 10) == 0:
                save_npz_safe(args.output_npz, repr_dict)
                tqdm.write(f"[save] {args.output_npz}  points={len(repr_dict)}")

        pbar.set_postfix(
            {
                "events": processed_events,
                "points": newly_added_points,
                "truncated": truncated_prompts,
            }
        )

    save_npz_safe(args.output_npz, repr_dict)

    meta = {
        "model_id": args.model_id,
        "input_jsonl": args.input_jsonl,
        "output_npz": args.output_npz,
        "layers": args.layers,
        "target_event_types": list(target_set),
        "include_tool_pool": args.include_tool_pool,
        "max_prompt_tokens": args.max_prompt_tokens,
        "n_trajectories": len(trajs),
        "event_type_counts_all": dict(event_type_counter),
        "target_events_total": total_targets,
        "repr_points": len(repr_dict),
        "truncated_prompts": truncated_prompts,
        "empty_target_text_events": skipped_empty_target_text,
        "repr_position": "action_end",
        "definition": "hidden state at the final token after including the target event itself",
        "chat_template_add_generation_prompt": False,
        "key_format": "{traj_id}_event_{event_idx}_{layer}",
    }

    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("[done] ATBench action-end representation extraction complete")
    print(f"  npz:             {args.output_npz}")
    print(f"  meta:            {args.output_meta}")
    print(f"  repr points:     {len(repr_dict)}")

    if repr_dict:
        any_key = next(iter(repr_dict.keys()))
        print(f"  hidden dim:      {repr_dict[any_key].shape}")

    print(f"  truncated:       {truncated_prompts}")
    print(f"  empty targets:   {skipped_empty_target_text}")
    print("  repr position:   action_end / post_event")
    print("=" * 60)


if __name__ == "__main__":
    main()

