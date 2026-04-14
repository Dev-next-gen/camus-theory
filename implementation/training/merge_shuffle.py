"""Merge OASST1 + code-feedback jsonl.gz with root-shuffled interleaving.

Groups messages by tree root, shuffles root order so the first N roots picked
by the graft loader give a ~target_ratio mix.
"""
import argparse, gzip, json, random
from collections import defaultdict


def load_msgs(path):
    msgs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            msgs.append(json.loads(line))
    return msgs


def group_by_root(msgs):
    by_id = {m["message_id"]: m for m in msgs}
    root_of = {}
    for m in msgs:
        cur = m
        while cur.get("parent_id") and cur["parent_id"] in by_id:
            cur = by_id[cur["parent_id"]]
        root_of[m["message_id"]] = cur["message_id"]
    groups = defaultdict(list)
    for m in msgs:
        groups[root_of[m["message_id"]]].append(m)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oasst", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio_code", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    print("Loading...")
    o_msgs = load_msgs(args.oasst)
    c_msgs = load_msgs(args.code)
    print(f"  oasst {len(o_msgs)} msgs, code {len(c_msgs)} msgs")

    o_groups = group_by_root(o_msgs)
    c_groups = group_by_root(c_msgs)
    random.shuffle(o_groups); random.shuffle(c_groups)
    print(f"  oasst {len(o_groups)} trees, code {len(c_groups)} trees")

    # Interleave by target ratio
    out_groups = []
    oi = ci = 0
    while oi < len(o_groups) or ci < len(c_groups):
        if random.random() < args.ratio_code and ci < len(c_groups):
            out_groups.append(c_groups[ci]); ci += 1
        elif oi < len(o_groups):
            out_groups.append(o_groups[oi]); oi += 1
        elif ci < len(c_groups):
            out_groups.append(c_groups[ci]); ci += 1

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for g in out_groups:
            for m in g:
                f.write(json.dumps(m) + "\n")
    print(f"Wrote {len(out_groups)} interleaved trees -> {args.out}")


if __name__ == "__main__":
    main()
