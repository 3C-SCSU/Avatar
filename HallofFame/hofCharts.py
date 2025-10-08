#!/usr/bin/env python3


from __future__ import annotations
import argparse
import os
import subprocess
import math
import csv
from collections import OrderedDict

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib")

# ------------------------------
# Utilities to read git counts
# ------------------------------
def run_shortlog_all() -> list[tuple[str,int]]:
    proc = subprocess.run(
        ["git", "shortlog", "-sne", "--all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True
    )
    if proc.stdout is None:
        # nothing captured
        raise RuntimeError("git shortlog produced no stdout")
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    out = []
    for ln in lines:
        # typical formats: "  123\tName <email>" or "  123 Name <email>"
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        cnt_str, name = parts
        try:
            cnt = int(cnt_str)
        except ValueError:
            # fallback to tab-split
            if '\t' in ln:
                left, right = ln.split('\t', 1)
                try:
                    cnt = int(left.strip())
                    name = right.strip()
                except Exception:
                    continue
            else:
                continue
        out.append((name.strip(), cnt))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ------------------------------
# Bucketing strategies
# ------------------------------
def cumulative_bucket(data: list[tuple[str,int]], gold_pct: float, silver_pct: float) -> list[tuple[str,int,str]]:
    """
    Cumulative-tiering:
      - gold_pct, silver_pct are fractions in [0,1] (gold < silver)
      - Gold = smallest top set whose cumulative commits >= gold_pct*total
      - Silver = smallest top set (continuing) until cumulative >= silver_pct*total
      - Bronze = rest
    Returns list of tuples (author, commits, tier), in the same order as data (desc).
    """
    if not (0 < gold_pct < silver_pct <= 1.0):
        raise ValueError("gold_pct and silver_pct must satisfy 0 < gold_pct < silver_pct <= 1")

    total = sum(c for _, c in data)
    if total == 0:
        return [(name, c, "Bronze") for name, c in data]

    gold_cut = gold_pct * total
    silver_cut = silver_pct * total

    cum = 0
    out = []
    for name, c in data:
        cum += c
        if cum <= gold_cut:
            tier = "Gold"
        elif cum <= silver_cut:
            tier = "Silver"
        else:
            tier = "Bronze"
        out.append((name, c, tier))

    # ensure at least one gold if there are contributors
    if out and all(t != "Gold" for _, _, t in out):
        n0, c0, _ = out[0]
        out[0] = (n0, c0, "Gold")
    return out

def quantile_bucket(data: list[tuple[str,int]]) -> list[tuple[str,int,str]]:
    """
    Split authors into three roughly equal-sized groups by rank:
      top 1/3 -> Gold, middle 1/3 -> Silver, bottom 1/3 -> Bronze
    """
    n = len(data)
    if n == 0:
        return []
    third = math.ceil(n / 3)
    out = []
    for i, (name, c) in enumerate(data):
        if i < third:
            t = "Gold"
        elif i < 2*third:
            t = "Silver"
        else:
            t = "Bronze"
        out.append((name, c, t))
    return out

# ------------------------------
# Filter and pipeline
# ------------------------------
def filter_min_commits(data: list[tuple[str,int]], min_commits: int) -> list[tuple[str,int]]:
    """Return only authors with commits > min_commits (ignore <= min_commits)."""
    return [(n,c) for (n,c) in data if c > min_commits]

# ------------------------------
# IO: CSV + plots
# ------------------------------
def save_csv(rows: list[tuple[str,int,str]], outpath: str):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["author","commits","tier"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])

def plot_bar(rows: list[tuple[str,int,str]], outpath: str, top_n: int = 30):
    color_map = {"Gold":"#D4AF37", "Silver":"#C0C0C0", "Bronze":"#CD7F32"}
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)[:top_n]
    names = [r[0] for r in rows_sorted][::-1]
    counts = [r[1] for r in rows_sorted][::-1]
    tiers = [r[2] for r in rows_sorted][::-1]
    colors = [color_map.get(t, "#888888") for t in tiers]

    plt.figure(figsize=(10, max(4, 0.3*len(names))))
    bars = plt.barh(range(len(names)), counts, color=colors)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Commits")
    plt.title(f"Top {min(top_n, len(rows))} contributors (filtered and tiered)")
    for i, b in enumerate(bars):
        w = b.get_width()
        plt.text(w + max(1, 0.01*w), b.get_y() + b.get_height()/2, str(counts[i]), va="center", fontsize=8)
    # legend
    from matplotlib.patches import Patch
    legend_items = [Patch(color=color_map[k], label=k) for k in ["Gold","Silver","Bronze"]]
    plt.legend(handles=legend_items, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_donut(rows: list[tuple[str,int,str]], outpath: str):
    totals = {"Gold":0,"Silver":0,"Bronze":0}
    for _, c, t in rows:
        totals[t] = totals.get(t,0) + c
    labels = []
    sizes = []
    colors = {"Gold":"#D4AF37","Silver":"#C0C0C0","Bronze":"#CD7F32"}
    for k in ["Gold","Silver","Bronze"]:
        if totals[k] > 0:
            labels.append(k)
            sizes.append(totals[k])
    if not sizes:
        # nothing to plot
        return
    plt.figure(figsize=(6,6))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
                                       colors=[colors[l] for l in labels], wedgeprops=dict(width=0.4))
    plt.title("Commit share by tier")
    plt.savefig(outpath, dpi=150)
    plt.close()

# ------------------------------
# Main CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tier contributors (Gold/Silver/Bronze) ignoring low-activity contributors.")
    ap.add_argument("--min-commits", type=int, default=8,
                    help="Ignore contributors with commits <= this number (default: 8).")
    ap.add_argument("--gold-pct", type=float, default=60.0,
                    help="Gold threshold as percent (0-100). For cumulative method: gold covers this pct of commits. Default 60.")
    ap.add_argument("--silver-pct", type=float, default=85.0,
                    help="Gold+Silver threshold as percent (0-100). Default 85.")
    ap.add_argument("--method", choices=["cumulative","quantile"], default="cumulative",
                    help="Bucket method. Default 'cumulative' (percentage-based).")
    ap.add_argument("--top", type=int, default=30, help="Top-N contributors to show in bar chart (default 30).")
    ap.add_argument("--out-dir", type=str, default="commit_tiers_output", help="Output directory.")
    ap.add_argument("--limit", type=int, default=0, help="Limit to top-N contributors (after filtering). 0=no limit")
    args = ap.parse_args()

    # Validate percentages
    if not (0 < args.gold_pct < args.silver_pct <= 100):
        raise SystemExit("gold-pct and silver-pct must satisfy: 0 < gold-pct < silver-pct <= 100")

    data = run_shortlog_all()
    if not data:
        raise SystemExit("No contributors found. Run from a git repository root.")

    exclude = ["3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>"]
    data = [(n, c) for (n, c) in data if n not in exclude]

    # Filter by min_commits (ignore contributors with commits <= min_commits)
    filtered = filter_min_commits(data, args.min_commits)
    if not filtered:
        print(f"No contributors remain after filtering with --min-commits {args.min_commits}. Exiting.")
        return

    # Optionally apply limit
    if args.limit and args.limit > 0:
        filtered = filtered[:args.limit]

    # Choose bucketing
    if args.method == "quantile":
        rows = quantile_bucket(filtered)
    else:
        gold_frac = args.gold_pct / 100.0
        silver_frac = args.silver_pct / 100.0
        rows = cumulative_bucket(filtered, gold_frac, silver_frac)

    # Save & plot
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "commits_by_author_tiers_filtered.csv")
    print("Current working directory:", os.getcwd())

    save_csv(rows, csv_path)

    bar_path = os.path.join(args.out_dir, f"commits_tiers_bar_top{args.top}.png")
    plot_bar(rows, bar_path, top_n=args.top)

    donut_path = os.path.join(args.out_dir, "commits_tiers_donut.png")
    plot_donut(rows, donut_path)

    # Summary print
    total_commits = sum(c for _, c, _ in rows)
    totals_by_tier = {}
    counts_by_tier = {}
    for _, c, t in rows:
        totals_by_tier[t] = totals_by_tier.get(t, 0) + c
        counts_by_tier[t] = counts_by_tier.get(t, 0) + 1

    print(f"Wrote CSV -> {csv_path}")
    print(f"Wrote bar  -> {bar_path}")
    print(f"Wrote donut-> {donut_path}")
    print("\nSummary (after filtering):")
    for t in ["Gold","Silver","Bronze"]:
        commits = totals_by_tier.get(t, 0)
        people = counts_by_tier.get(t, 0)
        pct = (commits / total_commits * 100) if total_commits else 0.0
        print(f"  {t:6s}: {people:3d} contributors, {commits:6d} commits ({pct:.1f}%)")

if __name__ == "__main__":
    main()
