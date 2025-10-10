#!/usr/bin/env python3
# Author: Jason D, 10/06/2025

import subprocess
import os
import csv
import argparse
import re
from typing import Dict, List
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib")


# ----------------------------
# Read git commit data
# ----------------------------
def run_shortlog_all() -> list[tuple[str, int]]:
    proc = subprocess.run(
        ["git", "shortlog", "-sne", "--all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True
    )
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    out = []
    for ln in lines:
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        cnt_str, name = parts
        try:
            cnt = int(cnt_str)
        except ValueError:
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


# ----------------------------
# Assign tiers
# ----------------------------
def assign_fixed_tiers(data: list[tuple[str, int]]) -> list[tuple[str, int, str]]:
    top_15 = data[:15]
    tiers = ["Gold"] * 5 + ["Silver"] * 5 + ["Bronze"] * 5
    return [(name, commits, tier) for (name, commits), tier in zip(top_15, tiers)]


# ----------------------------
# Save CSV
# ----------------------------
def save_csv(rows: list[tuple[str, int, str]], outpath: str):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["author", "commits", "tier"])
        for row in rows:
            w.writerow(row)


def devList() -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "shortlog", "-sne", "--all"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", check=True
        )
    except subprocess.CalledProcessError:
        return []

    exclude = {
        "3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>",
    }

    developers = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^\s*\d+\s+(?P<author>.+)$', line)
        if match:
            author = match.group("author").strip()
            if author not in exclude:
                developers.append(author)
    return developers


def ticketsByDev_map() -> Dict[str, List[str]]:
    """
    Returns a mapping author -> sorted list of unique ticket IDs.
    Recognizes JIRA-like (ABC-123) and GitHub issues (#123).
    """
    pretty = "%x1e%an <%ae>%x1f%s%x1f%b"
    try:
        proc = subprocess.run(
            ["git", "log", "--all", f"--pretty=format:{pretty}"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", check=True
        )
    except subprocess.CalledProcessError:
        return {}

    raw = proc.stdout or ""
    commits = raw.split("\x1e")  # split by commit

    jira_re = re.compile(r'\b([A-Za-z]{2,}-\d+)\b', re.IGNORECASE)
    hash_re = re.compile(r'(?<![A-Za-z0-9])#\d+\b')

    author_to_ticketset = defaultdict(set)

    for entry in commits:
        entry = entry.strip()
        if not entry:
            continue

        parts = entry.split("\x1f", 2)
        author = parts[0].strip()
        subject = parts[1] if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""
        msg = (subject + "\n" + body).strip()

        found = set()
        for m in jira_re.findall(msg):
            found.add(m.upper())
        for m in hash_re.findall(msg):
            found.add(m)

        # ensures author exists even if no tickets
        author_to_ticketset[author]  # defaultdict auto-inits
        author_to_ticketset[author].update(found)

    # convert sets -> sorted lists
    result: Dict[str, List[str]] = {a: sorted(list(ts)) for a, ts in author_to_ticketset.items()}
    return result



def ticketsByDev_text() -> str:
    """
    Return a human-readable text block suitable for TextArea.
    Format: "Author Name <email>: TICKET-1, #23, TICKET-5"
    One author per line, authors sorted by number of tickets (desc).
    """
    m = ticketsByDev_map()
    if not m:
        return "No tickets found."

    exclude = {
        "3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>"
    }

    lines = []
    # sort authors by number of tickets desc, then by name
    for author, tickets in sorted(m.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
        if author in exclude:
            continue  # skip this author
        lines.append(f"{author}: {', '.join(tickets)}")
    return "\n".join(lines)


# ----------------------------
# Plot individual tier bar chart
# ----------------------------
def plot_single_tier(rows: list[tuple[str, int, str]], tier: str, outpath: str):

    color_map = {"Gold": "#D4AF37", "Silver": "#C0C0C0", "Bronze": "#CD7F32"}
    data = [r for r in rows if r[2] == tier]

    if not data:
        print(f"No data to plot for {tier}")
        return

    # Extract just the name before any email or < >
    def extract_name(full):
        # Removes email parts like <email@domain.com> or (email@domain.com)
        name = re.split(r"[<(]", full)[0].strip()
        return name

    names = [extract_name(r[0]) for r in data]
    counts = [r[1] for r in data]
    color = color_map.get(tier, "#888888")

    plt.figure(figsize=(max(10, len(data) * 2.2), 7))

    bars = plt.bar(names, counts, color=color, width=0.6)

    plt.xticks(rotation=24, ha="right", fontsize=20)   # Smaller font for names
    plt.yticks(fontsize=14)
    plt.ylabel("Number of Commits", fontsize=18)
    plt.xlabel("Top Contributors", fontsize=14, labelpad=40)
    #plt.title(f"{tier} Tier", fontsize=36, weight="bold")

    # Add value labels with more vertical padding and smaller font size
    for i, b in enumerate(bars):
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            h - 0.05 * h,  # slightly below the top
            str(counts[i]),
            ha="center", va="top",
            fontsize=16, weight="bold", color="white"
        )

    plt.tight_layout(pad=2.0)  # Add padding to avoid clipping
    plt.savefig(outpath, dpi=150)
    plt.close()




# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Gold, Silver, and Bronze contributor charts.")
    parser.add_argument("--out-dir", type=str, default="commit_tiers_output", help="Directory to save outputs.")
    args = parser.parse_args()

    data = run_shortlog_all()
    if not data:
        raise SystemExit("No contributors found. Run inside a Git repository.")

    # Optional exclusion
    exclude = ["3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>"]
    data = [(n, c) for (n, c) in data if n not in exclude]

    # Tiered top 15 contributors
    tiered = assign_fixed_tiers(data)

    os.makedirs(args.out_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(args.out_dir, "top15_contributors_tiers.csv")
    save_csv(tiered, csv_path)

    # Save tier-specific bar charts
    for tier in ["Gold", "Silver", "Bronze"]:
        chart_path = os.path.join(args.out_dir, f"{tier.lower()}_contributors.png")
        plot_single_tier(tiered, tier, chart_path)

    # Summary
    print(f"\nWrote CSV -> {csv_path}")
    for tier in ["Gold", "Silver", "Bronze"]:
        chart_file = os.path.join(args.out_dir, f"{tier.lower()}_contributors.png")
        print(f"Wrote {tier} chart -> {chart_file}")

    print("\nSummary:")
    for tier in ["Gold", "Silver", "Bronze"]:
        members = [r for r in tiered if r[2] == tier]
        total = sum(r[1] for r in members)
        print(f"  {tier:6}: {len(members)} contributors, {total} commits")


if __name__ == "__main__":
    main()
