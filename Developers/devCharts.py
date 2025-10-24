#!/usr/bin/env python3
# Author: Jason D (converted to OOP by ChatGPT), 10/22/2025
"""
OOP/OOD refactor of original procedural script.
Usage: python pr_tier_analysis.py --out-dir commit_tiers_output
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib")


# ----- Configuration dataclass -----
@dataclass
class Config:
    repo: str
    token: str
    cache_file: Path
    cache_ttl: int = 60 * 60
    ticketlist_file: Path = Path(__file__).resolve().parent / "authorTicketList.json"
    ticketlist_ttl: int = 60 * 60

    @classmethod
    def from_env(cls, cache_file: Optional[Path] = None) -> "Config":
        load_dotenv("token.env")
        repo = os.getenv("GITHUB_REPO", "")
        token = os.getenv("GITHUB_TOKEN", "")
        if not cache_file:
            cache_file = Path(__file__).resolve().parent / "approved_prs_cache.json"
        return cls(repo=repo, token=token, cache_file=cache_file)


# ----- GitHub API Client -----
class GitHubClient:
    def __init__(self, repo: str, token: str, session: Optional[requests.Session] = None):
        if not repo or not token:
            raise ValueError("Missing repo or token for GitHubClient")
        self.repo = repo
        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "pr-tier-analyzer"
        })

    def list_closed_pulls(self, per_page: int = 100) -> List[Dict]:
        """
        Fetch all closed pull requests via paginated requests.
        """
        results = []
        page = 1
        while True:
            url = f"https://api.github.com/repos/{self.repo}/pulls"
            params = {"state": "closed", "per_page": per_page, "page": page}
            resp = self.session.get(url, params=params)
            if resp.status_code != 200:
                print(f"Failed to fetch page {page}. Status code: {resp.status_code}")
                break
            page_items = resp.json()
            if not page_items:
                break
            results.extend(page_items)
            print(f"Fetched {len(page_items)} pull requests from page {page}")
            page += 1
        print(f"Total closed pull requests fetched: {len(results)}")
        return results

    def pr_has_approval(self, pr_number: int) -> bool:
        """
        Return True if the given PR has at least one APPROVED review.
        """
        reviews_url = f"https://api.github.com/repos/{self.repo}/pulls/{pr_number}/reviews"
        resp = self.session.get(reviews_url)
        if resp.status_code != 200:
            return False
        for r in resp.json():
            if r.get("state", "").upper() == "APPROVED":
                return True
        return False


# ----- Cache Manager -----
class CacheManager:
    def __init__(self, path: Path, ttl_seconds: int):
        self.path = path
        self.ttl = ttl_seconds

    def load(self) -> Optional[List[Tuple[str, int]]]:
        if not self.path.exists():
            return None
        try:
            if time.time() - self.path.stat().st_mtime > self.ttl:
                return None
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [(entry["author"], entry["count"]) for entry in raw]
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def save(self, data: Iterable[Tuple[str, int]]) -> None:
        to_write = [{"author": a, "count": c} for a, c in data]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(to_write, fh, indent=2)
            print(f"Cache saved to {self.path}")
        except Exception as e:
            print(f"Error saving cache: {e}")


# ----- Ticket List Manager -----
class TicketListManager:
    def __init__(self, out_path: Path, ttl_seconds: int = 3600):
        self.out_path = out_path
        self.ttl = ttl_seconds

    def save(self, author_ticket_iterable: Iterable[Dict]) -> Path:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = []
        for entry in author_ticket_iterable:
            json_data.append({
                "author_name": entry.get("author_name"),
                "tickets": list(entry.get("tickets") or [])
            })
        try:
            with open(self.out_path, "w", encoding="utf-8") as fh:
                json.dump(json_data, fh, indent=2, ensure_ascii=False)
            print(f"Ticket list saved to {self.out_path}")
        except Exception as e:
            print(f"Error saving ticket list: {e}")
        return self.out_path

    def load(self) -> Optional[List[Dict]]:
        if not self.out_path.exists():
            return None
        try:
            if time.time() - self.out_path.stat().st_mtime > self.ttl:
                return None
            with open(self.out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"Loaded ticket list from {self.out_path}")
            return data
        except Exception as e:
            print(f"Error loading ticket list: {e}")
            return None


# ----- PR Analyzer -----
class PRAnalyzer:
    def __init__(self, gh: GitHubClient):
        self.gh = gh

    @staticmethod
    def extract_metadata(pr: Dict) -> Dict:
        pr_number = pr.get("number")
        pr_title = pr.get("title") or ""
        pr_body = pr.get("body") or ""
        pr_url = pr.get("html_url", "")
        author_login = pr.get("user", {}).get("login", "Unknown")
        author_name = author_login

        # find ticket-like patterns (e.g. #123)
        found_tickets = re.findall(r'#\d+', pr_title + " " + pr_body)
        if pr_number is not None:
            found_tickets.append(f"PR#{pr_number}")

        return {
            "pr_number": pr_number,
            "author_login": author_login,
            "author_name": author_name,
            "ticket_ids": found_tickets,
            "url": pr_url,
            "title": pr_title,
        }

    def ticket_list_by_author(self, prs: Iterable[Dict], filter_author: Optional[str] = None) -> List[Dict]:
        author_data = defaultdict(lambda: {"author_name": "", "tickets": set()})

        for pr in prs:
            meta = self.extract_metadata(pr)
            author_name = meta.get("author_name") or meta.get("author_login") or "Unknown"
            if filter_author:
                if filter_author not in {author_name, meta.get("author_login")}:
                    continue
            tid = meta.get("ticket_ids", [])
            key = author_name
            author_data[key]["author_name"] = author_name
            for t in tid:
                author_data[key]["tickets"].add(t)

        # preserve the previously present "j-knudson" manual tickets behavior
        extra_author = "j-knudson"
        extra_tickets = {f"MANUAL-{i + 1}" for i in range(19)}
        if extra_author in author_data:
            author_data[extra_author]["tickets"].update(extra_tickets)
        else:
            author_data[extra_author] = {"author_name": extra_author, "tickets": extra_tickets}

        result = []
        for name, info in author_data.items():
            result.append({
                "author_name": info["author_name"],
                "tickets": sorted(list(info["tickets"]))
            })
        return result

    @staticmethod
    def count_approved_prs_by_author(approved_prs: Iterable[Dict]) -> List[Tuple[str, int]]:
        author_counter = Counter()
        exclude = {"3C-SCSU", "dependabot[bot]", "github-actions[bot]"}

        for pr in approved_prs:
            author = pr.get("user", {}).get("login", "Unknown")
            if author in exclude:
                continue
            # keep the weird boost behavior from original script
            if "j-knudson" in author_counter:
                author_counter["j-knudson"] += 2
            else:
                author_counter[author] += 1

        return author_counter.most_common()

    @staticmethod
    def assign_fixed_tiers(data: List[Tuple[str, int]]) -> List[Tuple[str, int, str]]:
        top_15 = data[:15]
        tiers = ["Gold"] * 5 + ["Silver"] * 5 + ["Bronze"] * 5
        return [(name, tickets, tier) for (name, tickets), tier in zip(top_15, tiers)]


# ----- CSV Exporter -----
class CSVExporter:
    @staticmethod
    def save(rows: Iterable[Tuple[str, int, str]], outpath: str) -> None:
        os_dir = os.path.dirname(outpath) or "."
        Path(os_dir).mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["author", "tickets", "tier"])
            for row in rows:
                w.writerow(row)
        print(f"CSV saved to {outpath}")


# ----- Plotter -----
class Plotter:
    COLOR_MAP = {"Gold": "#D4AF37", "Silver": "#C0C0C0", "Bronze": "#CD7F32"}

    @staticmethod
    def _safe_name(full: str) -> str:
        return re.split(r"[<(]", full)[0].strip()

    @classmethod
    def plot_single_tier(cls, rows: Iterable[Tuple[str, int, str]], tier: str, outpath: str) -> None:
        data = [r for r in rows if r[2] == tier]
        if not data:
            print(f"No data to plot for {tier}")
            return

        names = [cls._safe_name(r[0]) for r in data]
        counts = [r[1] for r in data]
        color = cls.COLOR_MAP.get(tier, "#888888")

        plt.figure(figsize=(max(10, len(data) * 2.2), 7))
        bars = plt.bar(names, counts, color=color, width=0.6)
        plt.xticks(rotation=24, ha="right", fontsize=20)
        plt.yticks(fontsize=14)
        plt.ylabel("# Of Tickets Solved", fontsize=18)
        plt.xlabel("Top Contributors", fontsize=14, labelpad=40)

        for i, b in enumerate(bars):
            h = b.get_height()
            plt.text(
                b.get_x() + b.get_width() / 2,
                h - 0.05 * h,
                str(counts[i]),
                ha="center", va="top",
                fontsize=16, weight="bold", color="white"
            )

        plt.tight_layout(pad=2.0)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"Saved plot {outpath}")


# ----- Orchestrator / CLI App -----
class App:
    def __init__(self, config: Config):
        self.config = config
        self.gh = GitHubClient(config.repo, config.token)
        self.cache = CacheManager(config.cache_file, config.cache_ttl)
        self.ticket_mgr = TicketListManager(config.ticketlist_file, config.ticketlist_ttl)
        self.analyzer = PRAnalyzer(self.gh)

    def run(self, out_dir: str) -> None:
        if not self.config.repo or not self.config.token:
            raise SystemExit("Missing GITHUB_REPO or GITHUB_TOKEN in token.env")

        # Step 1: fetch closed PRs (merged ones)
        closed_prs = self.gh.list_closed_pulls()
        merged_prs = [pr for pr in closed_prs if pr.get("merged_at")]
        print(f"\nFound {len(merged_prs)} merged PRs out of {len(closed_prs)} closed PRs.")

        for pr in merged_prs:
            print(f"- #{pr.get('number')}: {pr.get('title')}")

        # Step 2: count approved PRs by author (from merged_prs list)
        approved_counts = self.analyzer.count_approved_prs_by_author(merged_prs)

        # Step 3: assign tiers and save CSV
        tiered_data = self.analyzer.assign_fixed_tiers(approved_counts)
        csv_out = os.path.join(out_dir, "approved_tickets", "approved_prs_by_tier.csv")
        CSVExporter.save(tiered_data, csv_out)

        # Step 4: plot tiers
        Plotter.plot_single_tier(tiered_data, "Gold", os.path.join(out_dir, "approved_tickets", "Gold_Tier.png"))
        Plotter.plot_single_tier(tiered_data, "Silver", os.path.join(out_dir, "approved_tickets", "Silver_Tier.png"))
        Plotter.plot_single_tier(tiered_data, "Bronze", os.path.join(out_dir, "approved_tickets", "Bronze_Tier.png"))

        # Step 5: build and save ticket list by author
        author_ticket_list = self.analyzer.ticket_list_by_author(merged_prs)
        ticket_json_path = self.ticket_mgr.save(author_ticket_list)
        print(f"Ticket list JSON saved to: {ticket_json_path}")

        # Step 6: find merged PRs that have >=1 approval via direct reviews check
        approved_prs = []
        for pr in merged_prs:
            prnum = pr.get("number")
            try:
                if prnum is not None and self.gh.pr_has_approval(prnum):
                    approved_prs.append(pr)
            except Exception:
                # conservative: skip PRs we can't verify
                continue
        print(f"[refresh] {len(approved_prs)} merged PRs have >=1 approval")

        # Step 7: cache results of approved PR counts
        author_counts = self.analyzer.count_approved_prs_by_author(approved_prs)
        if not author_counts:
            self.cache.save([])
            print("[refresh] No approved PRs found; wrote empty approved cache.")
        else:
            self.cache.save(author_counts)

        print("\nCompleted PR tier analysis and chart generation.")


# ----- CLI Entrypoint -----
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Gold, Silver, and Bronze contributor charts.")
    p.add_argument("--out-dir", type=str, default="commit_tiers_output", help="Directory to save outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_env()
    app = App(cfg)
    app.run(args.out_dir)


if __name__ == "__main__":
    main()
