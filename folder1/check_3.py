#!/usr/bin/env python3
"""
IPL Match Predictor — Pre-match prediction engine
=================================================
Usage:
    python3 ipl_predictor.py                          # interactive mode
    python3 ipl_predictor.py "MI" "CSK"               # quick predict
    python3 ipl_predictor.py "MI" "CSK" --venue "Wankhede Stadium"
    python3 ipl_predictor.py "MI" "CSK" --venue "Wankhede Stadium" --toss MI --toss-decision bat
    python3 ipl_predictor.py --list-teams             # show all teams in dataset

Signals used:
    1. ELO Rating          — dynamic skill rating updated after every match
    2. Head-to-Head        — direct record between the two teams
    3. Recent Form         — last 10 matches win rate (time-weighted)
    4. Venue Advantage     — each team's win % at the specified venue
    5. Toss Impact         — toss winner's historical win rate
    6. Season Momentum     — current-season win rate
    7. ML Ensemble         — Logistic Regression + Random Forest on engineered features
"""

import sys
import os
import warnings
import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
ELO_K          = 32          # ELO K-factor (higher = faster adaptation)
ELO_BASE       = 1500        # starting ELO for every team
FORM_WINDOW    = 10          # last N matches for recent form
FORM_DECAY     = 0.85        # weight decay per match going back in time
MIN_MATCHES    = 5           # minimum matches to trust a team stat

# Expected dataset column names (handles both Kaggle v1 and v2 formats)
COL_MAPS = {
    "team1":       ["team1"],
    "team2":       ["team2"],
    "winner":      ["winner"],
    "venue":       ["venue"],
    "toss_winner": ["toss_winner"],
    "toss_decision":["toss_decision"],
    "season":      ["season"],
    "date":        ["date"],
}

# ──────────────────────────────────────────────────────────────────────────────
# TEAM NAME ALIASES  (short name → canonical full name)
# ──────────────────────────────────────────────────────────────────────────────
TEAM_ALIASES = {
    "MI":   "Mumbai Indians",
    "CSK":  "Chennai Super Kings",
    "RCB":  "Royal Challengers Bangalore",
    "RCB":  "Royal Challengers Bengaluru",
    "KKR":  "Kolkata Knight Riders",
    "DC":   "Delhi Capitals",
    "DD":   "Delhi Daredevils",
    "SRH":  "Sunrisers Hyderabad",
    "PBKS": "Punjab Kings",
    "KXIP": "Kings XI Punjab",
    "RR":   "Rajasthan Royals",
    "GT":   "Gujarat Titans",
    "LSG":  "Lucknow Super Giants",
    "PWI":  "Pune Warriors",
    "RPS":  "Rising Pune Supergiants",
    "GL":   "Gujarat Lions",
    "KTK":  "Kochi Tuskers Kerala",
    "DKXIP":"Delhi & District Cricket Association Super Kings",
}

TEAM_NORMALIZE = {
    "Delhi Daredevils":               "Delhi Capitals",
    "Delhi & District Cricket Association Super Kings": "Delhi Capitals",
    "Kings XI Punjab":                "Punjab Kings",
    "Rising Pune Supergiant":         "Rising Pune Supergiants",
    "Royal Challengers Bangalore":    "Royal Challengers Bengaluru",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def resolve_col(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None

def normalize_team(name: str, known: set) -> str:
    """Try to match user-typed team name to a known team in the dataset."""
    if not name:
        return name
    # exact
    if name in known:
        return name
    # alias lookup
    upper = name.upper().strip()
    if upper in TEAM_ALIASES and TEAM_ALIASES[upper] in known:
        return TEAM_ALIASES[upper]
    # normalize renames
    for old, new in TEAM_NORMALIZE.items():
        if name.lower() == old.lower() and new in known:
            return new
    # case-insensitive partial match
    matches = [k for k in known if name.lower() in k.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # pick longest overlap
        matches.sort(key=lambda x: len(set(name.lower().split()) & set(x.lower().split())), reverse=True)
        return matches[0]
    return name

def expected_elo(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def update_elo(ra, rb, result_a):
    """result_a = 1 if A won, 0 if B won"""
    ea = expected_elo(ra, rb)
    ra_new = ra + ELO_K * (result_a - ea)
    rb_new = rb + ELO_K * ((1 - result_a) - (1 - ea))
    return ra_new, rb_new

def weighted_form(results, decay=FORM_DECAY):
    """results: list of 1/0 from most recent to oldest"""
    if not results:
        return 0.5
    weights = [decay**i for i in range(len(results))]
    return sum(r * w for r, w in zip(results, weights)) / sum(weights)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

class IPLData:
    def __init__(self, csv_path: str):
        self.df = self._load(csv_path)
        self.teams = set()
        self.elo = {}
        self.form = defaultdict(list)          # team → [1/0 recent to old]
        self.h2h = defaultdict(lambda: [0,0]) # (t1,t2) → [wins_t1, wins_t2]
        self.venue_wins = defaultdict(lambda: defaultdict(lambda: [0,0]))
        self.season_record = defaultdict(lambda: defaultdict(lambda: [0,0]))
        self.toss_win_rate = defaultdict(lambda: [0,0])  # [toss_and_win, toss_total]
        self.match_log = []   # ordered list for ML feature extraction

        self._build_stats()

    # ── loading ───────────────────────────────────────────────────────────────
    def _load(self, path):
        df = pd.read_csv(path)

        # Normalize known renames
        for col in ["team1", "team2", "winner", "toss_winner"]:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: TEAM_NORMALIZE.get(x, x) if isinstance(x, str) else x
                )

        # Parse date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        elif "season" in df.columns:
            df = df.sort_values("season").reset_index(drop=True)

        return df

    # ── stats building ────────────────────────────────────────────────────────
    def _build_stats(self):
        df = self.df

        # Collect all teams
        for col in ["team1", "team2"]:
            if col in df.columns:
                self.teams.update(df[col].dropna().unique())

        # Init ELO
        for t in self.teams:
            self.elo[t] = ELO_BASE

        for _, row in df.iterrows():
            t1 = row.get("team1")
            t2 = row.get("team2")
            winner = row.get("winner")
            venue  = row.get("venue", "Unknown")
            toss_w = row.get("toss_winner")
            season = row.get("season", "?")

            if not isinstance(t1, str) or not isinstance(t2, str):
                continue
            if not isinstance(winner, str) or winner.strip() == "" or pd.isna(winner):
                # no result / tie — skip for win stats, still log
                continue

            t1_won = (winner == t1)

            # Snapshot ELO BEFORE update (for ML features)
            snap = {
                "team1": t1, "team2": t2, "winner": winner,
                "venue": venue, "season": season,
                "toss_winner": toss_w,
                "elo_t1": self.elo[t1],
                "elo_t2": self.elo[t2],
                "form_t1": weighted_form(self.form[t1][:FORM_WINDOW]),
                "form_t2": weighted_form(self.form[t2][:FORM_WINDOW]),
                "h2h_t1_wins":   self.h2h[(t1,t2)][0],
                "h2h_t2_wins":   self.h2h[(t1,t2)][1],
                "venue_wr_t1":   self._venue_wr(t1, venue),
                "venue_wr_t2":   self._venue_wr(t2, venue),
                "toss_is_t1":    int(toss_w == t1),
                "t1_won":        int(t1_won),
            }
            self.match_log.append(snap)

            # Update ELO
            if t1_won:
                self.elo[t1], self.elo[t2] = update_elo(self.elo[t1], self.elo[t2], 1)
            else:
                self.elo[t1], self.elo[t2] = update_elo(self.elo[t1], self.elo[t2], 0)

            # Update form (prepend = most recent first)
            self.form[t1].insert(0, int(t1_won))
            self.form[t2].insert(0, int(not t1_won))

            # H2H
            if t1_won:
                self.h2h[(t1,t2)][0] += 1
                self.h2h[(t2,t1)][1] += 1
            else:
                self.h2h[(t1,t2)][1] += 1
                self.h2h[(t2,t1)][0] += 1

            # Venue wins
            if isinstance(venue, str):
                self.venue_wins[venue][t1][0 if t1_won else 1] += 1
                self.venue_wins[venue][t2][1 if t1_won else 0] += 1

            # Season record
            self.season_record[season][t1][0 if t1_won else 1] += 1
            self.season_record[season][t2][1 if t1_won else 0] += 1

            # Toss
            if isinstance(toss_w, str):
                self.toss_win_rate[venue][1] += 1
                if toss_w == winner:
                    self.toss_win_rate[venue][0] += 1

    def _venue_wr(self, team, venue):
        rec = self.venue_wins.get(venue, {}).get(team)
        if rec is None:
            return 0.5
        total = rec[0] + rec[1]
        return rec[0] / total if total > 0 else 0.5

    # ── ML model ──────────────────────────────────────────────────────────────
    def train_ml(self):
        if len(self.match_log) < 30:
            self.ml_model = None
            return

        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            records = self.match_log
            X = np.array([
                [
                    r["elo_t1"] - r["elo_t2"],
                    r["form_t1"] - r["form_t2"],
                    r["h2h_t1_wins"] - r["h2h_t2_wins"],
                    r["venue_wr_t1"] - r["venue_wr_t2"],
                    r["toss_is_t1"],
                    r["elo_t1"],
                    r["elo_t2"],
                    r["form_t1"],
                    r["form_t2"],
                ]
                for r in records
            ])
            y = np.array([r["t1_won"] for r in records])

            lr  = Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000, C=1.0))])
            rf  = RandomForestClassifier(n_estimators=300, max_depth=6,
                                         min_samples_leaf=5, random_state=42)
            gb  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.05, random_state=42)

            ensemble = VotingClassifier(
                estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
                voting="soft", weights=[1, 2, 2]
            )
            ensemble.fit(X, y)
            self.ml_model = ensemble
            print(f"  ✓ ML ensemble trained on {len(records)} historical matches.")
        except ImportError:
            print("  ⚠ scikit-learn not found — ML signal disabled. Install: pip install scikit-learn")
            self.ml_model = None

    def ml_probability(self, t1, t2, venue=None, toss_is_t1=0):
        if not hasattr(self, "ml_model") or self.ml_model is None:
            return None
        venue = venue or "Unknown"
        features = np.array([[
            self.elo[t1] - self.elo[t2],
            weighted_form(self.form[t1][:FORM_WINDOW]) - weighted_form(self.form[t2][:FORM_WINDOW]),
            self.h2h[(t1,t2)][0] - self.h2h[(t1,t2)][1],
            self._venue_wr(t1, venue) - self._venue_wr(t2, venue),
            toss_is_t1,
            self.elo[t1],
            self.elo[t2],
            weighted_form(self.form[t1][:FORM_WINDOW]),
            weighted_form(self.form[t2][:FORM_WINDOW]),
        ]])
        prob = self.ml_model.predict_proba(features)[0]
        return prob[1]  # probability that t1 wins

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self, data: IPLData):
        self.d = data

    def predict(self, team1: str, team2: str, venue: str = None,
                toss_winner: str = None, toss_decision: str = None) -> dict:
        d = self.d
        t1, t2 = team1, team2
        venue = venue or "Unknown"

        # ── 1. ELO probability ─────────────────────────────────────────────
        elo_t1, elo_t2 = d.elo.get(t1, ELO_BASE), d.elo.get(t2, ELO_BASE)
        elo_prob_t1 = expected_elo(elo_t1, elo_t2)

        # ── 2. Head-to-head ───────────────────────────────────────────────
        h2h = d.h2h[(t1, t2)]
        h2h_total = h2h[0] + h2h[1]
        if h2h_total >= MIN_MATCHES:
            h2h_prob_t1 = h2h[0] / h2h_total
        else:
            h2h_prob_t1 = 0.5 + (h2h[0] - h2h[1]) / max(h2h_total * 4, 1) * 0.5

        # ── 3. Recent form ────────────────────────────────────────────────
        form_t1 = weighted_form(d.form[t1][:FORM_WINDOW])
        form_t2 = weighted_form(d.form[t2][:FORM_WINDOW])
        form_total = form_t1 + form_t2
        form_prob_t1 = form_t1 / form_total if form_total > 0 else 0.5

        # ── 4. Venue advantage ────────────────────────────────────────────
        vwr_t1 = d._venue_wr(t1, venue)
        vwr_t2 = d._venue_wr(t2, venue)
        v_total = vwr_t1 + vwr_t2
        venue_prob_t1 = vwr_t1 / v_total if v_total > 0 else 0.5

        # ── 5. Toss impact ───────────────────────────────────────────────
        toss_bonus = 0.0
        toss_info  = ""
        if toss_winner in (t1, t2):
            tw_rec = d.toss_win_rate.get(venue, [0, 0])
            if tw_rec[1] > 0:
                toss_wr = tw_rec[0] / tw_rec[1]
            else:
                toss_wr = 0.5
            toss_bonus = (toss_wr - 0.5)  # positive if toss helps
            toss_info  = f"toss-winner advantage at this venue ≈ {toss_wr:.1%}"
            if toss_winner == t2:
                toss_bonus = -toss_bonus

        # ── 6. ML model probability ───────────────────────────────────────
        toss_is_t1 = 1 if toss_winner == t1 else 0
        ml_prob = d.ml_probability(t1, t2, venue, toss_is_t1)

        # ── Weighted ensemble ─────────────────────────────────────────────
        if ml_prob is not None:
            # Weights: ELO=20%, H2H=15%, Form=15%, Venue=10%, ML=40%
            weights = {"elo": 0.20, "h2h": 0.15, "form": 0.15, "venue": 0.10, "ml": 0.40}
            raw_t1 = (
                weights["elo"]   * elo_prob_t1   +
                weights["h2h"]   * h2h_prob_t1   +
                weights["form"]  * form_prob_t1  +
                weights["venue"] * venue_prob_t1 +
                weights["ml"]    * ml_prob
            )
        else:
            # No ML — ELO=35%, H2H=25%, Form=25%, Venue=15%
            weights = {"elo": 0.35, "h2h": 0.25, "form": 0.25, "venue": 0.15, "ml": 0.0}
            raw_t1 = (
                weights["elo"]   * elo_prob_t1   +
                weights["h2h"]   * h2h_prob_t1   +
                weights["form"]  * form_prob_t1  +
                weights["venue"] * venue_prob_t1
            )

        # Apply toss adjustment (capped at ±5%)
        toss_adj = max(-0.05, min(0.05, toss_bonus * 0.10))
        final_t1 = raw_t1 + toss_adj
        final_t1 = max(0.05, min(0.95, final_t1))
        final_t2 = 1.0 - final_t1

        return {
            "team1": t1, "team2": t2, "venue": venue,
            "win_prob_t1": final_t1,
            "win_prob_t2": final_t2,
            "predicted_winner": t1 if final_t1 > final_t2 else t2,
            "confidence": abs(final_t1 - 0.5) * 200,  # 0-100 scale
            "signals": {
                "elo":   {"t1": elo_t1,      "t2": elo_t2,      "prob_t1": elo_prob_t1},
                "h2h":   {"t1": h2h[0],      "t2": h2h[1],      "prob_t1": h2h_prob_t1,  "total": h2h_total},
                "form":  {"t1": form_t1,     "t2": form_t2,     "prob_t1": form_prob_t1},
                "venue": {"t1": vwr_t1,      "t2": vwr_t2,      "prob_t1": venue_prob_t1},
                "ml":    {"prob_t1": ml_prob},
                "toss":  {"info": toss_info, "adj": toss_adj},
            },
        }

# ──────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT
# ──────────────────────────────────────────────────────────────────────────────

def bar(prob, width=30):
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)

def print_prediction(result: dict):
    t1 = result["team1"]
    t2 = result["team2"]
    p1 = result["win_prob_t1"]
    p2 = result["win_prob_t2"]
    winner = result["predicted_winner"]
    conf   = result["confidence"]
    sig    = result["signals"]

    w = 60
    print()
    print("═" * w)
    print(f"  🏏  IPL MATCH PREDICTION")
    print("═" * w)
    print(f"  {t1}  vs  {t2}")
    if result["venue"] != "Unknown":
        print(f"  📍 Venue: {result['venue']}")
    print()

    print(f"  {'WIN PROBABILITY':─<{w-4}}")
    print(f"  {t1:<30} {p1:>6.1%}  {bar(p1, 20)}")
    print(f"  {t2:<30} {p2:>6.1%}  {bar(p2, 20)}")
    print()

    print(f"  {'SIGNAL BREAKDOWN':─<{w-4}}")
    print(f"  {'Signal':<16}  {'':>8}  {'':>8}  {'Team-1 edge':>14}")

    def sig_line(name, label1, label2, val1, val2, edge_prob):
        edge = (edge_prob - 0.5) * 200
        direction = "▲" if edge > 0 else ("▼" if edge < 0 else "–")
        print(f"  {name:<16}  {label1}: {val1:>5}   {label2}: {val2:>5}   {direction} {abs(edge):5.1f}%")

    sig_line("ELO Rating",
             "ELO", "ELO",
             f"{sig['elo']['t1']:.0f}", f"{sig['elo']['t2']:.0f}",
             sig["elo"]["prob_t1"])

    h = sig["h2h"]
    sig_line(f"H2H (n={h['total']})",
             "W", "W",
             f"{h['t1']}", f"{h['t2']}",
             h["prob_t1"])

    f = sig["form"]
    sig_line(f"Recent Form (L{FORM_WINDOW})",
             "FR", "FR",
             f"{f['t1']:.2f}", f"{f['t2']:.2f}",
             f["prob_t1"])

    v = sig["venue"]
    sig_line("Venue Win%",
             "VW%", "VW%",
             f"{v['t1']:.0%}", f"{v['t2']:.0%}",
             v["prob_t1"])

    if sig["ml"]["prob_t1"] is not None:
        mp = sig["ml"]["prob_t1"]
        edge = (mp - 0.5) * 200
        direction = "▲" if edge > 0 else ("▼" if edge < 0 else "–")
        print(f"  {'ML Ensemble':<16}  {'prob':>8}: {mp:.2%}   {direction} {abs(edge):5.1f}%")

    if sig["toss"]["info"]:
        print(f"\n  🎲 Toss: {sig['toss']['info']}")
        if sig["toss"]["adj"] != 0:
            who = t1 if sig["toss"]["adj"] > 0 else t2
            print(f"       Advantage adjusted toward {who} by {abs(sig['toss']['adj']):.1%}")

    print()
    print(f"  {'VERDICT':─<{w-4}}")

    conf_label = (
        "🔥 HIGH CONFIDENCE" if conf > 30 else
        "✅ MODERATE CONFIDENCE" if conf > 15 else
        "🤔 LOW CONFIDENCE (very close match)"
    )
    print(f"  Predicted Winner: ⭐ {winner}")
    print(f"  Win Probability : {p1 if winner == t1 else p2:.1%}")
    print(f"  Confidence Score: {conf:.1f}/100  {conf_label}")
    print("═" * w)
    print()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def find_csv():
    """Search common locations for the IPL CSV."""
    candidates = [
        "historical_ipl.csv",
        "ipl_matches.csv",
        "matches.csv",
        "ipl.csv",
        os.path.join(os.path.dirname(__file__), "historical_ipl.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser(
        description="IPL Match Predictor — predict pre-match outcome"
    )
    parser.add_argument("team1", nargs="?", help="Team 1 name or abbreviation")
    parser.add_argument("team2", nargs="?", help="Team 2 name or abbreviation")
    parser.add_argument("--venue",         default=None, help="Venue / stadium name")
    parser.add_argument("--toss",          default=None, help="Toss winner team")
    parser.add_argument("--toss-decision", default=None, choices=["bat","field"], help="Toss decision")
    parser.add_argument("--data",          default=None, help="Path to CSV file")
    parser.add_argument("--list-teams",    action="store_true", help="List all teams in dataset")
    parser.add_argument("--list-venues",   action="store_true", help="List all venues in dataset")
    args = parser.parse_args()

    # ── locate CSV ─────────────────────────────────────────────────────────
    csv_path = args.data or find_csv()
    if not csv_path or not os.path.exists(csv_path):
        print("❌  Could not find IPL CSV file.")
        print("    Place 'historical_ipl.csv' in the same folder as this script, or")
        print("    pass --data /path/to/file.csv")
        sys.exit(1)

    print(f"\n📂 Loading data from: {csv_path}")
    data = IPLData(csv_path)
    print(f"  ✓ {len(data.df)} matches loaded | {len(data.teams)} teams found")

    # ── list teams / venues ────────────────────────────────────────────────
    if args.list_teams:
        print("\n📋 Teams in dataset:")
        for t in sorted(data.teams):
            print(f"   • {t}")
        sys.exit(0)

    if args.list_venues:
        venues = data.df["venue"].dropna().unique() if "venue" in data.df.columns else []
        print("\n🏟  Venues in dataset:")
        for v in sorted(venues):
            print(f"   • {v}")
        sys.exit(0)

    print("  ⚙  Training ML ensemble …")
    data.train_ml()

    predictor = Predictor(data)

    # ── resolve team names ──────────────────────────────────────────────────
    def resolve(name):
        r = normalize_team(name, data.teams)
        if r not in data.teams:
            print(f"\n❌  Team '{name}' not found. Run with --list-teams to see all teams.")
            sys.exit(1)
        return r

    if args.team1 and args.team2:
        # Single prediction from CLI args
        t1 = resolve(args.team1)
        t2 = resolve(args.team2)
        result = predictor.predict(
            t1, t2,
            venue=args.venue,
            toss_winner=args.toss,
            toss_decision=args.toss_decision,
        )
        print_prediction(result)
    else:
        # ── Interactive loop ────────────────────────────────────────────────
        print("\n🏏  IPL Match Predictor — Interactive Mode")
        print("    Type team names or abbreviations (e.g. MI, CSK, RCB …)")
        print("    Commands: 'teams' | 'venues' | 'quit'\n")

        while True:
            try:
                raw = input("Enter match (Team1 vs Team2): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not raw:
                continue
            if raw.lower() in ("quit", "exit", "q"):
                break
            if raw.lower() == "teams":
                for t in sorted(data.teams):
                    print(f"  {t}")
                continue
            if raw.lower() == "venues":
                if "venue" in data.df.columns:
                    for v in sorted(data.df["venue"].dropna().unique()):
                        print(f"  {v}")
                continue

            # parse "X vs Y" or "X v Y" or "X, Y"
            for sep in [" vs ", " vs", " v ", " v\t", ",", "/"]:
                if sep in raw.lower():
                    parts = raw.split(sep, 1)
                    t1_raw, t2_raw = parts[0].strip(), parts[1].strip()
                    break
            else:
                parts = raw.split()
                if len(parts) >= 2:
                    t1_raw, t2_raw = parts[0], parts[-1]
                else:
                    print("  Format: 'Team1 vs Team2'")
                    continue

            t1 = normalize_team(t1_raw, data.teams)
            t2 = normalize_team(t2_raw, data.teams)

            if t1 not in data.teams:
                print(f"  ❌ '{t1_raw}' not found. Type 'teams' to see all.")
                continue
            if t2 not in data.teams:
                print(f"  ❌ '{t2_raw}' not found. Type 'teams' to see all.")
                continue

            # Optional venue
            venue_input = input("  Venue (press Enter to skip): ").strip() or None

            # Optional toss
            toss_input = input("  Toss winner (press Enter to skip): ").strip() or None
            toss_decision = None
            if toss_input:
                toss_winner = normalize_team(toss_input, data.teams)
                toss_decision = input("  Toss decision [bat/field] (Enter to skip): ").strip() or None
            else:
                toss_winner = None

            result = predictor.predict(t1, t2, venue=venue_input,
                                       toss_winner=toss_winner,
                                       toss_decision=toss_decision)
            print_prediction(result)


if __name__ == "__main__":
    main()
